# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for solving the time-independent Schroedinger equation on a discrete periodic mesh for a single particle.
"""

from pyiron_base import PythonTemplateJob, DataContainer, ImportAlarm
import numpy as np
from pyiron_continuum.mesh import RectMesh
from scipy.sparse.linalg import eigsh, LinearOperator
from abc import ABC, abstractmethod
from scipy.constants import physical_constants
import matplotlib.pyplot as plt
from typing import Union, Type
from pyiron_continuum.schroedinger.potentials import Potential
from pyiron_base import ImportAlarm
with ImportAlarm(
        'shcrodinger functionality requires the `k3d` module (and its dependencies) specified as extra'
        'requirements. Please install it and try again.'
) as k3d_alarm:
    import k3d

HBAR = physical_constants['reduced Planck constant in eV s'][0]
KB = physical_constants['Boltzmann constant in eV/K'][0]
# conversion factor to convert the units of all terms in the Schroedinder equation are in eV. More documentation
# in _hamiltonian of the TISE class.
EV2_S2_PER_ANG2_PER_AMU_IN_EV = 9.64853322e27
# keep the default mass as electron mass in AMU.
M = physical_constants['electron mass in u'][0]

__author__ = "Liam Huber, Raynol Dsouza"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Sep 8, 2021"


class _TISEInput(DataContainer):
    """TISE custom input holder"""
    def __init__(self, init=None, table_name='input', lazy=False):
        super().__init__(init=init, table_name=table_name, lazy=lazy)
        self._mesh = None
        self._potential = None
        self.n_states = 1
        self.mass = M

    @property
    def mesh(self) -> RectMesh:
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh: RectMesh):
        if not isinstance(new_mesh, RectMesh):
            raise TypeError(f'Meshes must be of type {RectMesh} but got type {type(new_mesh)}')
        new_mesh.simplify_1d = False
        self._mesh = new_mesh

    @property
    def potential(self) -> Union[Type[Potential], np.ndarray]:
        return self._potential

    @potential.setter
    def potential(self, new_potential: Union[Type[Potential], np.ndarray]):
        self._potential = new_potential


class _TISEOutput(DataContainer):
    """TISE custom output holder"""
    def __init__(self, init=None, table_name='input', lazy=False):
        super().__init__(init=init, table_name=table_name, lazy=lazy)
        self.energy = None
        self.psi = None
        self._rho = None

    @property
    def _space_axes(self) -> tuple:
        return tuple(n for n in range(1, len(self.psi.shape)))

    def _broadcast_vector_to_space(self, vector: np.ndarray) -> np.ndarray:
        return vector.reshape(-1, *(1,)*len(self._space_axes))

    def _weight_states(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return states * self._broadcast_vector_to_space(weights)

    @property
    def rho(self) -> np.ndarray:
        if self._rho is None and self.psi is not None:
            self._rho = self.psi ** 2
        return self._rho

    def get_boltzmann_occupation(self, temperature: float) -> np.ndarray:
        if self.energy is not None:
            w = np.exp(-(self.energy - self.energy.min()) / (KB * temperature))
            return w / w.sum()

    def get_boltzmann_rho(self, temperature: float) -> np.ndarray:
        if self.psi is not None:
            w = self.get_boltzmann_occupation(temperature)
            return self._weight_states(self.rho, w).sum(axis=0)


class TISE(PythonTemplateJob):
    """
    A class for solving the time-independent Schroedinger equation on discrete, periodic meshes for a single particle.

    Input:
        mesh (pyiron_continuum.RectMesh): The 1-, 2-, or 3d mesh on which to solve. Assumes pyrion distance units.
        potential (pyiron_continuum.Potential | numpy.ndarray): The background potential for which to solve. A dedicated
            `Potential` can be provided, but any numpy array whose shape represents a scalar field on the mesh will
            work. Assumes pyiron energy units.
        n_states (int): The number of eigenstates for which to solve, starting with the ground state and working up.
            (default is 1, just the ground state.)
        mass (float): The mass of the particle. (Default is one electron mass.)

    Output:
        energy (numpy.ndarray): The eigen energy for each state.
        psi (numpy.ndarray): The eigen vector for each state, i.e. wave function, on the mesh.
        rho (numpy.ndarray): The probability density for each state, i.e. |psi|^2, on the mesh.

    Output methods:
        get_boltzmann_occupation: Given the temperature, returns the Boltzmann-weighted occupation probability for each
            states, normalized by the partition function. Note: if the most excited states have non-trivial occupations,
            you need to re-run the calculation solving for more states to get reliable results.
        get_boltzmann_rho: Given the temperature, returns the Boltzmann-weighted sum of occupation probabilities, i.e.
            the finite-temperature probability distribution for the system.

    Attributes:
        mesh (pyiron_continuum.RectMesh): read-only quick access to `.input.mesh`.
        potential (pyiron_continuum.Potential | numpy.ndarray): read-only quick access to `.input.potential`.
        plot_1/2/3d: hold quick-and-dirty plotting routines for various dimensions to get a quick peek at data. Some
            customization is possible by passing in existing axes and kwargs. Note that 3d plotting requires the `k3d`
            module.
    """

    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self.storage.input = _TISEInput()
        self.storage.output = _TISEOutput()

    @property
    def mesh(self) -> RectMesh:
        return self.input.mesh

    @property
    def potential(self) -> Union[Type[Potential], np.ndarray]:
        return self.input.potential

    @property
    def plot_1d(self):
        return _Plot1D(self)

    @property
    def plot_2d(self):
        return _Plot2D(self)

    @property
    def plot_3d(self):
        return _Plot3D(self)

    def _potential_psi(self, psi) -> np.ndarray:
        try:
            return self.potential(self.mesh) * psi
        except TypeError:
            return self.potential * psi

    def _hamiltonian(self, psi: np.ndarray) -> np.ndarray:
        """
        H = -(hbar^2 / 2 m)(del^2 / del x^2) + V

        Pyiron units:
            - m = atomic mass units
            - x = Angstroms
            - V = eV

        Thus, to get the first term to eV we need to multiply it by eV s / A**2 / u.
        The math is here: https://www.wolframalpha.com/input/?i=%28eV*s%29%5E2%2F%28AMU%29+%2F%28Angstrom%5E2%29+in+eV
        """
        return -(HBAR ** 2 / (2 * self.input.mass)) * self.mesh.laplacian(psi) * EV2_S2_PER_ANG2_PER_AMU_IN_EV + \
               self._potential_psi(psi)

    def _flat_hamiltonian(self, psi_1d: np.ndarray) -> np.ndarray:
        """Matrix-vector product for `LinearOperator` to use."""
        return self._hamiltonian(psi_1d.reshape(self.mesh.divisions)).flatten()

    def run_static(self):
        self.status.running = True
        n_mat = np.prod(self.mesh.divisions)
        flat_ham = LinearOperator((n_mat, n_mat), self._flat_hamiltonian)

        eigenvalues, eigenvectors = eigsh(flat_ham, which='SA', k=self.input.n_states)
        self.output.energy = eigenvalues
        self.output.psi = np.array([np.reshape(v, self.mesh.divisions) for v in eigenvectors.T])
        self.status.finished = True
        self.to_hdf()


class _PlotCore(ABC):
    def __init__(self, job):
        self._job = job

    @property
    @abstractmethod
    def _plot(self, ax, data, **kwargs):
        pass

    def _gen_ax(self, ax):
        if ax is None:
            return plt.subplots()
        else:
            return None, ax

    def _make_plot(self, data, ax=None, **kwargs):
        fig, ax = self._gen_ax(ax)
        ax = self._plot(ax, data, **kwargs)
        return fig, ax

    def potential(self, ax=None, **kwargs):
        try:
            return self._make_plot(self._job.potential(self._job.mesh), ax=ax, **kwargs)
        except TypeError:
            return self._make_plot(self._job.potential, ax=ax, **kwargs)

    def psi(self, i, ax=None, **kwargs):
        return self._make_plot(self._job.output.psi[i], ax=ax, **kwargs)

    def rho(self, i, ax=None, **kwargs):
        return self._make_plot(self._job.output.rho[i], ax=ax, **kwargs)

    def boltzmann_rho(self, temperature, ax=None, **kwargs):
        return self._make_plot(self._job.output.get_boltzmann_rho(temperature), ax=ax, **kwargs)


class _Plot1D(_PlotCore):
    def _plot(self, ax, data, shift=0, **kwargs):
        ax = ax.plot(data + shift, **kwargs)
        return ax

    def levels(self, n_states=None):
        if n_states is None:
            n_states = self._job.input.n_states
        fig, ax = plt.subplots()
        self.potential(ax=ax.twinx(), label='potl', color='k')
        for i in range(n_states):
            self.psi(i, shift=self._job.output.energy[i], ax=ax, label=i)
        ax.legend()
        return fig, ax


class _Plot2D(_PlotCore):
    def _plot(self, ax, data, **kwargs):
        img = ax.contourf(data.T, **kwargs)
        return img


class _Plot3D(_PlotCore):
    @k3d_alarm
    def _gen_ax(self, ax):
        if ax is None:
            return None, k3d.plot()
        else:
            return None, ax

    def _plot(self, ax, data, level=0.01, **kwargs):
        plt_surface = k3d.marching_cubes(scalar_field=data.astype(np.float32), level=level,
                                         bounds=self._job.mesh.bounds.flatten())
        ax += plt_surface
        plt_surface = k3d.marching_cubes(scalar_field=data.astype(np.float32), level=-level,
                                         bounds=self._job.mesh.bounds.flatten(), color=10)
        ax += plt_surface
        return ax
