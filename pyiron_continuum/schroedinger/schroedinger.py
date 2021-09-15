# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for solving the time-independent Schroedinger equation on a discrete mesh.
"""

from pyiron_base import PythonTemplateJob, DataContainer, ImportAlarm
import numpy as np
from pyiron_continuum.schroedinger.mesh import RectMesh
from scipy.sparse.linalg import eigsh, LinearOperator
from abc import ABC, abstractmethod
from scipy.constants import physical_constants
import matplotlib.pyplot as plt

try:
    import k3d
    k3d_alarm = ImportAlarm()
except ImportAlarm:
    k3d_alarm = ImportAlarm(
        '3d plotting requires the k3d package which is accessible by `conda install -c conda-forge k3d`. Using k3d in '
        'a jupyter environment further requires the following shell commands: '
        '`jupyter nbextension install --py --sys-prefix k3d; jupyter nbextension enable --py --sys-prefix k3d`.'
    )


# TODO: Convert to pyiron units
HBAR = 1  # EV_TO_U_ANGSTROMSQ_PER_SSQ * physical_constants['Planck constant in eV/Hz'][0] / (2 * np.pi)
M_EL = 1  # physical_constants['electron mass in u'][0]
KB = physical_constants['Boltzmann constant in eV/K'][0]

__author__ = "Liam Huber"
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
    def __init__(self, init=None, table_name='input', lazy=False):
        super().__init__(init=init, table_name=table_name, lazy=lazy)
        self._mesh = None
        self._potential = None
        self.n_states = 1
        self.mass = M_EL

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
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, new_potential):
        self._potential = new_potential


class _TISEOutput(DataContainer):
    def __init__(self, init=None, table_name='input', lazy=False):
        super().__init__(init=init, table_name=table_name, lazy=lazy)
        self.energy = None
        self.psi = None
        self._rho = None

    @property
    def _space_axes(self):
        return tuple(n for n in range(1, len(self.psi.shape)))

    def _broadcast_vector_to_space(self, vector):
        return vector.reshape(-1, *(1,)*len(self._space_axes))

    def _normalize_per_state(self, states, ord=2):
        return states / self._broadcast_vector_to_space(np.linalg.norm(states, axis=self._space_axes, ord=ord))

    def _weight_states(self, states, weights):
        return states * self._broadcast_vector_to_space(weights)

    @property
    def rho(self):
        if self._rho is None and self.psi is not None:
            self._rho = self.psi ** 2
        return self._rho

    def get_boltzmann_occupation(self, temperature):
        if self.energy is not None:
            return np.exp(-self.energy / (KB * temperature))

    def get_boltzmann_psi(self, temperature):
        if self.psi is not None:
            weighted_psi = self._weight_states(self.psi, self.get_boltzmann_occupation(temperature))
            psi_tot = np.sum(weighted_psi, axis=0)
            return psi_tot / np.linalg.norm(psi_tot)

    def get_boltzmann_rho(self, temperature):
        if self.psi is not None:
            weighted_rho = self._weight_states(self.rho, self.get_boltzmann_occupation(temperature)**2)
            rho_tot = np.sum(weighted_rho, axis=0)
            return rho_tot / rho_tot.sum()

    def _boltzmann_rho_from_psi(self, temperature):
        """
        Equivalent to weighting rho directly since the psi are all orthogonal, just need a small weight change since
        rho = |psi|^2.
        """
        if self.psi is not None:
            weighted_psi = self._weight_states(self.psi, self.get_boltzmann_occupation(temperature))
            rho_tot = np.sum(weighted_psi ** 2, axis=0)
            return rho_tot / rho_tot.sum()


class TISE(PythonTemplateJob):
    """
    A class for solving the time-independent Schroedinger equation on discrete meshes.
    """

    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self._storage.input = _TISEInput()
        self._storage.output = _TISEOutput()

    @property
    def mesh(self):
        return self.input.mesh

    # @mesh.setter
    # def mesh(self, new_mesh):
    #     self.input.mesh = new_mesh

    @property
    def potential(self):
        return self.input.potential

    # @potential.setter
    # def potential(self, new_potential):
    #     self.input.potential = new_potential

    @property
    def plot_1d(self):
        return _Plot1D(self)

    @property
    def plot_2d(self):
        return _Plot2D(self)

    @property
    def plot_3d(self):
        return _Plot3D(self)

    def _potential_psi(self, psi):
        try:
            return self.potential(self.mesh) * psi
        except TypeError:
            return self.potential * psi

    def _hamiltonian(self, psi):
        """
        H = -(hbar^2 / 2 m)(del^2 / del x^2) + V

        Pyiron units:
            - m = atomic mass units
            - x = Angstroms
            - V = eV

        Thus, to get the first term to eV we need hbar in units of sqrt(eV Angstroms^2 u).
        Starting with hbar in eVs, we need to convert eV to (Angstroms^2 u / s^2) s.t. the first term comes out with eV.
        https://www.wolframalpha.com/input/?i=1+%28atomic+mass+units%29*%28angstroms%5E2%29%2F%28s%5E2%29+in+eV
        Except that didn't seem to work, so for now just use HBAR as defined in the header
        """
        return -(HBAR**2 / (2 * self.input.mass)) * self.mesh.laplacian(psi) + self._potential_psi(psi)

    def _flat_hamiltonian(self, psi_1d):
        """Matrix-vector product for `LinearOperator` to use."""
        return self._hamiltonian(psi_1d.reshape(self.mesh.shape[1:])).flatten()

    def run_static(self):
        self.status.running = True
        n_mat = np.prod(self.mesh.shape[1:])
        flat_ham = LinearOperator((n_mat, n_mat), self._flat_hamiltonian)

        eigenvalues, eigenvectors = eigsh(flat_ham, which='SA', k=self.input.n_states)
        self.output.energy = eigenvalues
        self.output.psi = np.array([np.reshape(v, self.mesh.shape[1:]) for v in eigenvectors.T])
        self.to_hdf()
        self.status.finished = True


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

    def boltzmann_psi(self, temperature, ax=None, **kwargs):
        return self._make_plot(self._job.output.get_boltzmann_psi(temperature), ax=ax, **kwargs)

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
        img = ax.contourf(data, **kwargs)
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
