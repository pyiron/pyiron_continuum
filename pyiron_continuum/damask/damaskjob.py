# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
DAMASK job, which runs a damask simulation, and create the necessary inputs
"""

from pyiron_base import TemplateJob
from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "DAMASK functionality requires the `damask` module (and its dependencies) specified as extra"
    "requirements. Please install it and try again."
) as damask_alarm:
    from damask import Result, YAML, ConfigMaterial
from pyiron_continuum.damask import factory
import pyiron_continuum.damask.regrid as rgg
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

__author__ = "Muhammad Hassani"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Muhammad Hassani"
__email__ = "hassani@mpie.de"
__status__ = "development"
__date__ = "Oct 04, 2021"


class DAMASK(TemplateJob):
    def __init__(self, project, job_name):
        """
        The damask job
        Args:
            project(pyiron.project): a pyiron project
            job_name(str): the name of the job
        """
        super(DAMASK, self).__init__(project, job_name)
        self._rotation = None
        self.input.grid = None
        self._elements = None
        self._executable_activate(codename="damask")
        self.input.elasticity = None
        self.input.plasticity = None
        self.input.homogenization = None
        self.input.phase = None
        self.input.material = None
        self.input.loading = None
        self.input.job_names = []
        self.input.regrid = False
        self.input.regrid_scale = 1.025

    def _join_path(self, path, return_str=True):
        file_path = Path(self.working_directory) / path
        if return_str:
            return str(file_path)
        return file_path

    def set_elasticity(self, **kwargs):
        """
        Example:

        >>> job.set_elasticity(
        ...     type='Hooke', C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
        ... )

        """
        self.input.elasticity = kwargs

    def set_plasticity(self, **kwargs):
        """

        Example:

        >>> job.set_plasticity(
        ...     N_sl=[12],
        ...     a_sl=2.25,
        ...     atol_xi=1.0,
        ...     dot_gamma_0_sl=0.001,
        ...     h_0_sl_sl=75e6,
        ...     h_sl_sl=[1, 1, 1.4, 1.4, 1.4, 1.4],
        ...     n_sl=20,
        ...     output=['xi_sl'],
        ...     type='phenopowerlaw',
        ...     xi_0_sl=[31e6],
        ...     xi_inf_sl=[63e6]
        ... )
        """
        self.input.plasticity = kwargs

    def set_homogenization(self, **kwargs):
        """
        Args:
            method(str): homogenization method
            parameters(dict): the required parameters

        Example:
        >>> job.set_homogenization(
        ...     method='SX',
        ...     parameters={'N_constituents': 1, "mechanical": {"type": "pass"}}
        ... )
        """
        self.input.homogenization = factory.get_homogenization(**kwargs)

    def set_phase(self, composition, lattice=None, output_list=None):
        """

        Args:
            composition(str)
            lattice(dict)
            output_list(str)

        Returns:
            None

        Example:
        >>> job.set_elasticity(
        ...     type='Hooke', C_11= 106.75e9, C_12= 60.41e9, C_44=28.34e9
        ... )
        >>> job.set_plasticity(
        ...     N_sl=[12],
        ...     a_sl=2.25,
        ...     atol_xi=1.0,
        ...     dot_gamma_0_sl=0.001,
        ...     h_0_sl_sl=75e6,
        ...     h_sl_sl=[1, 1, 1.4, 1.4, 1.4, 1.4],
        ...     n_sl=20,
        ...     output=['xi_sl'],
        ...     type='phenopowerlaw',
        ...     xi_0_sl=[31e6],
        ...     xi_inf_sl=[63e6]
        ... )
        >>> job.set_phase(
        ...     composition='Aluminum',
        ...     lattice='cF',
        ...     output_list='[F, P, F_e, F_p, L_p, O]',
        ... )
        """
        if None in [self.input.elasticity, self.input.plasticity]:
            raise ValueError(
                "phase can only be defined after elasticity and plasticity are"
                " defined (cf. job.set_elasticity and job.set_plasticity)"
            )
        self.input.phase = factory.get_phase(
            elasticity=self.input.elasticity,
            plasticity=self.input.plasticity,
            composition=composition,
            lattice=lattice,
            output_list=output_list,
        )
        self._attempt_init_material()

    def set_rotation(self, method="from_random", *args, **kwargs):
        """
        Args:
            method (damask.Rotation.*/str): Method of damask.Rotation class
                which based on the given arguments creates the Rotation object.
                If string is given, it looks for the method within
                `damask.Rotation` via `getattr`.
        """
        self._rotation = [factory.get_rotation(method, *args, **kwargs)]
        self._attempt_init_material()

    @property
    def material(self):
        return self.input.material

    @material.setter
    def material(self, value):
        self.input.material = value

    def _attempt_init_material(self):
        data = {
            "rotation": self._rotation,
            "elements": self._elements,
            "phase": self.input.phase,
            "homogenization": self.input.homogenization,
        }
        if None not in data.values():
            self.input.material = factory.MaterialFactory.config(**data)

    def set_material(self, rotation, elements, phase, homogenization):
        """
        Args:
            rotation(damask.Rotation): damask rotation object
            elements(str): elements describing the phase
            phase(dict): a dictionary describing the phase parameters
            homogenization(dict): a dictionary describing the damask homogenization

        Returns:
            None
        """
        self.input.material = factory.MaterialFactory.config(
            rotation, elements, phase, homogenization
        )

    def set_elements(self, elements):
        self._elements = np.array([elements]).flatten().tolist()
        self._attempt_init_material()

    def set_grid(self, method="voronoi_tessellation", **kwargs):
        if method == "voronoi_tessellation":
            self.input.grid = factory.GridFactory.via_voronoi_tessellation(**kwargs)
        else:
            raise NotImplementedError(
                "Currently only `voronoi_tessellation` is implemented"
            )

    @property
    def grid(self):
        return self.input.grid

    @grid.setter
    def grid(self, grid):
        self.input.grid = grid

    @property
    def loading(self):
        return self.input.loading

    @loading.setter
    def loading(self, value):
        self.input.loading = value

    def set_loading(self, solver, load_steps):
        """
        Creates the required damask loading.

        Args:
            solver(dict): a dictionary describing the solver: e.g, {'mechanical': 'spectral_basic'}
            load_steps(list/single dict): a list of dict or single dict, which describes the loading conditions
                example:
                {'mech_bc_dict':{'dot_F':[1e-2,0,0, 0,'x',0,  0,0,'x'],
                                'P':['x','x','x', 'x',0,'x',  'x','x',0]},
                'discretization':{'t': 10.,'N': 40, 'f_out': 4},
                'additional': {'f_out': 4}
        """
        self.input.loading = factory.get_loading(solver=solver, load_steps=load_steps)

    def append_loading(self, load_steps):
        if not isinstance(load_steps, list):
            load_steps = [load_steps]
        self.input.loading["loadstep"].extend(factory.translate_load_steps(load_steps))

    def _write_material(self):
        if self.input.material is not None:
            self.input.material.save(self._join_path("material.yaml"))

    def _write_loading(self):
        if self.input.loading is not None:
            self.input.loading.save(self._join_path("loading.yaml"))

    def _write_geometry(self):
        if self.input.grid is not None:
            self.input.grid.save(self._join_path("damask"))

    def write_input(self):
        if self.input.regrid and len(self.input.job_names) > 0:
            self.input.grid = rgg.Regrid(
                self.input.grid, self.restart_file_list[0], self.input.regrid_scale
            ).grid
        self._write_loading()
        self._write_geometry()
        self._write_material()

    def collect_output(self):
        def _average(d):
            return np.average(list(d.values()), axis=1)

        results = self._load_results()
        self.output.stress = _average(results.get("sigma"))
        self.output.strain = _average(results.get("epsilon_V^0.0(F)"))
        self.output.stress_von_Mises = _average(results.get("sigma_vM"))
        self.output.strain_von_Mises = _average(results.get("epsilon_V^0.0(F)_vM"))
        self.to_hdf()

    def _load_results(self, file_name="damask_loading_material.hdf5", run_all=True):
        """
        loads the results from damask hdf file
        Args:
            file_name(str): path to the hdf file
        """

        def _average(d):
            return np.average(list(d.values()), axis=1)

        results = Result(self._join_path(file_name))
        if not run_all:
            return results
        results.add_stress_Cauchy()
        results.add_strain()
        results.add_equivalent_Mises("sigma")
        results.add_equivalent_Mises("epsilon_V^0.0(F)")
        return results

    def writeresults2vtk(self, file_name="damask_loading_material.hdf5"):
        """
        save results to vtk files
        """
        results = self._load_results(file_name=file_name, run_all=False)
        results.export_VTK(target_dir=self.working_directory)

    @staticmethod
    def list_solvers():
        """
        lists the solvers for a damask job
        """
        return [
            {"mechanical": "spectral_basic"},
            {"mechanical": "spectral_polarization"},
            {"mechanical": "FEM"},
        ]

    def plot_stress_strain(self, component=None, von_mises=False):
        """
        Plot the stress strain curve from the job file
        Parameters
        ----------
        direction(str): 'xx, xy, xz, yx, yy, yz, zx, zy, zz
        """
        fig, ax = plt.subplots()
        if component is not None:
            if von_mises is True:
                raise ValueError(
                    "It is not allowed that component is specified and von_mises is also True "
                )
            if len(component) != 2:
                ValueError("The length of direction must be 2, like 'xx', 'xy', ... ")
            if component[0] != "x" or component[0] != "y" or component[0] != "z":
                ValueError("The direction should be from x, y, and z")
            if component[1] != "x" or component[1] != "y" or component[1] != "z":
                ValueError("The direction should be from x, y, and z")
            _component_dict = {"x": 0, "y": 1, "z": 2}
            _zero_axis = int(_component_dict[component[0]])
            _first_axis = int(_component_dict[component[1]])
            ax.plot(
                self.output.strain[:, _zero_axis, _first_axis],
                self.output.stress[:, _zero_axis, _first_axis],
                linestyle="-",
                linewidth="2.5",
            )
            ax.grid(True)
            ax.set_xlabel(
                rf"$\varepsilon_{component[0]}$" + rf"$_{component[1]}$", fontsize=18
            )
            ax.set_ylabel(
                rf"$\sigma_{component[0]}$" + rf"$_{component[1]}$" + "(Pa)",
                fontsize=18,
            )
        elif von_mises is True:
            ax.plot(
                self.output.strain_von_Mises,
                self.output.stress_von_Mises,
                linestyle="-",
                linewidth="2.5",
            )
            ax.grid(True)
            ax.set_xlabel(r"$\varepsilon_{vM}$", fontsize=18)
            ax.set_ylabel(r"$\sigma_{vM}$ (Pa)", fontsize=18)
        else:
            raise ValueError(
                "either direction should be passed in "
                "or vonMises should be set to True"
            )
        return fig, ax

    def restart(self, job_name=None, job_type=None):
        new_job = super().restart(job_name=job_name, job_type=job_type)
        new_job.storage.input = self.storage.input.copy()
        new_job.input.material = ConfigMaterial(**new_job.input.material)
        new_job.input.loading = YAML(**self.input.loading)
        new_job.input.job_names.append(self.job_name)
        new_job.restart_file_list.append(
            self._join_path("damask_loading_material.hdf5")
        )
        return new_job
