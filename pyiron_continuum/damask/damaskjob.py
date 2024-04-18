# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
DAMASK job, which runs a damask simulation, and create the necessary inputs
"""

from pyiron_base import TemplateJob, ImportAlarm

with ImportAlarm(
        'DAMASK functionality requires the `damask` module (and its dependencies) specified as extra'
        'requirements. Please install it and try again.'
) as damask_alarm:
    from damask import Result
from pyiron_continuum.damask.factory import Create as DAMASKCreator, GridFactory
import os
import numpy as np
import matplotlib.pyplot as plt

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
        self._results = None
        self._rotation = None
        self._geometry = None
        self._elements = None
        self._executable_activate()
        self.input.elasticity = None
        self.input.plasticity = None
        self.input.homogenization = None
        self.input.phase = None
        self.input.material = None

    def set_elasticity(self, **kwargs):
        self.input.elasticity = DAMASKCreator.elasticity(**kwargs)

    def set_plasticity(self, **kwargs):
        self.input.plasticity = DAMASKCreator.plasticity(**kwargs)

    def set_homogenization(self, **kwargs):
        self.input.homogenization = DAMASKCreator.homogenization(**kwargs)

    def set_phase(self, **kwargs):
        if None not in [self.input.elasticity, self.input.plasticity]:
            self.input.phase = DAMASKCreator.phase(
                elasticity=self.input.elasticity,
                plasticity=self.input.plasticity,
                **kwargs
            )
            self._attempt_init_material()

    def set_rotation(self, method, *args):
        self._rotation = [DAMASKCreator.rotation(method, *args)]
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
            "homogenization": self.input.homogenization
        }
        if None not in data.values():
            self.input.material = DAMASKCreator.material(**data)

    def set_material(self, rotation, elements, phase, homogenization):
        self.input.material = DAMASKCreator.material(
            rotation, elements, phase, homogenization
        )

    def set_elements(self, elements):
        self._elements = np.array([elements]).flatten().tolist()
        self._attempt_init_material()

    def set_grid(self, method="voronoi_tessellation", **kwargs):
        if method == "voronoi_tessellation":
            self._geometry = GridFactory.via_voronoi_tessellation(**kwargs)

    @property
    def grid(self):
        return self._geometry

    @grid.setter
    def grid(self, grid):
        self._geometry = grid

    @property
    def loading(self):
        return self.input.loading

    @loading.setter
    def loading(self, value):
        self.input.loading = value

    def set_loading(self, **kwargs):
        self.input.loading = DAMASKCreator.loading(**kwargs)
        self._attempt_init_material()

    def _write_material(self):
        file_path = os.path.join(self.working_directory, "material.yaml")
        self.input.material.save(fname=file_path)

    def _write_loading(self):
        file_path = os.path.join(self.working_directory, "loading.yaml")
        self.input.loading.save(file_path)

    def _write_geometry(self):
        file_path = os.path.join(self.working_directory, "damask")
        self._geometry.save(file_path)

    def write_input(self):
        self._write_loading()
        self._write_geometry()
        self._write_material()

    def collect_output(self):
        self._load_results()
        self.output.damask = self._results

    def _load_results(self, file_name="damask_loading_material.hdf5"):
        """
            loads the results from damask hdf file
            Args:
                file_name(str): path to the hdf file
        """
        damask_hdf = os.path.join(self.working_directory, file_name)

        def _average(d):
            return np.average(list(d.values()), axis=1)

        self._results = Result(damask_hdf)
        self._results.add_stress_Cauchy()
        self._results.add_strain()
        self._results.add_equivalent_Mises('sigma')
        self._results.add_equivalent_Mises('epsilon_V^0.0(F)')
        self.output.stress = _average(self._results.get('sigma'))
        self.output.strain = _average(self._results.get('epsilon_V^0.0(F)'))
        self.output.stress_von_Mises = _average(self._results.get('sigma_vM'))
        self.output.strain_von_Mises = _average(self._results.get('epsilon_V^0.0(F)_vM'))

    def writeresults2vtk(self):
        """
            save results to vtk files
        """
        if self._results is None:
            raise ValueError("Results not loaded; call collect_output")
        self._results.export_VTK(target_dir=self.working_directory)

    @staticmethod
    def list_solvers():
        """
        lists the solvers for a damask job
        """
        return [{'mechanical': 'spectral_basic'},
                {'mechanical': 'spectral_polarization'},
                {'mechanical': 'FEM'}]

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
                raise ValueError("It is not allowed that component is specified and von_mises is also True ")
            if len(component) != 2:
                ValueError("The length of direction must be 2, like 'xx', 'xy', ... ")
            if component[0] != 'x' or component[0] != 'y' or component[0] != 'z':
                ValueError("The direction should be from x, y, and z")
            if component[1] != 'x' or component[1] != 'y' or component[1] != 'z':
                ValueError("The direction should be from x, y, and z")
            _component_dict = {'x': 0, 'y': 1, 'z': 2}
            _zero_axis = int(_component_dict[component[0]])
            _first_axis = int(_component_dict[component[1]])
            ax.plot(self.output.strain[:, _zero_axis, _first_axis],
                    self.output.stress[:, _zero_axis, _first_axis],
                    linestyle='-', linewidth='2.5')
            ax.grid(True)
            ax.set_xlabel(rf'$\varepsilon_{component[0]}$' + rf'$_{component[1]}$', fontsize=18)
            ax.set_ylabel(rf'$\sigma_{component[0]}$' + rf'$_{component[1]}$' + '(Pa)', fontsize=18)
        elif von_mises is True:
            ax.plot(self.output.strain_von_Mises, self.output.stress_von_Mises,
                    linestyle='-', linewidth='2.5')
            ax.grid(True)
            ax.set_xlabel(r'$\varepsilon_{vM}$', fontsize=18)
            ax.set_ylabel(r'$\sigma_{vM}$ (Pa)', fontsize=18)
        else:
            raise ValueError("either direction should be passed in "
                             "or vonMises should be set to True")
        return fig, ax
