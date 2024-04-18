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
    from damask import Result as ResultDamask
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
        self._material = None
        self._loading = None
        self._grid = None
        self._results = None
        self._executable_activate()

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        self._material = value

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value):
        self._loading = value

    def _write_material(self):
        file_path = os.path.join(self.working_directory, "material.yaml")
        self._material.save(fname=file_path)
        self.input.material = self._material

    def _write_loading(self):
        file_path = os.path.join(self.working_directory, "loading.yaml")
        self._loading.save(file_path)
        self.input.loading = self._loading

    def _write_geometry(self):
        file_path = os.path.join(self.working_directory, "damask")
        self._grid.save(file_path)
        self.input.geometry = self._grid

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
