# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
DAMASK job, which runs a damask simulation, and create the necessary inputs
"""

from pyiron_base import TemplateJob
from pyiron_continuum.damask.factory import ( MateriaFactory,
                                              GridFactory,
                                              DamaskLoading,
                                              LoadStep
                                              )
import numpy as np
import os


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
        self._damask_hdf = os.path.join(self.working_directory, "damask_loading.hdf5")
        self.create = Create()
        self._material = None
        self._loading = None
        self._grid = None
        self._results = None
        self.executable = "mpiexec -n 1 DAMASK_grid -l loading.yaml -g damask.vti"

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

    def _load_results(self, file_name="damask_loading.hdf5"):
        """
            loads the results from damask hdf file
            Args:
                file_name(str): path to the hdf file
        """
        if self._results is None:
            if file_name != "damask_loading.hdf5":
                self._damask_hdf = os.path.join(self.working_directory, file_name)
            self._results = Result(self._damask_hdf)
        return self._results

    @staticmethod
    def list_solvers():
        """
        lists the solvers for a damask job
        """
        return [{'mechanical': 'spectral_basic'},
                {'mechanical': 'spectral_polarization'},
                {'mechanical': 'FEM'}]


class Create:
    def __init__(self):
        """
        The create refactory for the damask job
        """
        self._grid = GridRefactory()

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @staticmethod
    def loading(solver, load_steps):
        """
        Creates the required damask loading
        Args:
            solver(dict): a dictionary desrcribing the solver: e.g, {'mechanical': 'spectral_basic'}
            load_steps(list/single dict): a list of dict or single dict, which describes the loading conditions
            an example would be:
            {'mech_bc_dict':{'dot_F':[1e-2,0,0, 0,'x',0,  0,0,'x'],
                            'P':['x','x','x', 'x',0,'x',  'x','x',0]},
            'discretization':{'t': 10.,'N': 40, 'f_out': 4},
            'additional': {'f_out': 4}
        """
        return DamaskLoading(solver=solver, load_steps=load_steps)

    @staticmethod
    def material(rotation, elements, phase, homogenization):
        """
        creates the damask material
        Args:
            rotation(damask.Rotation): damask rotation object
            elements(str): elements describing the phase
            phase(dict): a dictionary describing the phase parameters
            homogenization(dict): a dictionary describing the damask homogenization
        """
        return MaterialRefactory.config(rotation, elements, phase, homogenization)

    @staticmethod
    def homogenization(method, parameters):
        """
        returns damask homogenization as a dictionary
        Args:
            method(str): homogenization method
            parameters(dict): the required parameters
        Examples:
            homogenization(method='SX', parameters={'N_constituents': 1, "mechanical": {"type": "pass"}})
        """
        return {method: parameters}

    @staticmethod
    def phase(composition, lattice, output_list, elasticity, plasticity):
        """
        returns a dictionary describing the phases for damask
        Args:
            composition(str)
            lattice(dict)
            output_list(str)
            elasticity(dict)
            plasticity(dict)
        Examples:
            phase(composition='Aluminum', lattice= 'cF',
                  output_list='[F, P, F_e, F_p, L_p, O]',
                   elasticity=elasticity, plasticity=plasticity)
            # elasticity= elasticity(type= 'Hooke', C_11= 106.75e9,
                                        C_12= 60.41e9, C_44=28.34e9)
            #  plasticity = plasticity(N_sl=[12], a_sl=2.25,
                                    atol_xi=1.0, dot_gamma_0_sl=0.001,
                                    h_0_sl_sl=75e6,
                                    h_sl_sl=[1, 1, 1.4, 1.4, 1.4, 1.4],
                                    n_sl=20, output=['xi_sl'],
                                    type='phenopowerlaw', xi_0_sl=[31e6],
                                    xi_inf_sl=[63e6])
        """
        return {composition: {'lattice': lattice,
                              'mechanical': {'output': output_list,
                                             'elastic': elasticity,
                                             'plasticity': plasticity}}}

    @staticmethod
    def elasticity(**kwargs):
        """
        returns a dictionary of elasticity parameters for damask input file
        Examples:
             elasticity= elasticity(type= 'Hooke', C_11= 106.75e9,
                                        C_12= 60.41e9, C_44=28.34e9)
        """
        _elast = {}
        for key, val in kwargs.items():
            _elast[key] = val
        return _elast

    @staticmethod
    def plasticity(**kwargs):
        """
        returns a dictionary of plasticity parameters for damask input file
        Examples:
            plasticity = plasticity(N_sl=[12], a_sl=2.25,
                                    atol_xi=1.0, dot_gamma_0_sl=0.001,
                                    h_0_sl_sl=75e6,
                                    h_sl_sl=[1, 1, 1.4, 1.4, 1.4, 1.4],
                                    n_sl=20, output=['xi_sl'],
                                    type='phenopowerlaw', xi_0_sl=[31e6],
                                    xi_inf_sl=[63e6])
        """
        _plast = {}
        for key, val in kwargs.items():
            _plast[key] = val
        return _plast

    @staticmethod
    def rotation(method, *args):
        """
        returns a damask.Rotation object by a given method
        Args:
            method(damask.Rotation.*): a method of damask.Rotation class which based on the
                            given arguments creates the Rotation object
        """
        return method(*args)
