# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
A toolkit for managing extensions to the project from atomistics.
"""

from pyiron_base import Toolkit, Project, JobFactoryCore
from pyiron_continuum.fenics.job.generic import Fenics
from pyiron_continuum.fenics.job.elastic import FenicsLinearElastic
from pyiron_continuum.project import DAMASKCreator
from pyiron_continuum.schroedinger.schroedinger import TISE
from pyiron_continuum.mesh import RectMesh
from pyiron_continuum.schroedinger.potentials import Sinusoidal, SquareWell

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Sep 23, 2021"


class JobFactory(JobFactoryCore):
    @property
    def _job_class_dict(self) -> dict:
        return {
            'Fenics': Fenics,
            'FenicsLinearElastic': FenicsLinearElastic,
            # 'DAMASK': DAMASK,
            'TISE': TISE,
        }


class Potential:
    @property
    def Sinusoidal(self):
        return Sinusoidal

    @property
    def SquareWell(self):
        return SquareWell


class DAMASK:
    def __init__(self):
        """Initializer of damask python objects."""
        self._creator = DAMASKCreator()

    @property
    def Grid(self):
        return self._creator.grid

    @Grid.setter
    def Grid(self, value):
        self._creator.grid = value

    @staticmethod
    def Loading(solver, load_steps):
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
        return DAMASKCreator.loading(solver=solver, load_steps=load_steps)

    @staticmethod
    def Material(rotation, elements, phase, homogenization):
        """
        creates the damask material
        Args:
            rotation(damask.Rotation): damask rotation object
            elements(str): elements describing the phase
            phase(dict): a dictionary describing the phase parameters
            homogenization(dict): a dictionary describing the damask homogenization
        """
        return DAMASKCreator.material(rotation, elements, phase, homogenization)

    @staticmethod
    def Homogenization(method, parameters):
        """
        returns damask homogenization as a dictionary
        Args:
            method(str): homogenization method
            parameters(dict): the required parameters
        Examples:
            homogenization(method='SX', parameters={'N_constituents': 1, "mechanical": {"type": "pass"}})
        """
        return DAMASKCreator.homogenization(method, parameters)

    @staticmethod
    def Phase(composition, lattice, output_list, elasticity, plasticity):
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
        return DAMASKCreator.phase(composition, lattice, output_list, elasticity, plasticity)

    @staticmethod
    def Elasticity(**kwargs):
        """
        returns a dictionary of elasticity parameters for damask input file
        Examples:
             elasticity= elasticity(type= 'Hooke', C_11= 106.75e9,
                                        C_12= 60.41e9, C_44=28.34e9)
        """
        return DAMASKCreator.elasticity(**kwargs)

    @staticmethod
    def Plasticity(**kwargs):
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
        return DAMASKCreator.plasticity(**kwargs)

    @staticmethod
    def Rotation(method, *args):
        """
        returns a damask.Rotation object by a given method
        Args:
            method(damask.Rotation.*): a method of damask.Rotation class which based on the
                            given arguments creates the Rotation object
        """
        return method(*args)

class Schroedinger:
    @property
    def potential(self):
        return Potential()


class Mesh:
    @property
    def RectMesh(self):
        return RectMesh


class ContinuumTools(Toolkit):
    def __init__(self, project: Project):
        super().__init__(project)
        self._job = JobFactory(project)
        self._schroedinger = Schroedinger()
        self._damask = DAMASK()
        self._mesh = Mesh()

    @property
    def job(self) -> JobFactory:
        return self._job

    @property
    def schroedinger(self):
        return self._schroedinger

    @property
    def mesh(self):
        return self._mesh

    @property
    def damask(self):
        return self._damask

    @damask.setter
    def damask(self, value):
        self._damask = value
