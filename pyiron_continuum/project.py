# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
# import warnings
from pyiron_base import JobTypeChoice, Project as ProjectCore
from pyiron_base import Creator as CreatorCore, PyironFactory, ImportAlarm
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
with ImportAlarm(
        'DAMASK functionality requires the `damask` module (and its dependencies) specified as extra'
        'requirements. Please install it and try again.'
) as damask_alarm:
    from pyiron_continuum.damask.factory import Create as DAMASKCreator
    from pyiron_continuum.damask.factory import GridFactory

try:
    from pyiron_base import ProjectGUI
except (ImportError, TypeError, AttributeError):
    pass


__author__ = "Joerg Neugebauer, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Damask(PyironFactory):
    def __init__(self):
        self.create = DAMASKCreator()

    @property
    def grid(self):
        return self.create.grid

    @grid.setter
    def grid(self, value):
        self.create.grid = value

    @staticmethod
    def loading(solver, load_steps):
        """
        Creates the required damask loading.
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
    def material(rotation, elements, phase, homogenization):
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
        return DAMASKCreator.phase(composition, lattice, output_list, elasticity, plasticity)

    @staticmethod
    def elasticity(**kwargs):
        """
        returns a dictionary of elasticity parameters for damask input file
        Examples:
             elasticity= elasticity(type= 'Hooke', C_11= 106.75e9,
                                        C_12= 60.41e9, C_44=28.34e9)
        """
        return DAMASKCreator.elasticity(**kwargs)

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
        return DAMASKCreator.plasticity(**kwargs)

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
        return DAMASKCreator.homogenization(method=method,parameters=parameters)

    @staticmethod
    def rotation(method, *args):
        """
        returns a damask.Rotation object by a given method
        Args:
            method(damask.Rotation.*): a method of damask.Rotation class which based on the
                            given arguments creates the Rotation object
        """
        return method(*args)

class Project(ProjectCore):
    def __init__(self, path="", user=None, sql_query=None, default_working_directory=False):
        super(Project, self).__init__(
            path=path,
            user=user,
            sql_query=sql_query,
            default_working_directory=default_working_directory
        )
        self.job_type = JobTypeChoice()
        self._creator = Creator(self)
        # TODO: instead of re-initialzing, auto-update pyiron_base creator with factories, like we update job class
        #  creation

    def load_from_jobpath(self, job_id=None, db_entry=None, convert_to_object=True):
        """
        Internal function to load an existing job either based on the job ID or based on the database entry dictionary.

        Args:
            job_id (int): Job ID - optional, but either the job_id or the db_entry is required.
            db_entry (dict): database entry dictionary - optional, but either the job_id or the db_entry is required.
            convert_to_object (bool): convert the object to an pyiron object or only access the HDF5 file - default=True
                                      accessing only the HDF5 file is about an order of magnitude faster, but only
                                      provides limited functionality. Compare the GenericJob object to JobCore object.

        Returns:
            GenericJob, JobCore: Either the full GenericJob object or just a reduced JobCore object
        """
        job = super(Project, self).load_from_jobpath(
            job_id=job_id, db_entry=db_entry, convert_to_object=convert_to_object
        )
        job.project_hdf5._project = self.__class__(path=job.project_hdf5.file_path)
        return job

    def load_from_jobpath_string(self, job_path, convert_to_object=True):
        """
        Internal function to load an existing job either based on the job ID or based on the database entry dictionary.

        Args:
            job_path (str): string to reload the job from an HDF5 file - '/root_path/project_path/filename.h5/h5_path'
            convert_to_object (bool): convert the object to an pyiron object or only access the HDF5 file - default=True
                                      accessing only the HDF5 file is about an order of magnitude faster, but only
                                      provides limited functionality. Compare the GenericJob object to JobCore object.

        Returns:
            GenericJob, JobCore: Either the full GenericJob object or just a reduced JobCore object
        """
        job = super(Project, self).load_from_jobpath_string(
            job_path=job_path, convert_to_object=convert_to_object
        )
        job.project_hdf5._project = Project(path=job.project_hdf5.file_path)
        return job

    # Graphical user interfaces
    def gui(self):
        """

        Returns:

        """
        ProjectGUI(self)

Project.__doc__ = ProjectCore.__doc__


class MediumFactory(PyironFactory):
    @staticmethod
    def elastic_medium(elastic_tensor=None):
        return LinearElasticity(elastic_tensor)


class Creator(CreatorCore):

    def __init__(self, project):
        super().__init__(project)
        self._medium = MediumFactory()
        self._damask_creator = Damask()
        self._grid = GridFactory()

    @property
    def medium(self):
        return self._medium

    @property
    def DAMASK(self):
        return self._damask_creator

    @DAMASK.setter
    def DAMASK(self, value):
        self._damask_creator = value


