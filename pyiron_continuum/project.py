# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
import warnings
from pyiron_base import JobTypeChoice, Project as ProjectCore
from pyiron_base import Creator as CreatorCore, PyironFactory
from pyiron_snippets.import_alarm import ImportAlarm
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
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
        warnings.warn(
            "Setting loading via project creator is deprecated. Use job.set_loading instead"
        )
        return DAMASKCreator.loading(solver=solver, load_steps=load_steps)

    @staticmethod
    def material(rotation, elements, phase, homogenization):
        warnings.warn(
            "Setting material via project creator is deprecated. Use job.set_material instead"
        )
        return DAMASKCreator.material(rotation, elements, phase, homogenization)

    @staticmethod
    def phase(composition, lattice, output_list, elasticity, plasticity):
        warnings.warn(
            "Setting phase via project creator is deprecated. Use job.set_phase instead"
        )
        return DAMASKCreator.phase(
            composition, lattice, output_list, elasticity, plasticity
        )

    @staticmethod
    def elasticity(**kwargs):
        warnings.warn(
            "Setting elasticity via project creator is deprecated. Use job.set_elasticity instead"
        )
        return DAMASKCreator.elasticity(**kwargs)

    @staticmethod
    def plasticity(**kwargs):
        warnings.warn(
            "Setting plasticity via project creator is deprecated. Use job.set_plasticity instead"
        )
        return DAMASKCreator.plasticity(**kwargs)

    @staticmethod
    def homogenization(method, parameters):
        warnings.warn(
            "Setting homogenization via project creator is deprecated. Use job.set_homogenization instead"
        )
        return DAMASKCreator.homogenization(method=method, parameters=parameters)

    @staticmethod
    def rotation(method, *args, **kwargs):
        warnings.warn(
            "Setting rotation via project creator is deprecated. Use job.set_rotation instead"
        )
        return DAMASKCreator.rotation(method=method, *args, **kwargs)


class Project(ProjectCore):
    def __init__(
        self, path="", user=None, sql_query=None, default_working_directory=False
    ):
        super(Project, self).__init__(
            path=path,
            user=user,
            sql_query=sql_query,
            default_working_directory=default_working_directory,
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
