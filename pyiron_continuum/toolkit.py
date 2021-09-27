# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
A toolkit for managing extensions to the project from atomistics.
"""

from pyiron_base import Toolkit, Project, JobFactoryCore
from pyiron_continuum.fenics.job.generic import Fenics
from pyiron_continuum.fenics.job.elastic import FenicsLinearElastic
# from pyiron_continuum.damask.damaskjob import DAMASK  # Looks like Damask needs an import alarm
from pyiron_continuum.schroedinger.schroedinger import TISE
from pyiron_continuum.schroedinger.mesh import RectMesh
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


class Schroedinger:
    @property
    def RectMesh(self):
        return RectMesh

    @property
    def potential(self):
        return Potential()


class ContinuumTools(Toolkit):
    def __init__(self, project: Project):
        super().__init__(project)
        self._job = JobFactory(project)
        self._schroedinger = Schroedinger()

    @property
    def job(self) -> JobFactory:
        return self._job

    @property
    def schroedinger(self):
        return self._schroedinger

