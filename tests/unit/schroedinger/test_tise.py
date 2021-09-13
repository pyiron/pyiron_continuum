# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import TestWithCleanProject
from pyiron_base import JOB_CLASS_DICT
JOB_CLASS_DICT['TISE'] = 'pyiron_continuum.schroedinger.schroedinger'


class TestTISE(TestWithCleanProject):
    def test_instantiation(self):
        job = self.project.create.job.TISE('tmp')
