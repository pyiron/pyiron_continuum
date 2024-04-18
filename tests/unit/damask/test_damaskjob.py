# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_base._tests import PyironTestCase
from pyiron_continuum import Project


class TestDecorators(PyironTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.project = Project('DAMASK')

    def test_list_solvers(self):
        job = self.project.create.job.DAMASK("damask")
        self.assertIsInstance(job.list_solvers(), list)

    @classmethod
    def tearDownClass(cls):
        cls.project.remove(enable=True)


if __name__ == "__main__":
    unittest.main()
