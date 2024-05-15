# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_base._tests import PyironTestCase
from pyiron_continuum import Project


class TestDamask(PyironTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.project = Project('DAMASK')

    def test_plasticity(self):
        job = self.project.create.job.DAMASK("damask")
        self.assertEqual(job.input.plasticity, None)
        job.set_plasticity(hello="goodbye")
        self.assertEqual(job.input.plasticity["hello"], "goodbye")

    def test_elasticity(self):
        job = self.project.create.job.DAMASK("damask")
        self.assertEqual(job.input.elasticity, None)
        job.set_elasticity(hello="goodbye")
        self.assertEqual(job.input.elasticity["hello"], "goodbye")

    def test_list_solvers(self):
        job = self.project.create.job.DAMASK("damask")
        self.assertIsInstance(job.list_solvers(), list)

    def test_set_phase(self):
        job = self.project.create.job.DAMASK("damask")
        self.assertEqual(job.input.phase, None)
        with self.assertRaises(ValueError):
            job.set_phase(
                composition='Aluminum', lattice='cF', output_list='[F, P, F_e, F_p, L_p, O]'
            )
        job.set_elasticity(
            type='Hooke', C_11= 106.75e9, C_12= 60.41e9, C_44=28.34e9
        )
        job.set_plasticity(
            N_sl=[12],
            a_sl=2.25,
            atol_xi=1.0,
            dot_gamma_0_sl=0.001,
            h_0_sl_sl=75e6,
            h_sl_sl=[1, 1, 1.4, 1.4, 1.4, 1.4],
            n_sl=20,
            output=['xi_sl'],
            type='phenopowerlaw',
            xi_0_sl=[31e6],
            xi_inf_sl=[63e6]
        )
        job.set_phase(
            composition='Aluminum',
            lattice='cF',
            output_list='[F, P, F_e, F_p, L_p, O]',
        )
        self.assertIsInstance(job.input.phase, dict)

    @classmethod
    def tearDownClass(cls):
        cls.project.remove(enable=True)


if __name__ == "__main__":
    unittest.main()
