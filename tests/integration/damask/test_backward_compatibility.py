import unittest
from pyiron_continuum import Project
import os
from damask import Rotation


class TestDamask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project("DAMASK_CHECK_ALL")
        cls.project.remove_jobs(recursive=True, silently=True)

    @classmethod
    def tearDownClass(cls):
        cls.project.remove(enable=True)

    def _get_load_step(self, dotF=1.0e-2):
        return [
            {
                "mech_bc_dict": {
                    "dot_F": [1e-3, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 10.0, "N": 40},
                "additional": {"f_out": 4},
            },
            {
                "mech_bc_dict": {
                    "dot_F": [dotF, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 60.0, "N": 60},
                "additional": {"f_out": 4},
            },
        ]

    def _get_grid(self, sd=16, n=4):
        return self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=sd, num_grains=n
        )

    def _get_homogenization(self):
        return self.project.continuum.damask.Homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        ),

    def test_creator(self):
        self.assertEqual(
            self.project.create.DAMASK.homogenization(
                method="SX",
                parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
            ),
            self._get_homogenization(),
        )

    def get_plasticity_phenopowerlaw(self):
        return self.project.continuum.damask.Plasticity(
            type="phenopowerlaw",
            N_sl=[12],
            a_sl=[2.25],
            atol_xi=1.0,
            dot_gamma_0_sl=[0.001],
            h_0_sl_sl=[75.0e6],
            h_sl_sl=[1, 1, 1.4, 1.4, 1.4, 1.4, 1.4],
            n_sl=[20],
            output=["xi_sl"],
            xi_0_sl=[31.0e6],
            xi_inf_sl=[63.0e6],
        )

    def _get_elasticity(self):
        return self.project.continuum.damask.Elasticity(
            type="Hooke", C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
        )

    def get_phase_aluminum(self, elasticity, plasticity):
        return self.project.continuum.damask.Phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e", "F_p", "L_p", "O"],
            elasticity=elasticity,
            plasticity=plasticity,
        )

    def test_damask_tutorial(self):
        grains = 8
        job = self.project.create.job.DAMASK("tutorial")
        plasticity = self.get_plasticity_phenopowerlaw()
        phase = self.get_phase_aluminum(self._get_elasticity(), plasticity)
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, self._get_homogenization()
        )
        job.material = material
        job.grid = self._get_grid(n=grains)
        solver = job.list_solvers()[0]
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=self._get_load_step(dotF=1.0e-3)
        )
        job.run()
        job.plot_stress_strain(component="zz")
        job.plot_stress_strain(von_mises=True)
        job.output.damask.view(increments=80)
        self.assertEqual(job.output.stress.shape[1:], (3, 3))

    def test_linear_elastic(self):
        job = self.project.create.job.DAMASK("linear_elastic")
        grains = 8
        phase = self.project.continuum.damask.Phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e"],
            elasticity=self._get_elasticity(),
            plasticity=None,
        )
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, self._get_homogenization()
        )
        job.material = material
        job.grid = self._get_grid(n=grains)
        load_step = [
            {
                "mech_bc_dict": {
                    "dot_F": [1e-3, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 50.0, "N": 50},
                "additional": {"f_out": 5},
            }
        ]
        solver = job.list_solvers()[0]  # choose the mechanis solver
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=load_step
        )
        job.run()
        job.plot_stress_strain(component="xx")
        job.plot_stress_strain(von_mises=True)

    def test_elastoplasticity_isotropic(self):
        job = self.project.create.job.DAMASK("elastoplasticity_isotropic")
        plasticity = self.project.continuum.damask.Plasticity(
            type="isotropic",
            dot_gamma_0=0.001,
            n=20.0,
            xi_0=0.85e6,
            xi_inf=1.6e6,
            a=2.0,
            h_0=5.0e6,  # hardening modulus
            M=1.0,
            h=1.0,
        )
        phase = self.get_phase_aluminum(self._get_elasticity(), plasticity)
        rotation = self.project.continuum.damask.Rotation(Rotation.from_random, 4)
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, self._get_homogenization()
        )
        job.material = material
        grid = self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=16, num_grains=4
        )
        job.grid = grid
        solver = job.list_solvers()[0]  # choose the mechanis solver
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=self._get_load_step()
        )
        job.run()

    def test_elastoplasticity_powerlaw(self):
        job = self.project.create.job.DAMASK("elastoplasticity_powerlaw")
        grains = 4
        plasticity = self.get_plasticity_phenopowerlaw()
        phase = self.get_phase_aluminum(self._get_elasticity(), plasticity)
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, self._get_homogenization()
        )
        job.material = material
        job.grid = self._get_grid()
        solver = job.list_solvers()[0]  # choose the mechanis solver
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=self._get_load_step()
        )
        job.run()
        job.plot_stress_strain(component="zz")
        job.plot_stress_strain(von_mises=True)

    def test_multiple_rolling(self):
        job = self.project.create.job.ROLLING("multiple_rolling")
        plasticity = self.get_plasticity_phenopowerlaw()
        grains = 4
        phase = self.get_phase_aluminum(self._get_elasticity(), plasticity)
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, self._get_homogenization()
        )
        job.material = material
        job.grid = self._get_grid(4, grains)
        reduction_height = 0.05
        reduction_speed = 5.0e-2
        reduction_outputs = 250
        regrid_flag = False
        damask_exe = ""  # using default DAMASK_grid solver from PATH
        job.executeRolling(
            reduction_height,
            reduction_speed,
            reduction_outputs,
            regrid_flag,
            damask_exe,
        )
        job.postProcess()  # do the postprocess
        job.plotStressStrainCurve(0.0, 0.60, 0.0, 1.7e8)  # xmin,xmax, ymin,ymax
        reduction_speed = 4.5e-2
        regrid_flag = True
        for reduction_height, reduction_outputs in zip(
            [0.1, 0.1, 0.12], [300, 350, 300]
        ):
            job.executeRolling(
                reduction_height,
                reduction_speed,
                reduction_outputs,
                regrid_flag,
                damask_exe,
            )
            job.postProcess()  # do the postprocess
            job.plotStressStrainCurve(0.0, 0.60, 0.0, 1.7e8)  # xmin,xmax, ymin,ymax


if __name__ == "__main__":
    unittest.main()
