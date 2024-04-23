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

    def test_damask_tutorial(self):
        grains = 8
        grids = 16
        job = self.project.create.job.DAMASK("tutorial")
        homogenization = self.project.create.DAMASK.homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        )
        homogenization = self.project.continuum.damask.Homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        )
        elasticity = self.project.continuum.damask.Elasticity(
            type="Hooke", C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
        )
        plasticity = self.project.continuum.damask.Plasticity(
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
        phase = self.project.continuum.damask.Phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e", "F_p", "L_p", "O"],
            elasticity=elasticity,
            plasticity=plasticity,
        )
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, homogenization
        )
        job.material = material
        grid = self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=grids, num_grains=grains
        )
        job.grid = grid
        load_step = [
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
                    "dot_F": [1e-3, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 60.0, "N": 60},
                "additional": {"f_out": 4},
            },
        ]
        solver = job.list_solvers()[0]
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=load_step
        )
        job.run()
        job.plot_stress_strain(component="zz")
        job.plot_stress_strain(von_mises=True)
        job.output.damask.view(increments=80)
        self.assertEqual(job.output.stress.shape[1:], (3, 3))

    def test_linear_elastic(self):
        job = self.project.create.job.DAMASK("linear_elastic")
        grains = 8
        grids = 16
        elasticity = self.project.continuum.damask.Elasticity(
            type="Hooke", C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
        )
        phase = self.project.continuum.damask.Phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e"],
            elasticity=elasticity,
            plasticity=None,
        )
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        homogenization = self.project.continuum.damask.Homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        )
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, homogenization
        )
        job.material = material
        grid = self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=grids, num_grains=grains
        )
        job.grid = grid
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
<<<<<<< HEAD
        job = self.project.create.job.DAMASK("damask_job")
        elasticity = self.project.continuum.damask.Elasticity(
            type="Hooke", C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
        )
        plasticity = self.project.continuum.damask.Plasticity(
            type="isotropic",
            dot_gamma_0=0.001,
            n=20.0,
            xi_0=0.3e6,
            xi_inf=0.6e6,
            a=2.0,
            h_0=1.0e6,  # hardening modulus
            M=1.0,
            h=1.0,
            dilatation=True,
            output=["xi"],
        )
        phase = self.project.continuum.damask.Phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e", "F_p", "L_p", "O"],
            elasticity=elasticity,
            plasticity=plasticity,
        )
        rotation = self.project.continuum.damask.Rotation(Rotation.from_random, 4)
        homogenization = self.project.continuum.damask.Homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        )
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, homogenization
        )
        job.material = material
        grid = self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=16, num_grains=4
        )
        job.grid = grid
        load_step = [
            {
                "mech_bc_dict": {
                    "dot_F": [1e-2, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 20.0, "N": 100},
                "additional": {"f_out": 5},
            },
            {
                "mech_bc_dict": {
                    "dot_F": [1e-2, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 60.0, "N": 200},
                "additional": {"f_out": 5},
            },
        ]
        solver = job.list_solvers()[0]  # choose the mechanis solver
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=load_step
        )
        job.run()  # running your job, if you want the parallelization you can modify your 'pyiron/damask/bin/run_damask_3.0.0.sh file'
=======
        job.set_elasticity(type="Hooke", C_11=106.75e9, C_12=60.41e9, C_44=28.34e9)
        job.set_plasticity(
            type="isotropic",
            dot_gamma_0=0.001,
            n=20.0,
            xi_0=0.85e6,
            xi_inf=1.6e6,
            a=2.0,
            h_0=5.0e6,
            M=1.0,
            h=1.0,
        )
        job.set_phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e", "F_p", "L_p", "O"],
        )
        job.set_rotation(shape=4)
        job.set_homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        )
        job.set_elements("Aluminum")
        job.set_grid(
            method="voronoi_tessellation",
            box_size=1.0e-5,
            spatial_discretization=16,
            num_grains=4,
        )
        load_step = [
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
                    "dot_F": [1e-3, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 60.0, "N": 60},
                "additional": {"f_out": 4},
            },
        ]
        solver = job.list_solvers()[0]
        job.set_loading(solver=solver, load_steps=load_step)
        job.run()
        job.plot_stress_strain(component="xx")
        job.plot_stress_strain(von_mises=True)
>>>>>>> damask_create_integration_tests

    def test_elastoplasticity_powerlaw(self):
        job = self.project.create.job.DAMASK("elastoplasticity_powerlaw")
        grains = 4
        grids = 16
        elasticity = self.project.continuum.damask.Elasticity(
            type="Hooke", C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
        )
        plasticity = self.project.continuum.damask.Plasticity(
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
        phase = self.project.continuum.damask.Phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e", "F_p", "L_p", "O"],
            elasticity=elasticity,
            plasticity=plasticity,
        )
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        homogenization = self.project.continuum.damask.Homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        )
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, homogenization
        )
        job.material = material
        grid = self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=grids, num_grains=grains
        )
        job.grid = grid
        load_step = [
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
                    "dot_F": [1e-3, 0, 0, 0, "x", 0, 0, 0, "x"],
                    "P": ["x", "x", "x", "x", 0, "x", "x", "x", 0],
                },
                "discretization": {"t": 60.0, "N": 60},
                "additional": {"f_out": 4},
            },
        ]
        solver = job.list_solvers()[0]  # choose the mechanis solver
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=load_step
        )
        job.run()  # running your job, if you want the parallelization you can modify your 'pyiron/damask/bin/run_damask_3.0.0.sh file'
        job.plot_stress_strain(component="zz")
        job.plot_stress_strain(von_mises=True)

    def test_multiple_rolling(self):
        job = self.project.create.job.ROLLING("multiple_rolling")
        elasticity = self.project.continuum.damask.Elasticity(
            type="Hooke", C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
        )
        plasticity = self.project.continuum.damask.Plasticity(
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
        grains = 4
        grids = 4
        phase = self.project.continuum.damask.Phase(
            composition="Aluminum",
            lattice="cF",
            output_list=["F", "P", "F_e", "F_p", "L_p", "O"],
            elasticity=elasticity,
            plasticity=plasticity,
        )
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        homogenization = self.project.continuum.damask.Homogenization(
            method="SX",
            parameters={"N_constituents": 1, "mechanical": {"type": "pass"}},
        )
        material = self.project.continuum.damask.Material(
            [rotation], ["Aluminum"], phase, homogenization
        )
        job.material = material
        grid = self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=grids, num_grains=grains
        )
        job.grid = grid
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
        reduction_height = 0.1
        reduction_speed = 4.5e-2
        reduction_outputs = 300
        regrid_flag = True
        job.executeRolling(
            reduction_height,
            reduction_speed,
            reduction_outputs,
            regrid_flag,
            damask_exe,
        )
        job.postProcess()  # do the postprocess
        job.plotStressStrainCurve(0.0, 0.60, 0.0, 1.7e8)  # xmin,xmax, ymin,ymax
        reduction_height = 0.1
        reduction_speed = 4.5e-2
        reduction_outputs = 350
        regrid_flag = True
        job.executeRolling(
            reduction_height,
            reduction_speed,
            reduction_outputs,
            regrid_flag,
            damask_exe,
        )
        job.postProcess()  # do the postprocess
        job.plotStressStrainCurve(0.0, 0.60, 0.0, 1.7e8)  # xmin,xmax, ymin,ymax
        reduction_height = 0.12
        reduction_speed = 4.25e-2
        reduction_outputs = 300
        regrid_flag = True  # enable the regridding
        job.executeRolling(
            reduction_height,
            reduction_speed,
            reduction_outputs,
            regrid_flag,
            damask_exe,
        )
        job.postProcess()  # do the postprocess
        job.plotStressStrainCurve(0.0, 0.60, 0.0, 1.7e8)  # xmin,xmax, ymin,ymax
        print(job.ResultsFile)


if __name__ == "__main__":
    unittest.main()
