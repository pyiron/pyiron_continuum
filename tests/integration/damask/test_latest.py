import unittest
from pyiron_continuum import Project
import os


class TestDamask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project("DAMASK_LATEST")
        cls.project.remove_jobs(recursive=True, silently=True)

    @classmethod
    def tearDownClass(cls):
        cls.project.remove(enable=True)

    def test_damask_tutorial(self):
        job = self.project.create.job.DAMASK("damask_job")
        grains=8
        grids=16 # defines the number of grains and grids
        homogenization = self.project.continuum.damask.Homogenization()
        elasticity_list = self.project.continuum.damask.list_elasticity()
        elasticity = self.project.continuum.damask.Elasticity(
            **elasticity_list["Hooke_Al"]
        )
        plasticity_list = self.project.continuum.damask.list_plasticity()
        plasticity = self.project.continuum.damask.Plasticity(
            **plasticity_list["phenopowerlaw_Al"]
        )
        phase = self.project.continuum.damask.Phase(
            composition='Aluminum', elasticity=elasticity, plasticity=plasticity
        )
        rotation = self.project.continuum.damask.Rotation(shape=grains)
        material = self.project.continuum.damask.Material(
            [rotation],['Aluminum'], phase, homogenization
        )
        job.material = material
        grid = self.project.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=grids, num_grains=grains
        )
        job.grid = grid
        key, value = self.project.continuum.damask.generate_loading_tensor("dot_F")
        value[0, 0] = 1e-3
        key[1, 1] = "P"
        key[2, 2] = "P"
        data = self.project.continuum.damask.loading_tensor_to_dict(key, value)
        load_step = [
            self.project.continuum.damask.generate_load_step(N=40, t=10, f_out=4, **data),
            self.project.continuum.damask.generate_load_step(N=60, t=60, f_out=4, **data),
        ]
        solver = job.list_solvers()[0]
        job.loading = self.project.continuum.damask.Loading(
            solver=solver, load_steps=load_step
        )
        job.run()
        self.assertEqual(job.output.stress.shape[1:], (3, 3))


if __name__ == "__main__":
    unittest.main()
