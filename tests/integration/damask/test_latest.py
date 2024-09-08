import unittest
from pyiron_continuum import Project
import os
from damask import Rotation


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
        homogenization = pr.continuum.damask.Homogenization()
        elasticity_list = pr.continuum.damask.list_elasticity()
        elasticity = pr.continuum.damask.Elasticity(**elasticity_list["Hooke_Al"])
        plasticity_list = pr.continuum.damask.list_plasticity()
        plasticity = pr.continuum.damask.Plasticity(
            **plasticity_list["phenopowerlaw_Al"]
        )
        phase = pr.continuum.damask.Phase(
            composition='Aluminum', elasticity=elasticity, plasticity=plasticity
        )
        rotation = pr.continuum.damask.Rotation(shape=grains)
        material = pr.continuum.damask.Material(
            [rotation],['Aluminum'], phase, homogenization
        )
        job.material = material
        grid = pr.continuum.damask.Grid.via_voronoi_tessellation(
            box_size=1.0e-5, spatial_discretization=grids, num_grains=grains
        )
        job.grid = grid
        key, value = pr.continuum.damask.generate_loading_tensor("dot_F")
        value[0, 0] = 1e-3
        key[1, 1] = "P"
        key[2, 2] = "P"
        data = pr.continuum.damask.loading_tensor_to_dict(key, value)
        load_step = [
            pr.continuum.damask.generate_load_step(N=40, t=10, f_out=4, **data),
            pr.continuum.damask.generate_load_step(N=60, t=60, f_out=4, **data),
        ]
        solver = job.list_solvers()[0]
        job.loading = pr.continuum.damask.Loading(solver=solver, load_steps=load_step)
        job.run()
        self.assertEqual(job.output.stress.shape[1:], (3, 3))


if __name__ == "__main__":
    unittest.main()
