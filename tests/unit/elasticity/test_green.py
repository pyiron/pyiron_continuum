import numpy as np
import unittest
from pyiron_continuum.elasticity.green import Anisotropic
from create_elastic_tensor import create_random_C


class TestGreen(unittest.TestCase):
    def test_derivative(self):
        aniso = Anisotropic(create_random_C())
        positions = np.tile(np.random.random(3), 2).reshape(-1, 3)
        dz = 1.0e-6
        index = np.random.randint(3)
        positions[1, index] += dz
        G_an = aniso.get_greens_function(positions.mean(axis=0), derivative=1)[:, index, :]
        G_num = np.diff(aniso.get_greens_function(positions), axis=0)/dz
        self.assertTrue(np.isclose(G_num-G_an, 0).all())
        G_an = aniso.get_greens_function(positions.mean(axis=0), derivative=2)[:, :, :, index]
        G_num = np.diff(aniso.get_greens_function(positions, derivative=1), axis=0)/dz
        self.assertTrue(np.isclose(G_num-G_an, 0).all())


if __name__ == "__main__":
    unittest.main()
