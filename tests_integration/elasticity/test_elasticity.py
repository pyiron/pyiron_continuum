import numpy as np
import unittest
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity

class TestFenicsTutorials(unittest.TestCase):
    def test_frame(self):
        medium = LinearElasticity(np.random.random((6,6)))
        self.assertAlmostEqual(np.linalg.det(medium.frame), 1)
        medium.frame = np.random.random((3,3))
        self.assertAlmostEqual(np.linalg.det(medium.frame), 1)

    def test_not_initialized(self):
        medium = LinearElasticity()
        self.assertIsNone(medium.bulk_modulus)

    def test_youngs_modulus(self):
        medium = LinearElasticity(np.eye(6))
        self.assertAlmostEqual(medium.youngs_modulus, 1)

    def test_poissons_ratio(self):
        medium = LinearElasticity(np.eye(6))
        self.assertAlmostEqual(medium.poissons_ratio, 0)

    def test_isotropic(self):
        medium = LinearElasticity()
        medium.poissons_ratio = 0.3
        medium.youngs_modulus = 1
        self.assertTrue(medium._is_isotropic)

if __name__ == "__main__":
    unittest.main()
