import numpy as np
import unittest
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
from pyiron_continuum.elasticity import tools

def create_random_C():
    C = np.zeros((6,6))
    C[:3,:3] = np.random.random()
    C[:3,:3] += np.random.random()*np.eye(3)
    C[3:,3:] = np.random.random()*np.eye(3)
    return tools.C_from_voigt(C)

class TestElasticity(unittest.TestCase):
    def test_frame(self):
        medium = LinearElasticity(np.random.random((6,6)))
        self.assertAlmostEqual(np.linalg.det(medium.frame), 1)
        medium.frame = np.random.random((3,3))
        self.assertAlmostEqual(np.linalg.det(medium.frame), 1)

    def test_rotation(self):
        elastic_tensor = create_random_C()
        epsilon = np.random.random((3,3))
        epsilon += epsilon.T
        sigma = np.einsum('ijkl,kl->ij', elastic_tensor, epsilon)
        medium = LinearElasticity(elastic_tensor)
        medium.frame = np.array([[1,1,1],[1,0,-1]])
        sigma = np.einsum('iI,jJ,IJ->ij', medium.frame, medium.frame, sigma)
        sigma_calc = np.einsum(
            'ijkl,kK,lL,KL->ij', medium.elastic_tensor, medium.frame, medium.frame, epsilon
        )
        self.assertTrue(np.isclose(sigma-sigma_calc, 0).all())

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
