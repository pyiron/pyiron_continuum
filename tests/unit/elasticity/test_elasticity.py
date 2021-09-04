import numpy as np
import unittest
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
from pyiron_continuum.elasticity import tools


def create_random_C():
    C = np.zeros((6, 6))
    C[:3, :3] = np.random.random()
    C[:3, :3] += np.random.random()*np.eye(3)
    C[3:, 3:] = np.random.random()*np.eye(3)
    return tools.C_from_voigt(C)


class TestElasticity(unittest.TestCase):
    def test_frame(self):
        medium = LinearElasticity(np.random.random((6, 6)))
        self.assertAlmostEqual(np.linalg.det(medium.orientation), 1)
        medium.orientation = np.random.random((3, 3))
        self.assertAlmostEqual(np.linalg.det(medium.orientation), 1)

    def test_orientation(self):
        elastic_tensor = create_random_C()
        epsilon = np.random.random((3, 3))
        epsilon += epsilon.T
        sigma = np.einsum('ijkl,kl->ij', elastic_tensor, epsilon)
        medium = LinearElasticity(elastic_tensor)
        medium.orientation = np.array([[1, 1, 1], [1, 0, -1]])
        sigma = np.einsum('iI,jJ,IJ->ij', medium.orientation, medium.orientation, sigma)
        sigma_calc = np.einsum(
            'ijkl,kK,lL,KL->ij', medium.elastic_tensor, medium.orientation, medium.orientation, epsilon
        )
        self.assertTrue(np.allclose(sigma-sigma_calc, 0))

    def test_youngs_modulus(self):
        medium = LinearElasticity(np.eye(6))
        self.assertTrue(np.allclose(medium.youngs_modulus, 1))

    def test_poissons_ratio(self):
        medium = LinearElasticity(np.eye(6))
        self.assertTrue(np.allclose(medium.poissons_ratio, 0))

    def test_isotropic(self):
        C = np.zeros((6, 6))
        C[:3, :3] = np.ones((3, 3))*np.random.random()+np.eye(3)*np.random.random()
        C[3:, 3:] = np.eye(3)*(C[0, 0]-C[0, 1])/2
        medium = LinearElasticity(C)
        self.assertTrue(medium._is_isotropic)

    def test_energy(self):
        elastic_tensor = create_random_C()
        medium = LinearElasticity(elastic_tensor)
        r_max = 1e6*np.random.random()+10
        r_min_one = 10*np.random.random()
        r_min_two = 10*np.random.random()
        E_one = medium.get_dislocation_energy([0, 0, 1], r_min_one, r_max)
        E_two = medium.get_dislocation_energy([0, 0, 1], r_min_two, r_max)
        self.assertGreater(E_one, 0)
        self.assertGreater(E_two, 0)
        self.assertAlmostEqual(E_one/np.log(r_max/r_min_one), E_two/np.log(r_max/r_min_two))


if __name__ == "__main__":
    unittest.main()
