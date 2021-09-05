import numpy as np
import unittest
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
from create_elastic_tensor import create_random_C


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
        medium = LinearElasticity(create_random_C(isotropic=True))
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

    def test_force(self):
        elastic_tensor = create_random_C()
        medium = LinearElasticity(elastic_tensor)
        medium.orientation = [[1, -2, 1], [1, 1, 1], [-1, 0, 1]]
        lattice_constant = 3.52
        partial_one = np.array([-0.5, 0, np.sqrt(3)/2])*lattice_constant
        partial_two = np.array([0.5, 0, np.sqrt(3)/2])*lattice_constant
        stress = medium.get_dislocation_stress([0, 10, 0], partial_one)
        force = medium.get_dislocation_force(stress, [0, 1, 0], partial_two)
        self.assertAlmostEqual(force[1], 0)
        self.assertAlmostEqual(force[2], 0)


if __name__ == "__main__":
    unittest.main()
