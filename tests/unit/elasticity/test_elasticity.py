import numpy as np
import unittest
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
from pyiron_continuum.elasticity import tools


def create_random_C(isotropic=False):
    C11_range = np.array([0.7120697386322292, 1.5435656086034886])
    coeff_C12 = np.array([0.65797601, -0.0199679])
    coeff_C44 = np.array([0.72753844, -0.30418746])
    C = np.zeros((6, 6))
    C11 = C11_range[0] + np.random.random() * C11_range.ptp()
    C12 = np.polyval(coeff_C12, C11) + 0.2 * (np.random.random() - 0.5)
    C44 = np.polyval(coeff_C44, C11) + 0.2 * (np.random.random() - 0.5)
    C[:3, :3] = C12
    C[:3, :3] += (C11 - C12) * np.eye(3)
    if isotropic:
        C[3:, 3:] = np.eye(3) * (C[0, 0] - C[0, 1]) / 2
    else:
        C[3:, 3:] = C44 * np.eye(3)
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
        sigma = np.einsum("ijkl,kl->ij", elastic_tensor, epsilon)
        medium = LinearElasticity(elastic_tensor)
        medium.orientation = np.array([[1, 1, 1], [1, 0, -1]])
        sigma = np.einsum("iI,jJ,IJ->ij", medium.orientation, medium.orientation, sigma)
        sigma_calc = np.einsum(
            "ijkl,kK,lL,KL->ij",
            medium.elastic_tensor,
            medium.orientation,
            medium.orientation,
            epsilon,
        )
        self.assertTrue(np.allclose(sigma - sigma_calc, 0))

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
        r_max = 1e6 * np.random.random() + 10
        r_min_one = 10 * np.random.random()
        r_min_two = 10 * np.random.random()
        E_one = medium.get_dislocation_energy([0, 0, 1], r_min_one, r_max)
        E_two = medium.get_dislocation_energy([0, 0, 1], r_min_two, r_max)
        self.assertGreater(E_one, 0)
        self.assertGreater(E_two, 0)
        self.assertAlmostEqual(
            E_one / np.log(r_max / r_min_one), E_two / np.log(r_max / r_min_two)
        )

    def test_force(self):
        elastic_tensor = create_random_C()
        medium = LinearElasticity(elastic_tensor)
        medium.orientation = [[1, -2, 1], [1, 1, 1], [-1, 0, 1]]
        lattice_constant = 3.52
        partial_one = np.array([-0.5, 0, np.sqrt(3) / 2]) * lattice_constant
        partial_two = np.array([0.5, 0, np.sqrt(3) / 2]) * lattice_constant
        stress = medium.get_dislocation_stress([0, 10, 0], partial_one)
        force = medium.get_dislocation_force(stress, [0, 1, 0], partial_two)
        self.assertAlmostEqual(force[1], 0)
        self.assertAlmostEqual(force[2], 0)

    def test_elastic_tensor_input(self):
        C = create_random_C()
        medium = LinearElasticity([C[0, 0, 0, 0], C[0, 0, 1, 1], C[0, 1, 0, 1]])
        self.assertTrue(np.allclose(C, medium.elastic_tensor))

    def test_isotropy_tolerance(self):
        medium = LinearElasticity(create_random_C())
        with self.assertRaises(ValueError):
            medium.isotropy_tolerance = -1


if __name__ == "__main__":
    unittest.main()
