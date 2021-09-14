import numpy as np
import unittest
from pyiron_continuum.elasticity.green import Anisotropic
from pyiron_continuum.elasticity import tools


def create_random_C(isotropic=False):
    C11_range = np.array([0.7120697386322292, 1.5435656086034886])
    coeff_C12 = np.array([0.65797601, -0.0199679])
    coeff_C44 = np.array([0.72753844, -0.30418746])
    C = np.zeros((6, 6))
    C11 = C11_range[0]+np.random.random()*C11_range.ptp()
    C12 = np.polyval(coeff_C12, C11)+0.2*(np.random.random()-0.5)
    C44 = np.polyval(coeff_C44, C11)+0.2*(np.random.random()-0.5)
    C[:3, :3] = C12
    C[:3, :3] += (C11-C12)*np.eye(3)
    if isotropic:
        C[3:, 3:] = np.eye(3)*(C[0, 0]-C[0, 1])/2
    else:
        C[3:, 3:] = C44*np.eye(3)
    return tools.C_from_voigt(C)


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
