import numpy as np
import unittest
from pyiron_continuum.elasticity.eschelby import Eschelby
from pyiron_continuum.elasticity import tools


def create_random_HL(b=None):
    C = np.zeros((6, 6))
    C[:3, :3] = np.random.random()
    C[:3, :3] += np.random.random()*np.eye(3)
    C[3:, 3:] = np.random.random()*np.eye(3)
    C = tools.C_from_voigt(C)
    if b is None:
        b = np.random.random(3)
    return Eschelby(C, b)


class TestEschelby(unittest.TestCase):
    def test_p(self):
        hl = create_random_HL()
        self.assertTrue(
            np.allclose(np.absolute(np.linalg.det(hl._get_pmat(hl.p))), 0),
            'p-matrix has a full dimension'
        )

    def test_Ak(self):
        hl = create_random_HL()
        self.assertTrue(
            np.allclose(np.absolute(np.einsum('nk,nik->ni', hl.Ak, hl._get_pmat(hl.p))), 0),
            'Ak is not the kernel of the p-matrix'
        )

    def test_DAk(self):
        hl = create_random_HL()
        self.assertTrue(
            np.isclose(np.real(np.einsum('nk,n->k', hl.Ak, hl.D)), hl.burgers_vector).all(),
            'Magnitude not corresponding to the Burgers vector'
        )

    def test_Dq(self):
        hl = create_random_HL()
        F = np.einsum('n,ij->nij', hl.p, hl.elastic_tensor[:, 1, :, 1])
        F += hl.elastic_tensor[:, 1, :, 0]
        F = np.einsum('nik,nk->ni', F, hl.Ak)
        self.assertTrue(
            np.allclose(np.real(np.einsum('nk,n->k', F, hl.D)), 0),
            'Equilibrium condition not satisfied'
        )

    def test_displacement(self):
        hl = create_random_HL(b=[0, 0, 1])
        positions = (np.random.random((100, 2))-0.5)*10
        d_analytical = np.arctan2(*positions.T[:2][::-1])/2/np.pi
        self.assertTrue(
            np.all(np.absolute(hl.get_displacement(positions)[:, -1]-d_analytical) < 1.0e-4),
            'Screw dislocation displacement field not reproduced'
        )

    def test_strain(self):
        hl = create_random_HL(b=[0, 0, 1])
        positions = (np.random.random((100, 2))-0.5)*10
        strain_analytical = positions[:, 0]/np.sum(positions**2, axis=-1)/4/np.pi
        self.assertTrue(
            np.all(np.absolute(hl.get_strain(positions)[:, 1, 2]-strain_analytical) < 1.0e-4),
            'Screw dislocation strain field (yz-component) not reproduced'
        )
        strain_analytical = -positions[:,1]/np.sum(positions**2, axis=-1)/4/np.pi
        self.assertTrue(
            np.all(np.absolute(hl.get_strain(positions)[:, 0, 2]-strain_analytical) < 1.0e-4),
            'Screw dislocation strain field (xz-component) not reproduced'
        )


if __name__ == "__main__":
    unittest.main()
