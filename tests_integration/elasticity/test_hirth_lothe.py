import numpy as np
import unittest
from pyiron_continuum.elasticity.hirth_lothe import HirthLothe
from pyiron_continuum.elasticity.tools import *

def create_random_HL():
    C = np.zeros((6,6))
    C[:3,:3] = np.random.random()
    C[:3,:3] += np.random.random()*np.eye(3)
    C[3:,3:] = np.random.random()*np.eye(3)
    C = C_from_voigt(C)
    b = np.random.random(3)
    return HirthLothe(C, b)

class TestFenicsTutorials(unittest.TestCase):
    def test_p(self):
        hl = create_random_HL()
        self.assertTrue(np.isclose(np.absolute(np.linalg.det(hl.get_pmat(hl.p))), 0).all())

    def test_Ak(self):
        hl = create_random_HL()
        self.assertTrue(
            np.isclose(np.absolute(np.einsum('nk,nik->ni', hl.Ak, hl.get_pmat(hl.p))), 0).all()
        )

    def test_DAk(self):
        hl = create_random_HL()
        self.assertTrue(
            np.isclose(np.real(np.einsum('nk,n->k', hl.Ak, hl.D)), hl.burgers_vector).all()
        )

    def test_Dq(self):
        hl = create_random_HL()
        F = np.einsum('n,ij->nij', hl.p, hl.elastic_tensor[:,1,:,1])
        F += hl.elastic_tensor[:,1,:,0]
        F = np.einsum('nik,nk->ni', F, hl.Ak)
        self.assertTrue(
            np.isclose(np.real(np.einsum('nk,n->k', F, hl.D)), 0).all()
        )

if __name__ == "__main__":
    unittest.main()
