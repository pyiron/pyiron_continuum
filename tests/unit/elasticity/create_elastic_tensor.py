import numpy as np
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
