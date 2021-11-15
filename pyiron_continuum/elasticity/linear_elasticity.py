# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_continuum.elasticity.green import Anisotropic, Isotropic, Green
from pyiron_continuum.elasticity.eschelby import Eschelby
from pyiron_continuum.elasticity import tools

__author__ = "Sam Waseda"
__copyright__ = "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "development"
__date__ = "Aug 21, 2021"


def value_or_none(func):
    def f(self):
        if self.elastic_tensor is None:
            return None
        else:
            return func(self)
    return f


point_defect_explanation = '''
According to the definition of the Green's function (cf. docstring of `get_greens_function`):

.. math:
    u_i(r) = \\sum_a G_{ij}(r-a)f_j(a)

where :math:`u_i(r)` is the displacement field of component :math:`i` at position :math:`r` and
:math:`f_j(a)` is the force component :math:`j` of the atom at position :math:`a`. By taking the
polynomial development we obtain:

.. math:
    u_i(r) \\approx G_{ij}(r)\\sum_a f_j(a)-\\frac{\\partial G_{ij}}{\\partial r_k}(r)\\sum_a a_k f_j(a)

The first term disappears because the sum of the forces is zero. From the second term we define
the dipole tensor :math:`P_{jk} = a_k f_j(a)`. Following the definition above, we can obtain the
displacement field, strain field, stress field and energy density field if the dipole tensor and
the elastic tensor are known.

The dipole tensor of a point defect is commonly obtained from the following equation:

.. math:
    U = \\frac{V}{2} \\varepsilon_{ij}C_{ijkl}\\varepsilon_{kl}-P_{kl}\\varepsilon_{kl}

where :math:`U` is the potential energy, :math:`V` is the volume and :math:`\\varepsilon` is the
strain field. At equilibrium, the derivative of the potential energy with respect to the strain
disappears:

.. math:
    P_{ij} = VC_{ijkl}\\varepsilon_{kl} = V\\sigma_{ij}

With this in mind, we can calculate the dipole tensor of Ni in Al with the following lines:

```python
from pyiron_atomistics import Project
pr = Project('dipole_tensor')
job = pr.create.job.Lammps('dipole')
n_repeat = 3
job.structure = pr.create.structure.bulk('Al', cubic=True).repeat(n_repeat)
job.structure[0] = 'Ni'
job.calc_minimize()
job.run()
dipole_tensor = -job.structure.get_volume()*job['output/generic/pressures'][-1]
```
'''


class LinearElasticity:
    """
    Linear elastic field class based on the 3x3x3x3 elastic tensor C_ijkl:

    sigma_ij = C_ijkl*epsilon_kl

    where sigma_ij is the ij-component of stress and epsilon_kl is the kl-component of strain.

    Examples I: Get bulk modulus from the elastic tensor:

    >>> medium = LinearElasticity(elastic_tensor)
    >>> print(medium.bulk_modulus)

    Example II: Get strain field around a point defect:

    >>> import numpy as np
    >>> medium = LinearElasticity(elastic_tensor)
    >>> random_positions = np.random.random((10, 3))-0.5
    >>> dipole_tensor = np.eye(3)
    >>> print(medium.get_point_defect_strain(random_positions, dipole_tensor))

    Example III: Get stress field around a dislocation:

    >>> import numpy as np
    >>> medium = LinearElasticity(elastic_tensor)
    >>> random_positions = np.random.random((10, 3))-0.5
    >>> burgers_vector = np.array([0, 0, 1])
    >>> print(medium.get_dislocation_stress(random_positions, burgers_vector))

    Example IV: Estimate the distance between partial dislocations:

    >>> medium = LinearElasticity(elastic_tensor)
    >>> partial_one = np.array([-0.5, 0, np.sqrt(3)/2])*lattice_constant
    >>> partial_two = np.array([0.5, 0, np.sqrt(3)/2])*lattice_constant
    >>> distance = 100
    >>> stress_one = medium.get_dislocation_stress([0, distance, 0], partial_one)
    >>> print('Choose `distance` in the way that the value below corresponds to SFE')
    >>> medium.get_dislocation_force(stress_one, [0, 1, 0], partial_two)

    """
    def __init__(self, elastic_tensor, orientation=None):
        """
        Args:

            elastic_tensor ((3,3,3,3)-, (6,6)- or (3,)-array): Elastic tensor (in C_ijkl notation,
                Voigt notation or a 3-component array containing [C_11, C_12, C_44]).

        """
        self.elastic_tensor = elastic_tensor
        self._isotropy_tolerance = 1.0e-4
        self._orientation = np.eye(3)
        if orientation is not None:
            self.orientation = orientation
        self._eschelby = None

    @property
    def orientation(self):
        """
        Rotation matrix that defines the orientation of the system. If set, the elastic tensor
        will be rotated accordingly. For example a box with a dislocation should get:

        ```
        >>> medium.orientation = np.array([[1, 1, 1], [1, 0, -1], [1, -2, 1]])
        ```

        If a non-orthogonal orientation is set, the second vector is orthogonalized with the Gram
        Schmidt process. It is not necessary to specify the third axis as it is automatically
        calculated.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, r):
        orientation = self._orientation.copy()
        orientation[:2] = r[:2]
        self._orientation = tools.orthonormalize(orientation)

    @property
    def _is_rotated(self):
        return np.isclose(np.einsum('ii->', self.orientation), 3)

    @property
    def elastic_tensor(self):
        """
        Elastic tensor. Regardless of whether it was given in the Voigt notation or in the
        full form, always the full tensor (i.e. (3,3,3,3)-array) is returned. For Voigt
        notation, use `elastic_tensor_voigt`
        """
        if self._elastic_tensor is None:
            self._update()
        if self._elastic_tensor is not None and not self._is_rotated:
            return np.einsum(
                'Ii,Jj,Kk,Ll,ijkl->IJKL',
                self.orientation,
                self.orientation,
                self.orientation,
                self.orientation,
                self._elastic_tensor,
                optimize=True
            )
        return self._elastic_tensor

    @elastic_tensor.setter
    def elastic_tensor(self, C):
        if C is not None:
            C = np.asarray(C)
            if C.shape != (6, 6) and C.shape != (3, 3, 3, 3) and C.shape != (3,):
                raise ValueError('Elastic tensor must be a (6,6), (3,3,3,3) or (3,)  array')
            if C.shape == (3,):
                C = tools.coeff_to_voigt(C)
            if C.shape == (6, 6):
                C = tools.C_from_voigt(C)
        self._elastic_tensor = C

    def _update(self):
        S = np.zeros((6, 6))
        S[:3, :3] = (np.eye(3)-self.poissons_ratio*(1-np.eye(3)))/self.youngs_modulus
        S[3:, 3:] = np.eye(3)/self.shear_modulus
        self.elastic_tensor = np.linalg.inv(S)

    @property
    @value_or_none
    def elastic_tensor_voigt(self):
        """
        Voigt notation of the elastic tensor, i.e. (i, j) = i, if i == j and
        (i, j) = 6-i-j if i!=j.
        """
        return tools.C_to_voigt(self.elastic_tensor)

    @property
    @value_or_none
    def compliance_matrix(self):
        """Compliance matrix in Voigt notation."""
        return np.linalg.inv(self.elastic_tensor_voigt)

    @property
    def zener_ratio(self):
        """
        Zener ratio or the anisotropy index. If 1, the medium is isotropic. If isotropic, the
        analytical form of the Green's function is used for the calculation of strain and
        displacement fields.
        """
        return 2*(1+self.poissons_ratio.mean())*self.shear_modulus.mean()/self.youngs_modulus.mean()

    @property
    def isotropy_tolerance(self):
        """
        Maximum tolerance deviation from 1 for the Zener ratio to determine whether the medium
        is isotropic or not.
        """
        return self._isotropy_tolerance

    @isotropy_tolerance.setter
    def isotropy_tolerance(self, value):
        if value < 0:
            raise ValueError('`isotropy_tolerance` must be a positive float')
        self._isotropy_tolerance = value

    @property
    def _is_isotropic(self):
        return np.absolute(self.zener_ratio-1) < self.isotropy_tolerance

    @property
    def shear_modulus(self):
        """
        Returns:
            ((3,)-array): yz-, xz-, xy-components of shear modulus
        """
        return 1/self.compliance_matrix[3:, 3:].diagonal()

    @property
    def bulk_modulus(self):
        """
        Returns:
            (float): Bulk modulus
        """
        return 3*(1-2*self.poissons_ratio.mean())/self.youngs_modulus.mean()

    @property
    def poissons_ratio(self):
        """
        Returns:
            ((3,)-array): yz-, xz-, xy-components of Poisson's ratio
        """
        nu = -self.compliance_matrix[:3, :3]*self.youngs_modulus
        return np.array([nu[1, 2], nu[0, 2], nu[0, 1]])

    @property
    def youngs_modulus(self):
        """
        Returns:
            ((3,)-array): xx-, yy-, zz-components of Young's modulus
        """
        return 1/self.compliance_matrix[:3, :3].diagonal()

    def get_greens_function(
        self, positions, derivative=0, fourier=False, n_mesh=100, isotropic=False, optimize=True
    ):
        """
        Green's function of the equilibrium condition:

        C_ijkl d^2u_k/dx_jdx_l = 0

        Args:
            positions ((n,3)-array): Positions in real space or reciprocal space (if fourier=True).
            derivative (int): 0th, 1st or 2nd derivative of the Green's function. Ignored if
                `fourier=True`.
            fourier (bool): If `True`,  the Green's function of the reciprocal space is returned.
            n_mesh (int): Number of mesh points in the radial integration in case if anisotropic
                Green's function (ignored if isotropic=True or fourier=True)
            isotropic (bool): Whether to use the isotropic or anisotropic elasticity. If the medium
                is isotropic, it will automatically be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,3)-array): Green's function values for the given positions
        """
        if isotropic or self._is_isotropic:
            C = Isotropic(self.poissons_ratio.mean(), self.shear_modulus.mean(), optimize=optimize)
        else:
            C = Anisotropic(self.elastic_tensor, n_mesh=n_mesh, optimize=optimize)
        return C.get_greens_function(positions, derivative, fourier)

    get_greens_function.__doc__ += Green.__doc__

    def get_point_defect_displacement(
        self, positions, dipole_tensor, n_mesh=100, isotropic=False, optimize=True
    ):
        """
        Displacement field around a point defect

        Args:
            positions ((n,3)-array): Positions in real space or reciprocal space (if fourier=True).
            dipole_tensor ((3,3)-array): Dipole tensor
            n_mesh (int): Number of mesh points in the radial integration in case if anisotropic
                Green's function (ignored if isotropic=True or fourier=True)
            isotropic (bool): Whether to use the isotropic or anisotropic elasticity. If the medium
                is isotropic, it will automatically be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,3)-array): Displacement field
        """
        g_tmp = self.get_greens_function(
            positions,
            derivative=1,
            fourier=False,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize
        )
        return -np.einsum('...ijk,...jk->...i', g_tmp, dipole_tensor)

    get_point_defect_displacement.__doc__ += point_defect_explanation

    def get_point_defect_strain(
        self, positions, dipole_tensor, n_mesh=100, isotropic=False, optimize=True
    ):
        """
        Strain field around a point defect using the Green's function method

        Args:
            positions ((n,3)-array): Positions in real space or reciprocal space (if fourier=True).
            dipole_tensor ((3,3)-array): Dipole tensor
            n_mesh (int): Number of mesh points in the radial integration in case if anisotropic
                Green's function (ignored if isotropic=True or fourier=True)
            isotropic (bool): Whether to use the isotropic or anisotropic elasticity. If the medium
                is isotropic, it will automatically be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,3,3)-array): Strain field
        """
        g_tmp = self.get_greens_function(
            positions,
            derivative=2,
            fourier=False,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize
        )
        v = -np.einsum('...ijkl,...kl->...ij', g_tmp, dipole_tensor)
        return 0.5*(v+np.einsum('...ij->...ji', v))

    get_point_defect_strain.__doc__ += point_defect_explanation

    def get_point_defect_stress(
        self, positions, dipole_tensor, n_mesh=100, isotropic=False, optimize=True
    ):
        """
        Stress field around a point defect using the Green's function method

        Args:
            positions ((n,3)-array): Positions in real space or reciprocal space (if fourier=True).
            dipole_tensor ((3,3)-array): Dipole tensor
            n_mesh (int): Number of mesh points in the radial integration in case if anisotropic
                Green's function (ignored if isotropic=True or fourier=True)
            isotropic (bool): Whether to use the isotropic or anisotropic elasticity. If the medium
                is isotropic, it will automatically be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,3,3)-array): Stress field
        """
        strain = self.get_point_defect_strain(
            positions=positions,
            dipole_tensor=dipole_tensor,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize
        )
        return np.einsum('ijkl,...kl->...ij', self.elastic_tensor, strain)

    get_point_defect_stress.__doc__ += point_defect_explanation

    def get_point_defect_energy_density(
        self, positions, dipole_tensor, n_mesh=100, isotropic=False, optimize=True
    ):
        """
        Energy density field around a point defect using the Green's function method

        Args:
            positions ((n,3)-array): Positions in real space or reciprocal space (if fourier=True).
            dipole_tensor ((3,3)-array): Dipole tensor
            n_mesh (int): Number of mesh points in the radial integration in case if anisotropic
                Green's function (ignored if isotropic=True or fourier=True)
            isotropic (bool): Whether to use the isotropic or anisotropic elasticity. If the medium
                is isotropic, it will automatically be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,)-array): Energy density field
        """
        strain = self.get_point_defect_strain(
            positions=positions,
            dipole_tensor=dipole_tensor,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize
        )
        return np.einsum('ijkl,...kl,...ij->...', self.elastic_tensor, strain, strain)

    get_point_defect_energy_density.__doc__ += point_defect_explanation

    def get_dislocation_displacement(self, positions, burgers_vector):
        """
        Displacement field around a dislocation according to anisotropic elasticity theory
        described by [Eschelby](https://doi.org/10.1016/0001-6160(53)90099-6).

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
                coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n, 3)-array): Displacement field (z-axis coincides with the dislocation line)
        """
        eschelby = Eschelby(self.elastic_tensor, burgers_vector)
        return eschelby.get_displacement(positions)

    def get_dislocation_strain(self, positions, burgers_vector):
        """
        Strain field around a dislocation according to anisotropic elasticity theory
        described by [Eschelby](https://doi.org/10.1016/0001-6160(53)90099-6).

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
                coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n, 3, 3)-array): Strain field (z-axis coincides with the dislocation line)
        """
        eschelby = Eschelby(self.elastic_tensor, burgers_vector)
        return eschelby.get_strain(positions)

    def get_dislocation_stress(self, positions, burgers_vector):
        """
        Stress field around a dislocation according to anisotropic elasticity theory
        described by [Eschelby](https://doi.org/10.1016/0001-6160(53)90099-6).

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
                coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n, 3, 3)-array): Stress field (z-axis coincides with the dislocation line)
        """
        strain = self.get_dislocation_strain(positions, burgers_vector)
        return np.einsum('ijkl,...kl->...ij', self.elastic_tensor, strain)

    def get_dislocation_energy_density(self, positions, burgers_vector):
        """
        Energy density field around a dislocation (product of stress and strain, cf. corresponding
        methods)

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
                coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n,)-array): Energy density field
        """
        strain = self.get_dislocation_strain(positions, burgers_vector)
        return np.einsum('ijkl,...kl,...ij->...', self.elastic_tensor, strain, strain)

    def get_dislocation_energy(self, burgers_vector, r_min, r_max, mesh=100):
        """
        Energy per unit length along the dislocation line.

        Args:
            burgers_vector ((3,)-array): Burgers vector
            r_min (float): Minimum distance from the dislocation core
            r_max (float): Maximum distance from the dislocation core
            mesh (int): Number of grid points for the numerical integration along the angle

        Returns:
            (float): Energy of dislocation per unit length

        The energy is defined by the product of the stress and strain (i.e. energy density),
        which is integrated over the plane vertical to the dislocation line. The energy density
        :math:`w` according to the linear elasticity is given by:

        .. math:
            w(r, \\theta) = A(\\theta)/r^2

        Therefore, the energy per unit length :math:`U` is given by:

        .. math:
            U = \\log(r_max/r_min)\\int A(\\theta)\\mathrm d\\theta

        This implies :math:`r_min` cannot be 0 as well as :math:`r_max` cannot be infinity. This
        is the consequence of the fact that the linear elasticity cannot describe the core
        structure properly, and a real medium is not infinitely large. While :math:`r_max` can
        be defined based on the real dislocation density, the choice of :math:`r_min` should be
        done carefully.
        """
        if r_min <= 0:
            raise ValueError('r_min must be a positive float')
        theta_range = np.linspace(0, 2*np.pi, 100, endpoint=False)
        r = np.stack((np.cos(theta_range), np.sin(theta_range)), axis=-1)*r_min
        strain = self.get_dislocation_strain(r, burgers_vector=burgers_vector)
        return np.einsum(
            'ijkl,nkl,nij->', self.elastic_tensor, strain, strain
        )/np.diff(theta_range)[0]*r_min**2*np.log(r_max/r_min)

    @staticmethod
    def get_dislocation_force(stress, glide_plane, burgers_vector):
        """
        Force per unit length along the dislocation line.

        Args:
            stress ((3,3)-array): External stress field at the dislocation line
            glide_plane ((3,)-array): Glide plane
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((3,)-array): Force per unit length acting on the dislocation.
        """
        g = np.asarray(glide_plane)/np.linalg.norm(glide_plane)
        return np.einsum('i,ij,j,k->k', g, stress, burgers_vector, np.cross(g, [0, 0, 1]))
