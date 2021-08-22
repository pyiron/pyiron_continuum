# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import Settings
from pyiron_continuum.elasticity.green import Anisotropic, Isotropic
from pyiron_continuum.elasticity.hirth_lothe import HirthLothe

__author__ = "Jan Janssen"
__copyright__ = "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Feb 20, 2020"

s = Settings()


def index_from_voigt(i, j):
    if i==j:
        return i
    else:
        return 6-i-j

def C_from_voigt(C_in):
    C = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = C_in[index_from_voigt(i,j), index_from_voigt(k,l)]
    return C

def C_to_voigt(C_in):
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(i+1):
            for k in range(3):
                for l in range(k+1):
                    C[index_from_voigt(i,j), index_from_voigt(k,l)] = C_in[i,j,k,l]
    return C

def value_or_none(func):
    def f(self):
        if self.elastic_tensor is None:
            return None
        else:
            return func(self)
    return f

def is_initialized(func):
    def f(self):
        if self._elastic_tensor is None:
            v = np.sum([param is not None for param in [
                self._lame_coefficient,
                self._shear_modulus,
                self._bulk_modulus,
                self._poissons_ratio,
                self._youngs_modulus
            ]])
            if v < 2:
                return None
        return func(self)
    return f

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
    >>> print(medium.get_strain_field(random_positions, dipole_tensor))

    """
    def __init__(self, elastic_tensor=None):
        """
        Args:

            elastic_tensor ((3,3,3,3)- or (6,6)-array): Elastic tensor (in C_ijkl notation or
                Voigt notation).

        It is not mandatory to set the `elastic_tensor` during the initialization. Instead, at
        least two of the elastic constants (such as `youngs_modulus`, `poissons_ratio` or
        `shear_modulus`) can be set and `elastic_tensor` will be calculated automatically.
        """
        self.elastic_tensor = elastic_tensor
        self._isotropy_tolerance = 1.0e-4
        self._frame = np.eye(3)
        self._lame_coefficient = None
        self._shear_modulus = None
        self._bulk_modulus = None
        self._poissons_ratio = None
        self._youngs_modulus = None
        self._hirth_lothe = None

    @property
    def frame(self):
        """
        Rotation matrix that defines the orientation of the system. If set, the elastic tensor
        and (optionally) the dipole tensor will be rotated.
        """
        return self._frame

    @frame.setter
    def frame(self, f):
        """
        Rotation matrix that defines the orientation of the system. If set, the elastic tensor
        and (optionally) the dipole tensor will be rotated.
        """
        frame = self._frame.copy()
        frame[:2] = f[:2]
        frame = (frame.T/np.linalg.norm(frame, axis=-1).T).T
        frame[2] = np.cross(frame[0], frame[1])
        if np.isclose(np.linalg.det(frame), 0):
            raise ValueError('Vectors not independent')
        self._frame = np.einsum('ij,i->ij', frame, 1/np.linalg.norm(frame, axis=-1))

    @property
    def _is_rotated(self):
        return np.isclose(np.einsum('ii->', self.frame), 3)

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
            return np.einsum('iI,IjKl,kK->ijkl', self.frame, self._elastic_tensor, self.frame)
        return self._elastic_tensor

    @elastic_tensor.setter
    def elastic_tensor(self, C):
        if C is not None:
            C = np.asarray(C)
            if C.shape != (6, 6) and C.shape != (3, 3, 3, 3):
                raise ValueError('Elastic tensor must be a (6,6) or (3,3,3,3) array')
            if C.shape == (6, 6):
                C = C_from_voigt(C)
        self._elastic_tensor = C

    def _update(self):
        S = np.zeros((6,6))
        S[:3,:3] = (np.eye(3)-self.poissons_ratio*(1-np.eye(3)))/self.youngs_modulus
        S[3:,3:] = np.eye(3)/self.shear_modulus
        self.elastic_tensor = np.linalg.inv(S)

    @property
    @value_or_none
    def elastic_tensor_voigt(self):
        """
        Voigt notation of the elastic tensor, i.e. (i,j) = i, if i==j and
        (i,j) = 6-i-j if i!=j.
        """
        return C_to_voigt(self.elastic_tensor)

    @property
    @value_or_none
    def compliance_matrix(self):
        return np.linalg.inv(self.elastic_tensor_voigt)

    @property
    @is_initialized
    def zener_ratio(self):
        """
        Zener ratio or the anistropy index. If 1, the medium is isotropic. If isotropic, the
        analytical form of the Green's function is used for the calculation of strain and
        displacement fields.
        """
        return 2*(1+self.poissons_ratio)*self.shear_modulus/self.youngs_modulus

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
    @is_initialized
    def _is_isotropic(self):
        return np.absolute(self.zener_ratio-1) < self.isotropy_tolerance

    @property
    @is_initialized
    def lame_coefficient(self):
        """
        Lame's first parameter. It is calculated either from Young's modulus and Poisson's ratio
        or Young's modulus and bulk modulus (depending on what is available)
        """
        if self._lame_coefficient is None:
            if self.youngs_modulus is not None:
                if self.poissons_ratio is not None:
                    self._lame_coefficient = self.youngs_modulus*self.poissons_ratio
                    self._lame_coefficient /= 1+self.poissons_ratio
                    self._lame_coefficient /= 1-2*self.poissons_ratio
                elif self.bulk_modulus is not None:
                    self._lame_coefficient = 3*self.bulk_modulus
                    self._lame_coefficient *= 3*self.bulk_modulus-self.youngs_modulus
                    self._lame_coefficient /= 9*self.bulk_modulus-self.youngs_modulus
        return self._lame_coefficient

    @lame_coefficient.setter
    def lame_coefficient(self, value):
        if value < 0:
            raise ValueError('lame_coefficient must be a positive float')
        self._lame_coefficient = value

    @property
    @is_initialized
    def shear_modulus(self):
        """
        Shear modulus (also known as Lame's second parameter). It is calculated from the
        average shear components of the compliance matrix if the elastic tensor is available.
        Otherwise it is calculated either from Lame's first parameter and Young's modulus and
        Poisson's ratio or Lame's first parameter and Poisson's ratio (depending on what is
        available)
        """
        if self._shear_modulus is None:
            if self._elastic_tensor is not None:
                self._shear_modulus = 1/self.compliance_matrix[3:,3:].diagonal().mean()
            elif self.lame_coefficient is not None:
                if self.youngs_modulus is not None:
                    R = self.youngs_modulus**2
                    R += 9*self.lame_coefficient**2
                    R += 2*self.youngs_modulus*self.lame_coefficient
                    R = np.sqrt(R)
                    self._shear_modulus = (self.youngs_modulus-3*self.lame_coefficient+R)/4
                elif self.poissons_ratio is not None:
                    self._shear_modulus = self.lame_coefficient*(1-2*self.poissons_ratio)
                    self._shear_modulus /= 2*self.poissons_ratio
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, value):
        if value < 0:
            raise ValueError('shear_modulus must be a positive float')
        self._shear_modulus = value

    @property
    @is_initialized
    def bulk_modulus(self):
        """
        Bulk modulus. It is calculated either from shear modulus and Lame's first parameter or
        shear modulus and Young's modulus (depending on what is available)
        """
        if self._bulk_modulus is None:
            if self.shear_modulus is not None:
                if self.lame_coefficient is not None:
                    self._bulk_modulus = self.lame_coefficient+2*self.shear_modulus/3
                elif self.youngs_modulus is not None:
                    self._bulk_modulus = self.youngs_modulus*self.shear_modulus
                    self._bulk_modulus /= 3*(3*self.youngs_modulus-self.shear_modulus)
        return self._bulk_modulus

    @bulk_modulus.setter
    def bulk_modulus(self, value):
        if value < 0:
            raise ValueError('bulk_modulus must be a positive float')
        self._bulk_modulus = value

    @property
    @is_initialized
    def poissons_ratio(self):
        """
        Poisson's ratio. If the elastic tensor is available, it is calculated from the compliance
        matrix. Otherwise it is calculated either from bulk modulus and shear modulus or from bulk
        modulus and lame coefficient (depending on what is available)
        """
        if self._poissons_ratio is None:
            if self._elastic_tensor is not None:
                self._poissons_ratio = -(self.compliance_matrix[:3,:3].sum()*self.youngs_modulus-3)/6
            elif self.bulk_modulus is not None:
                if self.shear_modulus is not None:
                    self._poissons_ratio = (3*self.bulk_modulus-2*self.shear_modulus)
                    self._poissons_ratio /= 2*(3*self.bulk_modulus+self.shear_modulus)
                elif self.lame_coefficient is not None:
                    self._poissons_ratio = self.lame_coefficient
                    self._poissons_ratio /= 3*self.bulk_modulus-self.lame_coefficient
        return self._poissons_ratio

    @poissons_ratio.setter
    def poissons_ratio(self, value):
        if value < 0:
            raise ValueError('poissons_ratio must be a positive float')
        self._poissons_ratio = value

    @property
    @is_initialized
    def youngs_modulus(self):
        """
        Young's modulus. If the elastic tensor is available, it is calculated from the compliance
        matrix. Otherwise it is calculated either from Poisson's ratio and bulk modulus or from
        Poisson's ratio and shear modulus (depending on what is available)
        """
        if self._youngs_modulus is None:
            if self._elastic_tensor is not None:
                self._youngs_modulus = 1/self.compliance_matrix[:3,:3].diagonal().mean()
            elif self.poissons_ratio is not None:
                if self.bulk_modulus is not None:
                    self._youngs_modulus = 3*self.bulk_modulus*(1-2*self.poissons_ratio)
                elif self.shear_modulus is not None:
                    self._youngs_modulus = 2*self.shear_modulus*(1+self.poissons_ratio)
        return self._youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, value):
        if value < 0:
            raise ValueError('youngs_modulus must be a positive float')
        self._youngs_modulus = value

    def get_greens_function(
        self, positions, derivative=0, fourier=False, n_mesh=100, isotropic=False, optimize=True
    ):
        """
        Green's function of the equilibrium condition:

        C_ijkl d^2u_k/dx_jdx_l = 0

        Args:
            positions ((n,3)-array): Positions in real space or reciprocal space (if fourier=True).
            derivative (int): 0th, 1st or 2nd derivative of the Green's function
            n_mesh (int): Number of mesh points in the radial integration in case if anisotropic
                Green's function (ignored if isotropic=True or fourier=True)
            isotropic (bool): Whether to use the isotropic or anisotropic elasticity. If the medium
                is isotropic, it will automatically be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,3)-array): Green's function values for the given positions
        """
        if isotropic or self._is_isotropic:
            C = Isotropic(self.poissons_ratio, self.shear_modulus, optimize=optimize)
        else:
            C = Anisotropic(self.elastic_tensor, n_mesh=n_mesh, optimize=optimize)
        return C.get_greens_function(positions, derivative, fourier)

    def get_point_defect_displacement(
        positions, dipole_tensor, n_mesh=100, isotropic=False, optimize=True
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
        return -np.einsum('...ijk,...kj->...i', g_tmp, dipole_tensor)

    def get_point_defect_strain(
        positions, dipole_tensor, n_mesh=100, isotropic=False, optimize=True
    ):
        """
        Strain field around a point defect

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

    def get_dislocation_displacement(self, positions, burgers_vector):
        hirth_lothe = HirthLothe(self.elastic_tensor, burgers_vector)
        return hirth_lothe.get_displacement(positions)

    def get_dislocation_strain(self, positions, burgers_vector):
        hirth_lothe = HirthLothe(self.elastic_tensor, burgers_vector)
        return hirth_lothe.get_strain(positions)

    def get_dislocation_stress(self, positions, burgers_vector):
        strain = self.get_dislocation_strain(positions, burgers_vector)
        return np.einsum('ijkl,...kl->...ij', self.elastic_tensor, strain)
