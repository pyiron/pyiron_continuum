# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import PyironFactory
from pyiron_base import Settings
from pyiron_continuum.elasticity.green import Anisotropic, Isotropic

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

def normalize(x):
    return (x.T/np.linalg.norm(x, axis=-1).T).T

def get_Ms(T, elastic_tensor):
    M = np.einsum('ijkl,...j,...l->...ik', elastic_tensor, T, T, optimize=True)
    return np.linalg.inv(M)

def get_plane(T):
    x = normalize(np.random.random(T.shape))
    x = normalize(x-np.einsum('...i,...i,...j->...j', x, T, T, optimize=True))
    y = np.cross(T, x)
    return x,y

def _get_integrand(C, z, T, Ms):
    zT = np.einsum('...p,...w->...pw', z, T)
    zT = zT+np.einsum('...ij->...ji', zT)
    F = np.einsum('jpnw,...ij,...nr,...pw->...ir', C, Ms, Ms, zT, optimize=True)
    MF = np.einsum('...ij,...nr->...ijnr', F, Ms, optimize=True)
    MF = MF+np.einsum('...ijnr->...nrij', MF, optimize=True)
    Air = np.einsum('...pw,...ijnr->...ijnrpw', zT, MF, optimize=True)
    Air -= 2*np.einsum('...ij,...nr,...p,...w->...ijnrpw', Ms, Ms, T, T, optimize=True)
    Air = np.einsum('jpnw,...ijnrpw->...ir', C, Air, optimize=True)
    results = 2*np.einsum('...s,...m,...ir->...irsm', T, T, Ms, optimize=True)
    results -= 2*np.einsum('...sm,...ir->...irsm', zT, F, optimize=True)
    results += np.einsum('...s,...m,...ir->...irsm', z, z, Air, optimize=True)
    return results

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
    def __init__(self, elastic_tensor=None):
        self._elastic_tensor = elastic_tensor
        self.isotropy_tolerance = 1.0e-4
        self._frame = np.eye(3)
        self.initialize()

    def initialize(self):
        self._lame_coefficient = None
        self._shear_modulus = None
        self._bulk_modulus = None
        self._poissons_ratio = None
        self._youngs_modulus = None

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, f):
        frame = self._frame.copy()
        frame[:2] = f[:2]
        frame = normalize(frame)
        frame[3] = np.cross(frame[0], frame[1])
        if np.isclose(np.linalg.det(frame), 0):
            raise ValueError('Vectors not independent')
        self._frame = np.einsum('ij,i->ij', self._frame, 1/np.linalg.norm(self._frame, axis=-1))
        self.initialize()

    @property
    def _is_rotated(self):
        return np.isclose(np.einsum('ii->', self.frame), 3)

    @property
    def elastic_tensor(self):
        if self._elastic_tensor is None:
            self._update()
        if self._elastic_tensor is not None and not self._is_rotated:
            return np.einsum('iI,IjKl,kK->ijkl', self.frame, self._elastic_tensor, self.frame)
        return self._elastic_tensor

    @elastic_tensor.setter
    def elastic_tensor(self, C):
        C = np.asarray(C)
        if C.shape != (6, 6) and C.shape != (3, 3, 3, 3):
            raise ValueError('Elastic tensor must be a (6,6) or (3,3,3,3) array')
        if C.shape == (6, 6):
            C = C_from_voigt(C)
        self._elastic_tensor = C

    @property
    @value_or_none
    def elastic_tensor_voigt(self):
        return C_to_voigt(self.elastic_tensor)

    @property
    @value_or_none
    def compliance_matrix(self):
        return np.linalg.inv(self.elastic_tensor_voigt)

    @property
    @is_initialized
    def zener_ratio(self):
        return 2*(1+self.poissons_ratio)*self.shear_modulus/self.youngs_modulus

    @property
    @is_initialized
    def is_isotropic(self):
        return np.absolute(self.zener_ratio-1) < self.isotropy_tolerance

    @property
    @is_initialized
    def lame_coefficient(self):
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

    @property
    @is_initialized
    def shear_modulus(self):
        if self._shear_modulus is None:
            if self._elastic_tensor is not None:
                self._shear_modulus = 1/self.compliance_matrix[3:,3:].diagonal().mean()
            elif self.lame_coefficient is not None:
                if self.youngs_modulus is not None:
                    R = self.youngs_modulus**2
                    R += 9*self.lame_coefficient**2
                    R += 2*self.youngs_modufus*self.lame_coefficient
                    R = np.sqrt(R)
                    self._shear_modulus = 2*self.lame_coefficient
                    self._shear_modulus /= self.youngs_modufus+self.lame_coefficient+R
                elif self.poissons_ratio is not None:
                    self._shear_modulus = self.lame_coefficient*(1-2*self.poissons_ratio)
                    self._shear_modulus /= 2*self.poissons_ratio
        return self._shear_modulus

    @property
    @is_initialized
    def bulk_modulus(self):
        if self._bulk_modulus is None:
            if self.shear_modulus is not None:
                if self.lame_coefficient is not None:
                    self._bulk_modulus = self.lame_coefficient+2*self.shear_modulus/3
                elif self.youngs_modulus is not None:
                    self._bulk_modulus = self.youngs_modulus*self.shear_modulus
                    self._bulk_modulus /= 3*(3*self.youngs_modulus-self.shear_modulus)
        return self._bulk_modulus

    @property
    @is_initialized
    def poissons_ratio(self):
        if self._poissons_ratio is None:
            if self._elastic_tensor is not None:
                self._poissons_ratio = -(self.compliance_matrix.sum()*self.youngs_modulus-3)/6
            elif self.bulk_modulus is not None:
                if self.shear_modulus is not None:
                    self._poissons_ratio = (3*self.bulk_modulus-2*self.shear_modulus)
                    self._poissons_ratio /= 2*(3*self.bulk_modulus+self.shear_modulus)
                elif self.lame_coefficient is not None:
                    self._poissons_ratio = self.lame_coefficient
                    self._poissons_ratio /= 3*self.bulk_modulus-self.lame_coefficient
        return self._poissons_ratio

    @property
    @is_initialized
    def youngs_modulus(self):
        if self._youngs_modulus is None:
            if self._elastic_tensor is not None:
                self._youngs_modulus = 1/self.compliance_matrix[:3,:3].diagonal().mean()
            elif self.poissons_ratio is not None:
                if self.bulk_modulus is not None:
                    self._youngs_modulus = 3*self.bulk_modulus*(1-2*self.poissons_ratio)
                elif self.shear_modulus is not None:
                    self._youngs_modulus = 2*self.shear_modulus*(1+self.poissons_ratio)
        return self._youngs_modulus

    def get_greens_function(
        self, positions, derivative=0, fourier=False, n_mesh=100, isotropic=False, optimize=True
    ):
        if isotropic or self.is_isotropic:
            C = Isotropic(self.poissons_ratio, self.shear_modulus)
        else:
            C = Anisotropic(self.elastic_tensor, n_mesh=n_mesh, optimize=optimize)
        return C.get_greens_function(positions, derivative, fourier)

