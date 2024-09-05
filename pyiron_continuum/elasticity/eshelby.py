# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np

__author__ = "Sam Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "development"
__date__ = "Aug 21, 2021"


class Eshelby:
    """
    Anisotropic elasticity theory for dislocations described by
    [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

    All notations follow the original paper.
    """

    def __init__(self, elastic_tensor, burgers_vector):
        self.elastic_tensor = elastic_tensor
        self.burgers_vector = burgers_vector
        self.fit_range = np.linspace(0, 1, 10)
        self._p = None
        self._Ak = None
        self._D = None

    def _get_pmat(self, x):
        return (
            self.elastic_tensor[:, 0, :, 0]
            + np.einsum(
                "...,ij->...ij",
                x,
                self.elastic_tensor[:, 0, :, 1] + self.elastic_tensor[:, 1, :, 0],
            )
            + np.einsum("...,ij->...ij", x**2, self.elastic_tensor[:, 1, :, 1])
        )

    @property
    def p(self):
        if self._p is None:
            coeff = np.polyfit(
                self.fit_range, np.linalg.det(self._get_pmat(self.fit_range)), 6
            )
            self._p = np.roots(coeff)
            self._p = self._p[np.imag(self._p) > 0]
        return self._p

    @property
    def Ak(self):
        if self._Ak is None:
            self._Ak = []
            for mat in self._get_pmat(self.p):
                values, vectors = np.linalg.eig(mat.T)
                self._Ak.append(vectors.T[np.absolute(values).argmin()])
            self._Ak = np.array(self._Ak)
        return self._Ak

    @property
    def D(self):
        if self._D is None:
            F = np.einsum("n,ij->nij", self.p, self.elastic_tensor[:, 1, :, 1])
            F += self.elastic_tensor[:, 1, :, 0]
            F = np.einsum("nik,nk->ni", F, self.Ak)
            F = np.concatenate((F.T, self.Ak.T), axis=0)
            F = np.concatenate((np.real(F), -np.imag(F)), axis=-1)
            self._D = np.linalg.solve(
                F, np.concatenate((np.zeros(3), self.burgers_vector))
            )
            self._D = self._D[:3] + 1j * self._D[3:]
        return self._D

    @property
    def dzdx(self):
        return np.stack((np.ones_like(self.p), self.p, np.zeros_like(self.p)), axis=-1)

    def _get_z(self, positions):
        z = np.stack((np.ones_like(self.p), self.p), axis=-1)
        return np.einsum("nk,...k->...n", z, np.asarray(positions)[..., :2])

    def get_displacement(self, positions):
        """
        Displacement vectors

        Args:
            positions ((n,3)-array): Positions for which the displacements are to be calculated

        Returns:
            ((n,3)-array): Displacement vectors
        """
        return np.imag(
            np.einsum(
                "nk,n,...n->...k", self.Ak, self.D, np.log(self._get_z(positions))
            )
        ) / (2 * np.pi)

    def get_strain(self, positions):
        """
        Strain tensors

        Args:
            positions ((n,3)-array): Positions for which the strains are to be calculated

        Returns:
            ((n,3,3)-array): Strain tensors
        """
        strain = np.imag(
            np.einsum(
                "ni,n,...n,nj->...ij",
                self.Ak,
                self.D,
                1 / self._get_z(positions),
                self.dzdx,
            )
        )
        strain = strain + np.einsum("...ij->...ji", strain)
        return strain / 4 / np.pi
