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


def normalize(x):
    return (x.T / np.linalg.norm(x, axis=-1).T).T


def orthonormalize(vectors):
    x = normalize(vectors)
    x[1] = x[1] - np.einsum("i,i,j->j", x[0], x[1], x[0])
    x[2] = np.cross(x[0], x[1])
    if np.isclose(np.linalg.det(x), 0):
        raise ValueError("Vectors not independent")
    return normalize(x)


def get_plane(T):
    x = normalize(np.random.random(T.shape))
    x = normalize(x - np.einsum("...i,...i,...j->...j", x, T, T))
    y = np.cross(T, x)
    return x, y


def index_from_voigt(i, j):
    if i == j:
        return i
    else:
        return 6 - i - j


def C_from_voigt(C_in):
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = C_in[index_from_voigt(i, j), index_from_voigt(k, l)]
    return C


def C_to_voigt(C_in):
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(i + 1):
            for k in range(3):
                for l in range(k + 1):
                    C[index_from_voigt(i, j), index_from_voigt(k, l)] = C_in[i, j, k, l]
    return C


def coeff_to_voigt(C_in):
    C = np.zeros((6, 6))
    C[:3, :3] = C_in[1]
    C[:3, :3] += np.eye(3) * (C_in[0] - C_in[1])
    C[3:, 3:] = C_in[2] * np.eye(3)
    return C
