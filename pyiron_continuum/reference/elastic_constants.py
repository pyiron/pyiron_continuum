# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.


from atomistics.referencedata.wikipedia import get_elastic_properties


def get_elastic_constants(chemical_symbol):
    """
    Get the elastic tensor of a material from the wikipedia database.

    Args:
        chemical_symbol (str): Chemical symbol of the element.

    Returns:
        dict: Dictionary with the elastic tensor of the material (isotropic).
    """
    d = get_elastic_properties(chemical_symbol)
    G = d["shear_modulus"]
    v = d["poissons_ratio"]
    E = d["youngs_modulus"]
    C_11 = E * (1 - v) / (1 + v) / (1 - 2 * v)
    C_12 = E * v / (1 + v) / (1 - 2 * v)
    C_44 = G
    d.update({"C_11": C_11, "C_12": C_12, "C_44": C_44})
    return d
