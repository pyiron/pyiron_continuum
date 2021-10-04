# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
Refactory of the damask classes and methods to a pyironized manner
"""

from damask import Grid, Result, Config, ConfigMaterial, seeds

__author__ = "Muhammad Hassani"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Muhammad Hassani"
__email__ = "hassani@mpie.de"
__status__ = "development"
__date__ = "Oct 04, 2021"

class MateriaFactory:
    def __init__(self):
        """a refactory for damask ConfigMaterial class"""
        pass

    @staticmethod
    def config(rotation, elements, phase, homogenization):
        _config = ConfigMaterial({'material': [], 'phase': phase, 'homogenization': homogenization})
        for r, e in zip(rotation, elements):
            _config = _config.material_add(O=r, phase=e, homogenization=list(homogenization.keys())[0])
        return _config

    @staticmethod
    def read(file_path):
        return ConfigMaterial.load(fname=file_path)

    @staticmethod
    def write(file_path):
        ConfigMaterial.save(fname=file_path)


class GridFactory:
    def __init__(self):
        """a refactory for damask Grid class"""
        self.origin = Grid(material=np.ones((1, 1, 1)), size=[1., 1., 1.])

    @staticmethod
    def read(file_path):
        return Grid.load(fname=file_path)

    @staticmethod
    def via_voronoi_tessellation(grid_dim, num_grains, box_size):
        if isinstance(grid_dim, int) or isinstance(grid_dim, float):
            grid_dim = np.array([grid_dim, grid_dim, grid_dim])
        if isinstance(box_size, int) or isinstance(box_size, float):
            box_size = np.array([box_size, box_size, box_size])
        seed = seeds.from_random(box_size, num_grains)
        return Grid.from_Voronoi_tessellation(grid_dim, box_size, seed)


class DamaskLoading(Config):
    def __init__(self, load_steps, solver):
        """a refactory for damask Loading class, which is a damask.Config object"""
        super(DamaskLoading, self).__init__(self)
        self["solver"] = solver
        if isinstance(load_steps, list):
            self["loadstep"] = [
                LoadStep(mech_bc_dict=load_step['mech_bc_dict'],
                         discretization=load_step['discretization'],
                         additional_parameters_dict=load_step["additional"])
                for load_step in load_steps]
        else:
            self["loadstep"] = [
                LoadStep(mech_bc_dict=load_steps['mech_bc_dict'],
                         discretization=load_steps['discretization'],
                         additional_parameters_dict=load_steps["additional"])
            ]


class LoadStep(dict):
    def __init__(self, mech_bc_dict, discretization, additional_parameters_dict=None):
        """An auxilary class, which helps to parse loadsteps to a dictionary"""
        super(LoadStep, self).__init__(self)
        self.update({'boundary_conditions': {'mechanical': {}},
                     'discretization': discretization})

        if additional_parameters_dict is not None and isinstance(additional_parameters_dict, dict):
            self.update(additional_parameters_dict)

        for key, val in mech_bc_dict.items():
            self['boundary_conditions']['mechanical'].update({key: LoadStep.load_tensorial(val)})

    @staticmethod
    def load_tensorial(arr):
        return [arr[0:3], arr[3:6], arr[6:9]]
