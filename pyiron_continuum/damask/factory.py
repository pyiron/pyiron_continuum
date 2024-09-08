# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""Factory of the damask classes and methods to a pyironized manner"""

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "DAMASK functionality requires the `damask` module (and its dependencies) specified as extra"
    "requirements. Please install it and try again."
) as damask_alarm:
    from damask import GeomGrid, YAML, ConfigMaterial, seeds, Rotation
import numpy as np
from pyiron_continuum.reference.mendeleev import get_atom_info
from pyiron_continuum.damask.reference.yaml import get_elasticity, get_plasticity

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


# TODO: reimplement export_vtk() here. Currently, damask dumps vtk files in the cwd


class MaterialFactory:
    def __init__(self):
        """A factory for damask ConfigMaterial class."""
        pass

    @staticmethod
    def config(rotation, elements, phase, homogenization):
        _config = ConfigMaterial(
            {"material": [], "phase": phase, "homogenization": homogenization}
        )
        for r, e in zip(rotation, elements):
            _config = _config.material_add(
                O=r, phase=e, homogenization=list(homogenization.keys())[0]
            )
        return _config

    @staticmethod
    def read(file_path):
        return ConfigMaterial.load(fname=file_path)

    @staticmethod
    def write(file_path):
        ConfigMaterial.save(fname=file_path)


class GridFactory:
    def __init__(self):
        """A factory for damask._grid.GeomGrid class."""
        self._origin = None

    @staticmethod
    def read(file_path):
        return GeomGrid.load(fname=file_path)

    @property
    def origin(self):
        """
        Returns damask._grid.GeomGrid, it can be used to call damask original methods.
        For example:
        origin.from_Voronoi_tessellation(...)
        """
        if self._origin is None:
            return GeomGrid(material=np.ones((1, 1, 1)), size=[1.0, 1.0, 1.0])
        else:
            return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value

    @staticmethod
    def via_voronoi_tessellation(spatial_discretization, num_grains, box_size):
        if isinstance(spatial_discretization, int) or isinstance(
            spatial_discretization, float
        ):
            spatial_discretization = np.array(
                [spatial_discretization, spatial_discretization, spatial_discretization]
            )
        if isinstance(box_size, int) or isinstance(box_size, float):
            box_size = np.array([box_size, box_size, box_size])
        seed = seeds.from_random(box_size, num_grains)
        return GeomGrid.from_Voronoi_tessellation(
            spatial_discretization, box_size, seed
        )


def generate_loading_tensor(default="F"):
    """
    Returns the default boundary conditions for the damask loading tensor.

    Args:
        default (str): Default value of the tensor. It can be 'F', 'P', 'dot_F'
            or 'dot_P'.

    Returns:
        tuple: A tuple of two numpy arrays. The first array is the keys and the
            second array is the values.
    """
    assert default in ["F", "P", "dot_F", "dot_P"]
    if default == "F":
        return np.full((3, 3), "F").astype("<U5"), np.eye(3)
    else:
        return np.full((3, 3), default).astype("<U5"), np.zeros((3, 3))


def loading_tensor_to_dict(key, value):
    """
    Converts the damask loading tensor to a dictionary.

    Args:
        key (numpy.ndarray): Keys of the tensor
        value (numpy.ndarray): Values of the tensor

    Returns:
        dict: A dictionary of the tensor

    Example:
        key, value = generate_loading_tensor()
        loading_tensor_to_dict(key, value)

    Comments:

        `key` and `value` should be generated from
        `generate_loading_tensor()` and as the format below:

        (array([['F', 'F', 'F'],
                ['F', 'F', 'F'],
                ['F', 'F', 'F']], dtype='<U5'),
         array([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]]))

        where the first array is the keys and the second array is the values.
        The keys can be 'F', 'P', 'dot_F' or 'dot_P'. These keys correspond to:

        F: deformation gradient at end of load step
        dot_F: rate of deformation gradient during load step
        P: first Piola–Kirchhoff stress at end of load step
        dot_P: rate of first Piola–Kirchhoff stress during load step
    """
    result = {}
    for tag in ["F", "P", "dot_F", "dot_P"]:
        if tag in key:
            mat = np.full((3, 3), "x").astype(object)
            mat[key == tag] = value[key == tag]
            result[tag] = mat.tolist()
    return result


def generate_load_step(
    N,
    t,
    F=None,
    dot_F=None,
    P=None,
    dot_P=None,
    f_out=None,
    r=None,
    f_restart=None,
    estimate_rate=None
):
    """
    Args:
        N (int): Number of increments
        t (float): Time of load step in seconds, i.e.
        F (numpy.ndarray): Deformation gradient at end of load step
        dot_F (numpy.ndarray): Rate of deformation gradient during load step
        P (numpy.ndarray): First Piola–Kirchhoff stress at end of load step
        dot_P (numpy.ndarray): Rate of first Piola–Kirchhoff stress during
            load step
        r (float): Scaling factor (default 1) in geometric time step series
        f_out (int): Output frequency of results, i.e. f_out=3 writes results
            every third increment
        f_restart (int): output frequency of restart information; e.g.
            f_restart=3 writes restart information every tenth increment
        estimate_rate (float): estimate field of deformation gradient
            fluctuations based on former load step (default) or assume to be
            homogeneous, i.e. no fluctuations

    Returns:
        dict: A dictionary of the load step

    You can find more information about the parameters in the damask documentation:
    https://damask-multiphysics.org/documentation/file_formats/grid_solver.html#load-case
    """
    result = {
        "boundary_conditions": {"mechanical": {}},
        "discretization": {"t": t, "N": N},
    }
    if r is not None:
        result["discretization"]["r"] = r
    if f_out is not None:
        result["f_out"] = f_out
    if f_restart is not None:
        result["f_restart"] = f_restart
    if estimate_rate is not None:
        result["estimate_rate"] = estimate_rate
    if F is None and dot_F is None and P is None and dot_P is None:
        raise ValueError("At least one of the tensors should be provided.")
    if F is not None:
        result["boundary_conditions"]["mechanical"]["F"] = F
    if dot_F is not None:
        result["boundary_conditions"]["mechanical"]["dot_F"] = dot_F
    if P is not None:
        result["boundary_conditions"]["mechanical"]["P"] = P
    if dot_P is not None:
        result["boundary_conditions"]["mechanical"]["dot_P"] = dot_P
    return result


def get_damask_loading(solver, load_steps):
    if not isinstance(load_steps, list):
        load_steps = [load_steps]
    if "mech_bc_dict" in load_steps[0]:
        load_steps = [
            translate_load_steps(
                mech_bc_dict=load_step["mech_bc_dict"],
                discretization=load_step["discretization"],
                additional_parameters_dict=load_step["additional"],
            )
            for load_step in load_steps
        ]
    return YAML(solver=solver, loadstep=load_steps)


def translate_load_steps(
    mech_bc_dict, discretization, additional_parameters_dict=None
):
    """An auxilary class, which helps to parse loadsteps to a dictionary."""
    result = {
        "boundary_conditions": {"mechanical": {}},
        "discretization": discretization,
    }

    if isinstance(additional_parameters_dict, dict):
        result.update(additional_parameters_dict)

    for key, val in mech_bc_dict.items():
        result["boundary_conditions"]["mechanical"].update(
            {key: [val[0:3], val[3:6], val[6:9]]}
        )
    return result


class Create:
    def __init__(self):
        """The create factory for the damask job."""
        self._grid = GridFactory()

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @staticmethod
    def loading(solver, load_steps):
        """
        Creates the required damask loading.
        Args:
            solver(dict): a dictionary desrcribing the solver: e.g, {'mechanical': 'spectral_basic'}
            load_steps(list/single dict): a list of dict or single dict, which describes the loading conditions
            an example would be:
            {'mech_bc_dict':{'dot_F':[1e-2,0,0, 0,'x',0,  0,0,'x'],
                            'P':['x','x','x', 'x',0,'x',  'x','x',0]},
            'discretization':{'t': 10.,'N': 40, 'f_out': 4},
            'additional': {'f_out': 4}
        """
        return get_damask_loading(solver=solver, load_steps=load_steps)

    @staticmethod
    def material(rotation, elements, phase, homogenization):
        """
        Creates a damask material.
        Args:
            rotation(damask.Rotation): damask rotation object
            elements(str): elements describing the phase
            phase(dict): a dictionary describing the phase parameters
            homogenization(dict): a dictionary describing the damask homogenization
        """
        return MaterialFactory.config(rotation, elements, phase, homogenization)

    @staticmethod
    def homogenization(method=None, parameters=None):
        """
        Returns damask homogenization as a dictionary.
        Args:
            method(str): homogenization method
            parameters(dict): the required parameters
        Examples:
            homogenization(method='SX', parameters={'N_constituents': 1, "mechanical": {"type": "pass"}})
        """
        if method is None:
            method = "SX"
        if parameters is None:
            parameters = {"N_constituents": 1, "mechanical": {"type": "pass"}}
        return {method: parameters}

    @staticmethod
    def phase(composition, elasticity, plasticity=None, lattice=None, output_list=None):
        """
        Returns a dictionary describing the phases for damask.
        Args:
            composition(str)
            lattice(str)
            output_list(str)
            elasticity(dict)
            plasticity(dict)
        Examples:
            phase = phase(
                composition='Aluminum',
                lattice='cF',
                elasticity=elasticity,
                plasticity=plasticity
            )

        For the details of isotropic model, one can refer to https://doi.org/10.1016/j.scriptamat.2017.09.047
        """
        if lattice is None:
            lattice = {"BCC": "cI", "HEX": "hP", "FCC": "cF"}[
                get_atom_info(name=composition)["lattice_structure"]
            ]
        if output_list is None:
            if plasticity is None:
                output_list = ["F", "P", "F_e"]
            else:
                output_list = ["F", "P", "F_e", "F_p", "L_p", "O"]
        d = {
            composition: {
                "lattice": lattice,
                "mechanical": {"output": output_list, "elastic": elasticity},
            }
        }
        if plasticity is not None:
            d[composition]["mechanical"]["plastic"] = plasticity
        return d

    @staticmethod
    def list_elasticity():
        return get_elasticity()

    @staticmethod
    def elasticity(**kwargs):
        """
        Args:
            type (str): Type of elasticity model (e.g. "Hooke")
            C_11, C_12, C_44 (float): Elastic constants in Pascals (material)

        Returns a dictionary of elasticity parameters for damask input file.
        Examples:
             # For aluminium
             elasticity = elasticity(
                type='Hooke', C_11=106.75e9, C_12=60.41e9, C_44=28.34e9
            )
        """
        return kwargs

    @staticmethod
    def list_plasticity():
        return get_plasticity()

    @staticmethod
    def plasticity(**kwargs):
        """
        Returns a dictionary of plasticity parameters for damask input file.

        Examples:
            plasticity = plasticity(N_sl=[12], a_sl=2.25,
                                    atol_xi=1.0, dot_gamma_0_sl=0.001,
                                    h_0_sl_sl=75e6,
                                    h_sl_sl=[1, 1, 1.4, 1.4, 1.4, 1.4],
                                    n_sl=20, output=['xi_sl'],
                                    type='phenopowerlaw', xi_0_sl=[31e6],
                                    xi_inf_sl=[63e6])

        Parameters for elastoplastic model ( power-law hardening behavior)
        type : plasticity model (Here phenopowerlaw : Phenomenological
            plasticity with power-law hardening behavior) (model)
        N_sl : Number of slip-systems for a given slip family (material)
        a_sl : Hardening exponent for slip (material)
        dot_gamma_0_sl : reference/initial shear strain rate for slip in per seconds (experiment)
        h_0_sl_sl : reference/initial hardening rate for slip-slip activity in pascals (material)
        h_sl_sl : slip resistance from slip activity. Value of unity
            corresponds to self hardening and 1.4 for latent hardening (not for
            coplannar slip systems) (model)
        n_sl : stress exponent for slip (material)
        xi_0_sl : initial critical shear stress for slip in pascals (material)
        xi_inf_sl : maximum critical shear stress for slip in pascals (material)
        output : Quantity as a output (Here xi_sl: shear stress) (damask)

        Parameters for elastoplastic model ( isotropic hardening)
        a : Hardening exponent for slip (material); cf. `a_sl`
        dot_gamma_0 : reference/initial shear strain rate for slip in per seconds (experiment); cf. dot_gamma_0_sl
        h_0 : reference/initial hardening stress in pascals (material); cf. h_0_sl_sl
        h : slip resistance from slip activity. Value of unity corresponds to
            self hardening and 1.4 for latent hardening (not for coplannar slip
            systems) (model); cf. h_sl
        n : stress exponent (material); cf. n_sl
        xi_0 : initial critical shear stress in pascals (material)
        xi_inf : maximum critical shear stress in pascals (material)
        M : Taylor factor (material)
        output : Quantity as a output (Here xi: shear stress) (damask)
        """
        has_h0 = False
        has_h = False
        vals = {}
        for key, value in kwargs.items():
            if "h_0_sl_sl" in key:
                has_h0 = True
                vals["h_0_sl-sl"] = value
            if "h_sl_sl" in key:
                has_h = True
                vals["h_sl-sl"] = value
        if has_h0:
            kwargs["h_0_sl-sl"] = vals["h_0_sl-sl"]
            del kwargs["h_0_sl_sl"]
        if has_h:
            kwargs["h_sl-sl"] = vals["h_sl-sl"]
            del kwargs["h_sl_sl"]
        return kwargs

    @staticmethod
    def rotation(method="from_random", *args, **kwargs):
        """
        Args:
            method (damask.Rotation.*/str): Method of damask.Rotation class
                which based on the given arguments creates the Rotation object.
                If string is given, it looks for the method within
                `damask.Rotation` via `getattr`.
        """
        if isinstance(method, str):
            method = getattr(Rotation, method)
        return method(*args, **kwargs)
