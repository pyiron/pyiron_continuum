# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for performing finite element simulations using the [FEniCS](https://fenicsproject.org) code.
"""
from pyiron_base import ImportAlarm
with ImportAlarm(
        'fenics functionality requires the `fenics`, `mshr` modules (and their dependencies) specified as extra'
        'requirements. Please install it and try again.'
) as fenics_alarm:
    import fenics as FEN
    import mshr
    from ufl import nabla_div as ufl_nabla_div
    import dolfin.cpp.mesh as FenicsMesh

import sympy
from pyiron_base import TemplateJob, DataContainer
from os.path import join
import warnings
import numpy as np
from pyiron_continuum.fenics.factory import SolverConfig, BoundaryConditions
from pyiron_continuum.fenics.wrappers import Mesh
from pyiron_continuum.fenics.plot import Plot

__author__ = "Muhammad Hassani, Liam Huber"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Muhammad Hassani"
__email__ = "hassani@mpie.de"
__status__ = "development"
__date__ = "Dec 6, 2020"


class Fenics(TemplateJob):
    """
    The job provides an interface to the [FEniCS project](https://fenicsproject.org) PDE solver using the finite element
    method (FEM).

    The objective is to streamline and simplify regular usage and connect FEniCS calculations to the full job management
    and execution distribution capabilities of pyiron, without losing the power or flexibility of the underlying fenics
    library.

    Flexibility and power are currently maintained by directly exposing the underlying `fenics` and `mshr` libraries as
    attributes of the job for power users.

    Ease of use is underway, e.g. elements, trial and test functions, and the mesh are automatically populated based on
    the provided domain. Quality of life will be continuously improved as pyiron and fenics get to know each other.

    TODO: Integration with pyiron's job and data management is incomplete, as some input data types (domains and
          boundary conditions) are not yet compatible with HDF5 storage. This is a simple problem to describe, but might
          be a pain to solve with sufficient flexibility. We also need to consider storing more sophisticated output.

    TODO: Full power and flexibility still needs to be realized by allowing (a) variable function space types, (b)
          variable number of elements and trial/test functions, and (c) multiple solve types.
          (a) Is a simple input option, we just need to be smart about how to limit the choices to existing fenics
              classes.
          (b) Can probably be nicely realized by subclassing off the main job type to allow for two sets of functions --
              `V:(u,v), Q:(p,q)` -- and a variable number of functions -- `V[0]:(u[0], v[0]),...,V[n]:(u[n], v[n])` --
              which are automatically populated during mesh generation and which are accessible for building the
              equation.
          (c) Solution types just means linear system `solve(A, x, b, ...)`, linear variational problems
              `solve(a == L, u, ...)`, and nonlinear variational problems `solve(F == 0, u, ...)`. This is probably also
              going to be pretty easy to control through an input parameter with a few fixed options an a bit of
              modification to how the LHS and RHS of equations are provided, and what actually is called during `run`.
              Currently the linear variational problem is hardcoded.

    Attributes:
        input (DataContainer): The input parameters controlling the run.
        output (DataContainer): The output from the run, i.e. data that comes from `solve`ing the PDE.
        domain (?): The spatial domain on which to build the mesh or, in the case of special meshes, the mesh itself.
            To be provided prior to running the job.
        BC (?): The boundary conditions for the mesh. To be provided prior to running the job.
        LHS/RHS (?): The left-hand and right-hand sides of the equation to solve.
        time_dependent_expressions (list[Expression]): All expressions used in the domain, BC, LHS and RHS which have a
            `t` attribute that needs updating at each step. (Default is None, which initializes an empty list.)
        assigned_u (?): The term which will be assigned the solution at each timestep. (Default is None, don't assign
            anything.)
        mesh (?): The mesh. Generated automatically.
        u:
        v:
        solution:
        F:

    Input:
        mesh_resolution (int): How dense the mesh should be (larger values = denser mesh). (Default is 2, but not used
            if the domain is a special mesh, e.g. unit or regular.)
        element_type (str): What type of element should be used. (Default is 'P'.) TODO: Restrict choices.
        element_order (int): What order the elements have. (Default is 1.)  TODO: Better description.
        n_steps (int): How many steps to run for, where the `t` attribute of all time dependent expressions gets updated
            at each step. (Default is 1.)
        n_print (int): The period of steps for saving to output. (Default is 1.) Note: The final step will always be
            saved regardless of the mod value.
        dt (float): How much to increase the `t` attribute of  all time dependent expressions each step. (Default is 1.)
        solver_parameters (dict): kwargs for FEniCS solver.
            Cf. [FEniCS docs](https://fenicsproject.org/pub/tutorial/html/._ftut1017.html) for more details. (Default is
            an empty dictionary.)

    Output:
        u (list): The solved function values evaluated at the mesh points at each time step.

    Example: OUT OF DATE
        >>> job = pr.create.job.Fenics('fenics_job')
        >>> job.input.mesh_resolution = 64
        >>> job.input.element_type = 'P'
        >>> job.input.element_order = 2
        >>> job.domain = job.create.domain.circle((0, 0), 1)
        >>> job.BC = job.create.bc.dirichlet(job.Constant(0))
        >>> p = job.Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))', degree=1, beta=8, R0=0.6)
        >>> job.LHS = job.dot(job.grad_u, job.grad_v) * job.dx
        >>> job.RHS = p * job.v * job.dx
        >>> job.run()
        >>> job.plot_u()
    """

    def __init__(self, project, job_name):
        """Create a new Fenics type job"""
        super(Fenics, self).__init__(project, job_name)
        warnings.warn("Currently, the c++ dolfin functions used by fenics are not stored in the HDF5 file."
                      " This includes the domains, boundary condition, spatial functions."
                      " Therefore, it is not possible to reload the job properly, from HDF5 file."
                      " It would be safe to remove the Fenics jobs, after defining the project.")
        self._python_only_job = True
        self._plot = Plot(self)

        self.input.boundaries = BoundaryConditions()
        self.input.mesh = Mesh(
            'BoxMesh(p1, p2, nx, ny, nz)',
            **{'p1': 'Point((0,0,0))', 'p2': 'Point((1, 1, 1))', 'nx': 1, 'ny': 1, 'nz': 1}
        )
        self.input.element_type = 'P'
        self.input.element_order = 1
        self.input.n_steps = 1
        self.input.n_print = 1
        self.input.dt = 1
        self.input.solver_parameters = {}

        self.output.solution = []

        # TODO: Figure out how to get these attributes into input/otherwise serializable
        self._vtk_filename = join(self.project_hdf5.path, 'output.pvd')

        self._solver = None

    # Wrap equations in setters so they can be easily protected in subclasses
    @property
    def mesh(self):
        return self.input.mesh()

    @property
    def bcs(self):
        return self.input.boundaries(self.solver.V)

    @property
    def solver(self):
        if self._solver is None:
            self._solver = SolverConfig(self)
        return self._solver

    @property
    def plot(self):
        return self._plot

    def _write_vtk(self):
        """
        Write the output to a .vtk file.
        """
        vtkfile = FEN.File(self._vtk_filename)
        vtkfile << self.solver.solution

    def validate_ready_to_run(self):
        if self.solver.rhs is None:
            raise ValueError("The bilinear form (RHS) is not defined")
        if self.solver.lhs is None:
            raise ValueError("The linear form (LHS) is not defined")
        if self.solver.V is None:
            raise ValueError("The volume is not defined; no V defined")
        if len(self.bcs) == 0:
            raise ValueError("The boundary condition(s) (BC) is not defined")
        for bc in self.input.boundaries.storage.values():
            if 't' in bc.value.kwargs.keys():
                self.solver.time_dependent_expressions.append(bc.value())

    def run_static(self):
        """
        Solve a PDE based on 'LHS=RHS' using u and v as trial and test function respectively. Here, u is the desired
        unknown and RHS is the known part.
        """
        self.status.running = True
        self.solver.u = self.solver.solution
        for step in np.arange(self.input.n_steps):
            for expr in self.solver.time_dependent_expressions:
                expr.t += self.input.dt
            FEN.solve(self.solver.lhs == self.solver.rhs, self.solver.u, self.bcs, solver_parameters=self.input.solver_parameters)
            if step % self.input.n_print == 0 or step == self.input.n_print - 1:
                self._append_to_output()
            try:
                self.solver.assigned_u.assign(self.solver.solution)
            except AttributeError:
                pass
        self.status.collect = True
        self.run()

    def _append_to_output(self):
        """Evaluate the result at nodes and store in the output as a numpy array."""
        nodal_solution = self.solver.solution.compute_vertex_values(self.mesh)
        nodes = self.mesh.coordinates()
        if len(nodal_solution) != len(nodes):
            nodal_solution = nodal_solution.reshape(nodes.T.shape).T
        self.output.solution.append(nodal_solution)

    @staticmethod
    def solution_to_original_shape(nodal_solution_frame):
        """Get a frame of the solution back to its original shape, even if it was reformed to match the mesh length."""
        if len(nodal_solution_frame.shape) == 1:
            return nodal_solution_frame
        else:
            return nodal_solution_frame.T.flatten()

    def collect_output(self):
        self._write_vtk()  # TODO: Get the output files so they're all tarballed after successful runs, like other codes
        self.to_hdf()
        self.status.finished = True

    @property
    def sympy(self):
        return sympy

    @property
    def fenics(self):
        return FEN
