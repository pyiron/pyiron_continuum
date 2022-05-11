# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for FEM linear elasticity with [fenics](https://fenicsproject.org/pub/tutorial/html/._ftut1008.html).
"""
from pyiron_base import ImportAlarm
with ImportAlarm(
        'fenics functionality requires the `fenics`, `mshr` modules (and their dependencies) specified as extra'
        'requirements. Please install it and try again.'
) as fenics_alarm:
    import fenics as FEN
    from ufl import nabla_div as ufl_nabla_div

from pyiron_continuum.fenics.job.generic import Fenics
from pyiron_continuum.fenics.plot import Plot
from pyiron_continuum.fenics.factory import SolverConfig
from pyiron_continuum.fenics.wrappers import Mesh, BoundaryConditions

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Dec 26, 2020"


class FenicsLinearElastic(Fenics):
    """
    Solves a linear elastic problem in three dimensions using bulk and shear moduli to describe the elasticity.
    Determines the displacement given the body load, traction, and boundary conditions.

    The variational equation solved is the integral of:
    `inner(sigma(u), epsilon(v)) * dx == dot(f, v) * dx + dot(T, v) * ds`

    Parameters:
        f (Constant/Expression): The body force term. (Default is Constant((0, 0, 0)).)
        T (Constant/Expression): The traction conditions. (Default is Constant((0, 0, 0)).)

    Input:
        bulk_modulus (float): Material elastic parameter. (Default is 76, the experimental value for Al in GPa.)
        shear_modulus (float): Material elastic parameter. (Default is 26, the experimental value for Al in GPa.)

    Output
        von_Mises (list): The von Mises stress from the mesh-evaluated solution at each step.
    """

    def __init__(self, project, job_name):
        """Create a new Fenics type job for linear elastic problems"""
        super().__init__(project=project, job_name=job_name)
        self._plot = ElasticPlot(self)

        self.input.bulk_modulus = 76
        self.input.shear_modulus = 26
        self.input.boundaries = BoundaryConditions()
        self.input.mesh = Mesh(
            'BoxMesh(p1, p2, nx, ny, nz)',
            **{'p1': 'Point((0,0,0))', 'p2': 'Point((1, 1, 1))', 'nx': 1, 'ny': 1, 'nz': 1}
        )

        self.output.von_Mises = []

    @property
    def solver(self):
        if self._solver is None:
            self._solver = ElasticSolver(job=self)
        return self._solver

    def validate_ready_to_run(self):
        self._mesh = self.input.mesh()
        self.domain._bcs = self.input.boundaries(self.solver.V)
        self.solver.set_sides_eq()
        super().validate_ready_to_run()

    def _append_to_output(self):
        super()._append_to_output()
        self.output.von_Mises.append(
            self.solver.von_Mises(self.solver.solution)\
                .compute_vertex_values(self.mesh)
            )


class ElasticSolver(SolverConfig):
    def __init__(self, job):
        super().__init__(job, func_space_class=FEN.VectorFunctionSpace)
        self.f = FEN.Constant((0, 0, 0))
        self.T = FEN.Constant((0, 0, 0))
        self._accepted_keys.append('T')

    @staticmethod
    def epsilon(u):
        return FEN.sym(FEN.nabla_grad(u))

    def sigma(self, u):
        lambda_ = self._job.input.bulk_modulus - (2 * self._job.input.shear_modulus / 3)
        return lambda_ * ufl_nabla_div(u) * FEN.Identity(u.geometric_dimension()) \
               + 2 * self._job.input.shear_modulus * self.epsilon(u)

    def von_Mises(self, u):
        s = self.sigma(u) - (1. / 3) * FEN.tr(self.sigma(u)) * FEN.Identity(u.geometric_dimension())
        return FEN.project(FEN.sqrt(3. / 2 * FEN.inner(s, s)),
                           FEN.FunctionSpace(
                               self._job.mesh,
                               self._job.input.element_type,
                               self._job.input.element_order
                               )
                           )

    def set_sides_eq(self):
        self.lhs = FEN.inner(self.sigma(self.u), self.epsilon(self.v)) * FEN.dx
        self.rhs = FEN.dot(self.f, self.v) * FEN.dx + FEN.dot(self.T, self.v) * FEN.ds

class ElasticPlot(Plot):
    def stress2d(
            self,
            frame=-1,
            n_grid=1000,
            n_grid_x=None,
            n_grid_y=None,
            add_colorbar=True,
            lognorm=False,
            projection_axis=None
    ):
        """
        Plot a heatmap of the von Mises stress interpolated onto a uniform grid.

        Args:
            frame (int): Which output frame to use. (Default is -1, most recent.)
            n_grid (int): Number of points to use when interpolating the mesh values. (Default is 1000.)
            n_grid_x (int): Number of grid points to use when interpolating the mesh values in the x-direction.
                (Default is None, use n_grid value.)
            n_grid_y (int): Number of grid points to use when interpolating the mesh values in the y-direction.
                (Default is None, use n_grid value.)
            add_colorbar (bool): Whether or not to add the colorbar to the figure. (Default is True.)
            lognorm (bool): Normalize the colorscheme to a log scale. (Default is False.)
            projection_axis (int): The axis onto which to project mesh nodes and nodal values if the data is 3d.
                (Default is None, project z onto xy-plane if needed.)

        Returns:
            (matplotlib.image.AxesImage): The imshow object.
            (matplotlib.figure.Figure): The parent figure.
            (matplotlib.axes._subplots.AxesSubplot): The subplots axis on which the plotting occurs.
        """
        return self.nodal2d(
            nodal_values=self._job.output.von_Mises[frame],
            nodes=self._job.mesh.coordinates() + self._job.output.solution[frame],
            n_grid=n_grid,
            n_grid_x=n_grid_x,
            n_grid_y=n_grid_y,
            add_colorbar=add_colorbar,
            lognorm=lognorm,
            projection_axis=projection_axis
        )

    def stress3d(self, frame=-1, add_colorbar=True):
        """
        3d scatter plot of the von Mises stress.

        Args:
            frame (int): Which output frame to use. (Default is -1, most recent.)
            add_colorbar (bool): Whether or not to add the colorbar to the figure. (Default is True.)

        Returns:
            (matplotlib.image.AxesImage): The scatter object.
            (matplotlib.figure.Figure): The parent figure.
            (matplotlib.axes._subplots.AxesSubplot): The subplots axis on which the plotting occurs.
        """
        return self.nodal3d(
            nodal_values=self._job.output.von_Mises[frame],
            nodes=self._job.mesh.coordinates() + self._job.output.solution[frame],
            add_colorbar=add_colorbar
        )
