from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "fenics functionality requires the `dolfinx`, `gmsh`, `pyvista` and"
    " `matplotlib` modules (and their dependencies) specified as extra"
    "requirements. Please install it and try again."
) as fenics_alarm:
    # from dolfinx import mesh, fem, plot, io, default_scalar_type
    # from dolfinx.fem.petsc import LinearProblem
    import dolfinx as DFX
    import ufl
    from dolfinx.fem.petsc import LinearProblem

from pyiron_base import GenericJob, DataContainer
import warnings
import numpy as np
from pyiron_continuum.fenicsx.factory import (
    GeometryFactory,
    MeshFactory,
    SpaceFactory,
)  # , BoundaryConditionFactory
from pyiron_continuum.fenicsx.plot import PlotMesh, PlotDeformed, PlotStresses
from pyiron_continuum.fenicsx.plot import PlotLoad, PlotValuesFunction


class Fenicsx(GenericJob):

    def __init__(self, project, job_name):
        """Create a new Fenics type job"""
        super(Fenicsx, self).__init__(project, job_name)
        warnings.warn(
            "Currently, the c++ dolfin functions used by fenicsx are not stored in the HDF5 file."
            " This includes the domains, boundary condition, spatial functions."
            " Therefore, it is not possible to reload the job properly, from HDF5 file."
            " It would be safe to remove the Fenics jobs, after defining the project."
        )
        self._python_only_job = True
        self.create = Creator(self)
        self.plot = Plot(self)
        self.geom = None
        self.domain = None
        self.mesh = None
        self.cell_markers = None
        self.facet_markers = None
        self.V = None
        self.bcs = []

        self.output = DataContainer(table_name="output")
        self.output.solution = []

        self.u = None
        self.v = None
        self.a = None
        self.p = None
        self.L = None
        self.uh = None

        self.f = None
        self.T = None

        self.lambda_ = None
        self.mu = None

    def dolfinx(self):
        return DFX

    def make_dirichletBC_functionspace(self, v, func, value):
        boundary_dofs = DFX.fem.locate_dofs_geometrical(v, func)
        return DFX.fem.dirichletbc(DFX.default_scalar_type(value), boundary_dofs, v)

    def make_dirichletBC_vectorfunctionspace_3D(
        self, v, func, value_x, value_y, value_z
    ):
        boundary_dofs = DFX.fem.locate_dofs_geometrical(v, func)
        u_D = np.array([value_x, value_y, value_z], dtype=DFX.default_scalar_type)
        return DFX.fem.dirichletbc(u_D, boundary_dofs, v)

    def appendBC(self, bc):
        self.bcs.append(bc)

    def spatial_coord(self, dmn):
        return ufl.SpatialCoordinate(dmn)

    def Constant_scalar(self, dmn, value):
        return DFX.fem.Constant(dmn, DFX.default_scalar_type(value))

    def Constant_vector_3D(self, dmn, value_x, value_y, value_z):
        return DFX.fem.Constant(
            dmn, DFX.default_scalar_type((value_x, value_y, value_z))
        )

    def Expression(self, *args, **kwargs):
        return ufl.exp(*args, **kwargs)

    def set_load(self, load):
        self.p = load

    def solvePoisson(self):
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.a = ufl.dot(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        self.L = self.p * self.v * ufl.dx
        problem = LinearProblem(
            self.a,
            self.L,
            bcs=self.bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        self.uh = problem.solve()
        return self.uh

    def set_bodyforce(self, f):
        self.f = f

    def set_traction(self, T):
        self.T = T

    def set_lambda(self, ll):
        self.lambda_ = ll

    def set_mu(self, mu):
        self.mu = mu

    def epsilon(self, u):
        return ufl.sym(ufl.grad(u))

    def sigma(self, u):
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(
            len(u)
        ) + 2 * self.mu * self.epsilon(u)

    def solveLinearElastic(self):
        ds = ufl.Measure("ds", domain=self.mesh)
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.a = ufl.inner(self.sigma(self.u), self.epsilon(self.v)) * ufl.dx
        self.L = ufl.dot(self.f, self.v) * ufl.dx + ufl.dot(self.T, self.v) * ds
        problem = LinearProblem(
            self.a,
            self.L,
            bcs=self.bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        self.uh = problem.solve()
        return self.uh

    # def Constant()


class Creator:
    def __init__(self, job):
        self._job = job
        self._geom = GeometryFactory()
        self._domain = MeshFactory()
        self._V = SpaceFactory()
        # self._bc = BoundaryConditionFactory(job)

    @property
    def geom(self):
        return self._geom

    @property
    def domain(self):
        return self._domain

    @property
    def V(self):
        return self._V

    def get_mesh(self, dm):
        self._mesh, self._cell_markers, self._facet_markers = dm
        return self._mesh

    def get_cell_markers(self, dm):
        self._mesh, self._cell_markers, self._facet_markers = dm
        return self._cell_markers

    def get_facet_markers(self, dm):
        self._mesh, self._cell_markers, self._facet_markers = dm
        return self._facet_markers

    # @property
    # def bc(self):
    #     return self._bc


class Plot:
    def __init__(self, job):
        PlotMesh()

    def plot_mesh(self, V1):
        return PlotMesh.plotMesh(V1)

    def plot_deformed_functionspace(self, V1, uh, factor):
        return PlotDeformed.plotDefomed_functionspace(V1, uh, factor)

    def plot_deformed_vectorfunctionspace(self, V1, uh, factor):
        return PlotDeformed.plotDefomed_vectorfunctionspace(V1, uh, factor)

    def plot_stresses_vonMises(self, mesh, V1, uh, lambda_, mu, factor):
        return PlotStresses.plotStresses_vonMises(mesh, V1, uh, lambda_, mu, factor)

    def plot_load_scalar(self, mesh, load, factor):
        return PlotLoad.plotLoad_scalar(mesh, load, factor)

    def getValues_deflection(self, mesh, uh, points):
        return PlotValuesFunction.getValues_Deflection(mesh, uh, points)

    def getValues_pressure(self, mesh, load, points):
        return PlotValuesFunction.getValues_Pressure(mesh, load, points)
