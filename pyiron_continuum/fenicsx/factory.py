from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "fenics functionality requires the `dolfinx`, `gmsh` modules (and their dependencies) specified as extra"
    "requirements. Please install it and try again."
) as fenics_alarm:
    import dolfinx as DFX
    import gmsh
    from mpi4py import MPI
    from dolfinx.io import gmshio
from pyiron_base import PyironFactory

gmsh.initialize()


class GeometryFactory(PyironFactory):
    def __init__(self):
        super().__init__()

    def disk(self, xc, yc, zc, rx, ry):
        return gmsh.model.occ.addDisk(xc, yc, zc, rx, ry)

    def rectangle(self, x0, y0, z0, x1, y1):
        return gmsh.model.occ.addRectangle(x0, y0, z0, x1, y1)

    def box(self, x0, y0, z0, x1, y1, z1):
        return gmsh.model.occ.addBox(x0, y0, z0, x1, y1, z1)

    def cylinder(self, x0, y0, z0, dx, dy, dz, r):
        return gmsh.model.occ.addCylinder(x0, y0, z0, dx, dy, dz, r)

    def cut(self, dimensions, geom1, geom2):
        return gmsh.model.occ.cut([(dimensions, geom1)], [(dimensions, geom2)])


class MeshFactory(PyironFactory):
    def __init__(self):
        super().__init__()

    def standardDomain(self, geom, gdim, minCL, maxCL):
        gmsh.model.occ.synchronize()
        gdim = gdim
        gmsh.model.addPhysicalGroup(gdim, [geom], 1)
        minCL = minCL or 0.05
        maxCL = maxCL or 0.05
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", minCL)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxCL)
        gmsh.model.mesh.generate(gdim)
        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        return gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

    def clear_gmsh(self):
        return gmsh.clear()


class SpaceFactory(PyironFactory):
    def __init__(self):
        super().__init__()

    def functionspace(self, mesh, elementType, elementDegree):
        elementType = elementType or "Lagrange"
        elementDegree = elementDegree or 1
        return DFX.fem.functionspace(mesh, (elementType, elementDegree))

    def vectorfunctionspace(self, mesh, elementType, elementDegree):
        elementType = elementType or "Lagrange"
        elementDegree = elementDegree or 1
        return DFX.fem.functionspace(
            mesh, (elementType, elementDegree, (mesh.geometry.dim,))
        )


# class BoundaryConditionFactory(PyironFactory):
#     def __init__(self, job):
#         self._job = job

#     def DirichletBC_functionspace(self, func, V, value):
#         boundary_dofs = DFX.fem.locate_dofs_geometrical(V, func)
#         bc = DFX.fem.dirichletbc(DFX.default_scalar_type(value), boundary_dofs, V)
#         return bc

#     def DirichletBC_vectorfunctionspace(self, func, V, value_x, value_y, value_z):
#         boundary_dofs = DFX.fem.locate_dofs_geometrical(V, func)
#         u_D = np.array([value_x, value_y, value_z], dtype=DFX.default_scalar_type)
#         bc = DFX.fem.dirichletbc(u_D, boundary_dofs, V)
#         return bc
