from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "fenics functionality requires the `dolfinx`, `gmsh`, `pyvista` and"
    " `matplotlib` modules (and their dependencies) specified as extra"
    " requirements. Please install it and try again."
) as fenics_alarm:
    import dolfinx as DFX
    import pyvista
    import ufl
    from dolfinx import geometry
    from dolfinx.plot import vtk_mesh
    import numpy as np

pyvista.start_xvfb()


class PlotMesh:
    def __init__(self):
        super().__init__()

    def plotMesh(V):
        p = pyvista.Plotter()
        topology, cell_types, geometry = DFX.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        p.add_mesh(grid, show_edges=True)
        return p.show()


class PlotDeformed:
    def __init__(self):
        super().__init__()

    def plotDefomed_functionspace(V, uh, factor):
        topology, cell_types, x = DFX.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        grid.point_data["u"] = uh.x.array
        warped = grid.warp_by_scalar("u", factor=factor)
        p = pyvista.Plotter()
        p.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
        return p.show()

    def plotDefomed_vectorfunctionspace(V, uh, factor):
        topology, cell_types, geometry = DFX.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        p = pyvista.Plotter()
        grid["u"] = uh.x.array.reshape((geometry.shape[0], geometry.shape[1]))
        warped = grid.warp_by_vector("u", factor=factor)
        # actor_0 = p.add_mesh(grid, style="wireframe", color="k")
        _ = p.add_mesh(warped, show_edges=True)
        return p.show()


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, lambda_, mu):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


class PlotStresses:
    def __init__(self):
        super().__init__()

    def plotStresses_vonMises(mesh, V, uh, lambda_, mu, factor):
        topology, cell_types, geometry = DFX.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        s = sigma(uh, lambda_, mu) - 1.0 / geometry.shape[1] * ufl.tr(
            sigma(uh, lambda_, mu)
        ) * ufl.Identity(len(uh))
        von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))
        V_von_mises = DFX.fem.functionspace(mesh, ("DG", 0))
        stress_expr = DFX.fem.Expression(
            von_Mises, V_von_mises.element.interpolation_points()
        )
        stresses = DFX.fem.Function(V_von_mises)
        stresses.interpolate(stress_expr)
        p = pyvista.Plotter()
        grid["u"] = uh.x.array.reshape((geometry.shape[0], geometry.shape[1]))
        warped = grid.warp_by_vector("u", factor=factor)
        warped.cell_data["von Mises"] = stresses.vector.array
        warped.set_active_scalars("von Mises")
        p = pyvista.Plotter()
        p.add_mesh(warped, show_edges=True)
        return p.show()


class PlotLoad:
    def __init__(self):
        super().__init__()

    def plotLoad_scalar(mesh, load, factor):
        Q = DFX.fem.functionspace(mesh, ("Lagrange", 5))
        expr = DFX.fem.Expression(load, Q.element.interpolation_points())
        pressure = DFX.fem.Function(Q)
        pressure.interpolate(expr)
        p = pyvista.Plotter()
        load_grid = pyvista.UnstructuredGrid(*vtk_mesh(Q))
        load_grid.point_data["Load"] = pressure.x.array.real
        warped_load = load_grid.warp_by_scalar("Load", factor=factor)
        warped_load.set_active_scalars("Load")
        p.add_mesh(warped_load, show_edges=False, show_scalar_bar=True)
        return p.show()


class PlotValuesFunction:
    def __init__(self):
        super().__init__()

    def getValues_Deflection(mesh, uh, points):
        u_values = []
        bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
        cells = []
        points_on_proc = []
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(
            mesh, cell_candidates, points.T
        )
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        u_values = uh.eval(points_on_proc, cells)
        return u_values, points_on_proc

    def getValues_Pressure(mesh, load, points):
        Q = DFX.fem.functionspace(mesh, ("Lagrange", 5))
        expr = DFX.fem.Expression(load, Q.element.interpolation_points())
        pressure = DFX.fem.Function(Q)
        pressure.interpolate(expr)
        p_values = []
        bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
        cells = []
        points_on_proc = []
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(
            mesh, cell_candidates, points.T
        )
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        p_values = pressure.eval(points_on_proc, cells)
        return p_values, points_on_proc
