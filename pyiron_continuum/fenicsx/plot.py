from pyiron_base import ImportAlarm
with ImportAlarm(
        'fenics functionality requires the `dolfinx`, `gmsh`, `pyvista` and `matplotlib` modules (and their dependencies) specified as extra'
        'requirements. Please install it and try again.'
) as fenics_alarm:
    import dolfinx as DFX
    import gmsh
    import pyvista
    from dolfinx.plot import vtk_mesh
    import matplotlib.pyplot as plt

pyvista.start_xvfb()

class PlotMesh():
    def __init__(self):
        super().__init__()

    def plotMesh(V):
        p = pyvista.Plotter()
        topology, cell_types, geometry = DFX.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        p.add_mesh(grid, show_edges=True)
        return p.show()

class PlotDeformed():
    def __init__(self):
        super().__init__()

    def plotDefomed_functionspae(V, uh, factor):
        topology, cell_types, x = DFX.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        grid.point_data["u"] = uh.x.array
        warped = grid.warp_by_scalar("u", factor=factor)
        p = pyvista.Plotter()
        p.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
        return p.show()

    def plotDefomed_vectorfunctionspae(V, uh, factor):
        topology, cell_types, geometry = DFX.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        p = pyvista.Plotter()
        grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
        #actor_0 = p.add_mesh(grid, style="wireframe", color="k")
        warped = grid.warp_by_vector("u", factor=factor)
        actor_1 = p.add_mesh(warped, show_edges=True)
        return p.show()
