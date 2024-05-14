from pathlib import Path
import h5py
import numpy as np
import scipy
import damask
from functools import cached_property


class Regrid:
    def __init__(self, work_dir, geom_name, load_name, seed_scale=1.0):
        self.work_dir = Path(work_dir)
        self.geom_name = geom_name
        self.load_name = load_name
        self.seed_scale = seed_scale

    def get_path(self, file_name):
        return str(self.work_dir / file_name)

    @cached_property
    def geom_0(self):
        return damask.GeomGrid.load(self.get_path(f"{self.geom_name}.vti"))

    @property
    def cells_0(self):
        """initial grid seeds"""
        return self.geom_0.cells

    @property
    def size_0(self):
        """initial RVE size"""
        return self.geom_0.size_0

    @cached_property
    def gridCoords_node_initial(self):
        mat = damask.grid_filters.coordinates0_node(self.cells_0, self.size_0)
        return mat.reshape((-1, mat.shape[-1]), order="F")

    @cached_property
    def gridCoords_point_initial(self):
        mat = damask.grid_filters.coordinates0_point(self.cells_0, self.size_0)
        return mat.reshape((-1, mat.shape[-1]), order="F")

    @cached_property
    def d5Out(self):
        d5Out = damask.Result(self.get_path(f"{self.geom_name}_{self.load_name}_material.hdf5"))
        d5Out = d5Out.view(increments=int(d5Out.increments[-1]))
        return d5Out

    @property
    def increment_title(self):
        return self.d5Out.increments[-1]

    @property
    def gridCoords_node_0(self):
        return self.gridCoords_node_initial + self.d5Out.get("u_n") - self.d5Out.origin

    @property
    def gridCoords_cell_0(self):
        return self.gridCoords_point_initial + self.d5Out.get("u_p") - self.d5Out.origin

    @property
    def size_rg(self):
        """regridded RVE size"""
        cornersRVE_coords = np.diag(self.size_0)
        cornersRVE_idx = [
            np.argwhere(
                np.isclose(self.gridCoords_node_initial, cornersRVE_coords[corner]).all(axis=1)
            ).item()
            for corner in range(3)
        ]
        return np.diag(self.gridCoords_node_0[cornersRVE_idx]) - self.gridCoords_node_0[0]

    @property
    def cells_rg(self):
        """regridded grid seeds"""
        seedSize_rg = np.min(self.size_rg / np.array(self.cells_0))
        if isinstance(self.seed_scale, float) or isinstance(self.seed_scale, int):
            return (self.seed_scale * np.round(self.size_rg / seedSize_rg)).astype(int)
        elif isinstance(self.seed_scale, list):
            return np.array(self.seed_scale).astype(int)
        else:
            raise ValueError("The seed_scale for regridded RVE size is not acceptable.")

    @property
    def map_0to_rg(self):
        gridCoords_cell_rg = damask.grid_filters.coordinates0_point(
            self.cells_rg, self.size_rg
        ).reshape((-1, 3), order="F")
        # apply periodic shift
        gridCoords_cell_0_Shifted = self.gridCoords_cell_0 % self.size_rg
        tree = scipy.spatial.cKDTree(gridCoords_cell_0_Shifted, boxsize=self.size_rg)
        return tree.query(gridCoords_cell_rg)[1].flatten()

    @property
    def regrid_geom_name(self):
        return f"{self.geom_name}_regridded_{self.increment_title}"

    @property
    def grid(self):
        grid_0 = damask.GeomGrid.load(self.get_path(self.geom_name + ".vti"))
        material_rg = grid_0.material.flatten("F")[self.map_0to_rg].reshape(self.cells_rg, order="F")
        grid = damask.GeomGrid(
            material_rg, self.size_rg, grid_0.origin, comments=grid_0.comments
        ).save(self.get_path(f"{self.geom_name}_regridded_{self.increment_title}.vti"))
        return grid


def regrid_Geom(work_dir, geom_name, load_name, seed_scale=1.0, increment="last"):
    """
    Regrid the geometry
    It requires the previous geom.vti and result.hdf5
    """
    work_dir = Path(work_dir)
    geom_0 = damask.GeomGrid.load(str(work_dir / f"{geom_name}.vti"))
    cells_0 = geom_0.cells
    size_0 = geom_0.size

    gridCoords_node_initial_matrix = damask.grid_filters.coordinates0_node(
        cells_0, size_0
    )
    gridCoords_node_initial = gridCoords_node_initial_matrix.reshape(
        (-1, gridCoords_node_initial_matrix.shape[-1]), order="F"
    )
    gridCoords_point_initial_matrix = damask.grid_filters.coordinates0_point(
        cells_0, size_0
    )
    gridCoords_point_initial = gridCoords_point_initial_matrix.reshape(
        (-1, gridCoords_point_initial_matrix.shape[-1]), order="F"
    )

    d5Out = damask.Result(str(work_dir / f"{geom_name}_{load_name}_material.hdf5"))

    if increment == "last":
        inc = int(d5Out.increments[-1])  # take the increment number
        d5Out = d5Out.view(increments=inc)
        increment_title = d5Out.increments[-1]
    elif increment in d5Out.increments:
        inc = int(increment)  # take the increment number
        d5Out = d5Out.view(increments=inc)
        increment_title = increment
    else:
        raise ValueError("The results for the increment is not availabble!")
    # calculate the deformed coordinates
    gridCoords_node_0 = gridCoords_node_initial + d5Out.get("u_n") - d5Out.origin
    gridCoords_cell_0 = gridCoords_point_initial + d5Out.get("u_p") - d5Out.origin

    print("------------------------")
    print("Start to regrid the geometry ...")

    cornersRVE_coords = np.diag(size_0)
    cornersRVE_idx = [
        np.argwhere(
            np.isclose(gridCoords_node_initial, cornersRVE_coords[corner]).all(axis=1)
        ).item()
        for corner in range(3)
    ]
    size_rg = np.diag(gridCoords_node_0[cornersRVE_idx]) - gridCoords_node_0[0]
    seedSize_rg = np.min(size_rg / np.array(cells_0))

    regriddingSeedScale = seed_scale
    if isinstance(regriddingSeedScale, float) or isinstance(regriddingSeedScale, int):
        cells_rg = (regriddingSeedScale * np.round(size_rg / seedSize_rg)).astype(int)
    elif isinstance(regriddingSeedScale, list):
        cells_rg = np.array(regriddingSeedScale).astype(int)
    else:
        raise ValueError("The seed_scale for regridded RVE size is not acceptable.")

    gridCoords_cell_rg = damask.grid_filters.coordinates0_point(
        cells_rg, size_rg
    ).reshape((-1, 3), order="F")
    gridCoords_cell_0_Shifted = gridCoords_cell_0 % size_rg  # apply periodic shift
    print("initial RVE size:\t", size_0)
    print("regridded RVE size:\t", size_rg)
    print("initial grid seeds:\t", cells_0)
    print("regridded grid seeds:\t", cells_rg)
    print("finding the nearest neighbors...")
    tree = scipy.spatial.cKDTree(gridCoords_cell_0_Shifted, boxsize=size_rg)
    map_0to_rg = tree.query(gridCoords_cell_rg)[1].flatten()
    print("all the information are ready !")
    return map_0to_rg, cells_rg, size_rg, increment_title


def write_RegriddedGeom(
    work_dir, geom_name, increment_title, map_0to_rg, cells_rg, size_rg
):
    """
    Save the regridded geometry to a new vti file
    """
    work_dir = Path(work_dir)
    grid_0 = damask.GeomGrid.load(geom_name + ".vti")
    material_rg = grid_0.material.flatten("F")[map_0to_rg].reshape(cells_rg, order="F")
    grid = damask.GeomGrid(
        material_rg, size_rg, grid_0.origin, comments=grid_0.comments
    ).save(work_dir / f"{geom_name}_regridded_{increment_title}.vti")
    print(f"save regrid geometry to {geom_name}_regridded_{increment_title}.vti")
    regrid_geom_name = f"{geom_name}_regridded_{increment_title}"
    return grid, regrid_geom_name


def write_RegriddedHDF5(
    work_dir,
    geom_name,
    regrid_geom_name,
    load_name,
    increment_title,
    map_0to_rg,
    cells_rg,
):
    """
    Write out the new hdf5 file based on the regridded geometry and deformation info
    """
    work_dir = Path(work_dir)
    fNameResults_0 = work_dir / f"{geom_name}_{load_name}_material.hdf5"
    fNameRestart_0 = work_dir / f"{geom_name}_{load_name}_material_restart.hdf5"
    fNameRestart_rg = (
        work_dir / f"{geom_name}_regridded_{increment_title}_{load_name}_material_restart.hdf5"
    )
    print("geom_name=", geom_name)
    print("load_name=", load_name)
    if fNameRestart_rg.exists(fNameRestart_rg):
        print("removing the existing restart file.")
        fNameRestart_rg.unlink()

    with h5py.File(str(fNameRestart_0), "r") as fRestart_0, h5py.File(
        str(fNameResults_0), "r"
    ) as fResults_0, h5py.File(str(fNameRestart_rg), "w") as fRestart_rg:

        map_0toRg_phaseBased = fResults_0["cell_to/phase"][:, 0][map_0to_rg]

        ### for phase
        for phase in fRestart_0["/phase"]:
            fRestart_rg.create_group(f"/phase/{phase}")

            F_0 = damask.tensor.transpose(fRestart_0[f"/phase/{phase}/F"])
            F_p_0 = damask.tensor.transpose(fRestart_0[f"/phase/{phase}/F_p"])
            F_e_0 = np.matmul(F_0, np.linalg.inv(F_p_0))
            R_e_0, V_e_0 = damask.mechanics._polar_decomposition(F_e_0, ["R", "V"])

            map_0to1_phase = map_0toRg_phaseBased[
                map_0toRg_phaseBased["label"] == phase.encode()
            ]["entry"]

            for dataset in fRestart_0[f"/phase/{phase}"]:
                path = f"/phase/{phase}/{dataset}"
                if dataset == "S":
                    data_rg = np.zeros((len(map_0to1_phase), 3, 3))
                elif dataset == "F_p":
                    data_rg = R_e_0[map_0to1_phase]
                elif dataset == "F":
                    data_rg = np.broadcast_to(np.eye(3), (len(map_0to1_phase), 3, 3))
                else:
                    data_0 = fRestart_0[path][()]
                    data_rg = data_0[map_0to1_phase, ...]

                fRestart_rg.create_dataset(path, data=data_rg)

        ### for homogenization
        for homogenization in fRestart_0["/homogenization"]:
            fRestart_rg.create_group(f"/homogenization/{homogenization}")
            # not implemented ...

        ### for solver
        for dataset in fRestart_0["/solver"]:
            path = f"/solver/{dataset}"
            if dataset in [
                "C_minMaxAvg",
                "C_volAvg",
                "C_volAvgLastInc",
                "F_aimDot",
                "P_aim",
            ]:
                data_rg = fRestart_0[path]
            elif dataset in ["F_aim", "F_aim_lastInc"]:
                data_rg = np.eye(3)
            elif dataset in ["T", "T_lastInc"]:
                shape = fRestart_0[path].shape[3:]
                data_0 = fRestart_0[path][()].reshape((-1,) + shape)
                data_rg = data_0[map_0to_rg, ...].reshape(tuple(cells_rg[::-1]) + shape)
            elif dataset in ["F", "F_lastInc"]:
                shape = fRestart_0[path].shape[3:]
                data_rg = np.broadcast_to(
                    np.eye(3), (len(map_0to_rg), 3, 3)
                ).reshape(tuple(cells_rg[::-1]) + shape)
            else:
                print("Warning: There is restart variables that cannot be handled!")

            fRestart_rg.create_dataset(path, data=data_rg)

        def reset_cellIndex(map_0toRg_phaseBased, fRestart_0):
            NewCellIndex = np.zeros(len(map_0toRg_phaseBased))
            for phase in fRestart_0["/phase"]:
                test = map_0toRg_phaseBased["label"] == phase.encode()
                phaseLength = len(
                    map_0toRg_phaseBased[
                        map_0toRg_phaseBased["label"] == phase.encode()
                    ]["entry"]
                )
                NewCellIndex[test] = range(phaseLength)
            map_0toRg_phaseBased["entry"] = NewCellIndex
            return map_0toRg_phaseBased

        with h5py.File(str(work_dir / "regridding.hdf5"), "w") as fRgHistory_0:
            path = "/map/0"
            fRgHistory_0.create_dataset(path, data=map_0to_rg)
            path = "/phase/0"
            # for phase in fRestart_0['/phase']:
            #     ll = len(map_0toRg_phaseBased[map_0toRg_phaseBased['label']==phase.encode()]['entry'])
            # strange this doesn't work!
            #     # map_0toRg_phaseBased[map_0toRg_phaseBased['label']==phase.encode()]['entry'][:] = range(ll)
            #     map_0toRg_phaseBased['entry'] = range(ll)

            map_0toRg_phaseBased = reset_cellIndex(map_0toRg_phaseBased, fRestart_0)

            fRgHistory_0.create_dataset(path, data=map_0toRg_phaseBased)
            print("A regridding history file is created.")
    file_to_copy = work_dir / f"{geom_name}_{load_name}_restart_regridded_{increment_title}_material.hdf5"
    file_to_copy.with_name(f"{regrid_geom_name}_{load_name}_restart_regridded_{increment_title}_material.hdf5")
    print("------------------------\nRegridding process is completed.")
