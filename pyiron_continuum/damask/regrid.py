import os
import h5py
import numpy as np
import scipy
import damask
import subprocess


def regrid_Geom(work_dir, geom_name, load_name, seed_scale=1.0, increment='last'):
    '''
    Regrid the geometry
    It requires the previous geom.vti and result.hdf5
    '''
    geom_0 = damask.GeomGrid.load(f'{geom_name}.vti')
    cells_0 = geom_0.cells
    size_0 = geom_0.size

    gridCoords_node_initial_matrix = damask.grid_filters.coordinates0_node(cells_0, size_0)
    gridCoords_node_initial = gridCoords_node_initial_matrix.reshape((-1, gridCoords_node_initial_matrix.shape[-1]),
                                                                     order='F')
    gridCoords_point_initial_matrix = damask.grid_filters.coordinates0_point(cells_0, size_0)
    gridCoords_point_initial = gridCoords_point_initial_matrix.reshape((-1, gridCoords_point_initial_matrix.shape[-1]),
                                                                       order='F')

    os.chdir(f'{work_dir}')
    d5Out = damask.Result(f'{work_dir}/{geom_name}_{load_name}_material.hdf5')

    if increment == 'last':
        inc = int(d5Out.increments[-1])  # take the increment number
        d5Out = d5Out.view(increments=inc)
        increment_title = d5Out.increments[-1]
    elif increment in d5Out.increments:
        inc = int(increment)  # take the increment number
        d5Out = d5Out.view(increments=inc)
        increment_title = increment
    else:
        raise ValueError('The results for the increment is not availabble!')
    # calculate the deformed coordinates
    gridCoords_node_0 = gridCoords_node_initial + d5Out.get('u_n') - d5Out.origin
    gridCoords_cell_0 = gridCoords_point_initial + d5Out.get('u_p') - d5Out.origin

    print('------------------------')
    print('Start to regrid the geometry ...')

    cornersRVE_coords = np.diag(size_0)
    cornersRVE_idx = [
        np.argwhere(np.isclose(gridCoords_node_initial, cornersRVE_coords[corner]).all(axis=1)).item() for corner in
        range(3)]
    size_rg = np.diag(gridCoords_node_0[cornersRVE_idx]) - gridCoords_node_0[0]
    seedSize_rg = np.min(size_rg / np.array(cells_0))

    regriddingSeedScale = seed_scale
    if isinstance(regriddingSeedScale, float) or isinstance(regriddingSeedScale, int):
        cells_rg = (regriddingSeedScale * np.round(size_rg / seedSize_rg)).astype(int)
    elif isinstance(regriddingSeedScale, list):
        cells_rg = np.array(regriddingSeedScale).astype(int)
    else:
        raise ValueError('The seed_scale for regridded RVE size is not acceptable.')

    gridCoords_cell_rg = damask.grid_filters.coordinates0_point(cells_rg, size_rg).reshape((-1, 3), order='F')
    gridCoords_cell_0_Shifted = gridCoords_cell_0 % size_rg  # apply periodic shift
    print('initial RVE size:\t', size_0)
    print('regridded RVE size:\t', size_rg)
    print('initial grid seeds:\t', cells_0)
    print('regridded grid seeds:\t', cells_rg)
    print('finding the nearest neighbors...')
    tree = scipy.spatial.cKDTree(gridCoords_cell_0_Shifted, boxsize=size_rg)
    map_0to_rg = tree.query(gridCoords_cell_rg)[1].flatten()
    print('all the information are ready !')
    return map_0to_rg, cells_rg, size_rg, increment_title


def write_RegriddedGeom(work_dir, geom_name, increment_title, map_0to_rg, cells_rg, size_rg):
    '''
    Save the regridded geometry to a new vti file
    '''
    os.chdir(f'{work_dir}')
    grid_0 = damask.GeomGrid.load(geom_name + '.vti')
    material_rg = grid_0.material.flatten('F')[map_0to_rg].reshape(cells_rg, order='F')
    grid = damask.GeomGrid(material_rg, size_rg, grid_0.origin, comments=grid_0.comments \
                       ).save(f'{geom_name}_regridded_{increment_title}.vti')
    print(f'save regrid geometry to {geom_name}_regridded_{increment_title}.vti')
    regrid_geom_name = f'{geom_name}_regridded_{increment_title}'
    return grid, regrid_geom_name


def write_RegriddedHDF5(work_dir, geom_name, regrid_geom_name, load_name, increment_title, map_0to_rg, cells_rg):
    '''
    Write out the new hdf5 file based on the regridded geometry and deformation info
    '''
    isElastic = False
    os.chdir(f'{work_dir}')
    fNameResults_0 = f"{geom_name}_{load_name}_material.hdf5"
    fNameRestart_0 = f"{geom_name}_{load_name}_material_restart.hdf5"
    fNameRestart_rg = f'{geom_name}_regridded_{increment_title}_{load_name}_material_restart.hdf5'
    print('geom_name=', geom_name)
    print('load_name=', load_name)
    if os.path.exists(fNameRestart_rg):
        print('removing the existing restart file.')
        os.remove(fNameRestart_rg)

    fName_rgHistory = 'regridding.hdf5'
    isRgHistoryExist = os.path.isfile(fName_rgHistory)

    with h5py.File(fNameRestart_0, 'r') as fRestart_0, \
            h5py.File(fNameResults_0, 'r') as fResults_0, \
            h5py.File(fNameRestart_rg, 'w') as fRestart_rg:

        isRgHistoryExist = False

        if not isRgHistoryExist:
            map_0toRg_phaseBased = fResults_0['cell_to/phase'][:, 0][map_0to_rg]
        else:
            with h5py.File(fName_rgHistory, 'r') as fRgHistory_0:
                historyRg_idx = np.array(fRgHistory_0['map']).astype(int)[-1]
                historyRg_map_0toRg_phaseBased = np.array(fRgHistory_0['phase'][str(historyRg_idx)])
            map_0toRg_phaseBased = historyRg_map_0toRg_phaseBased[map_0to_rg]

        ### for phase
        for phase in fRestart_0['/phase']:
            fRestart_rg.create_group(f'/phase/{phase}')

            F_0 = damask.tensor.transpose(fRestart_0[f'/phase/{phase}/F'])
            F_p_0 = damask.tensor.transpose(fRestart_0[f'/phase/{phase}/F_p'])
            F_e_0 = np.matmul(F_0, np.linalg.inv(F_p_0))
            R_e_0, V_e_0 = damask.mechanics._polar_decomposition(F_e_0, ['R', 'V'])

            map_0to1_phase = map_0toRg_phaseBased[map_0toRg_phaseBased['label'] == phase.encode()]['entry']

            for dataset in fRestart_0[f'/phase/{phase}']:
                path = f'/phase/{phase}/{dataset}'
                if dataset == 'S':
                    data_rg = np.zeros((len(map_0to1_phase), 3, 3))
                elif dataset == 'F_p':
                    data_rg = R_e_0[map_0to1_phase]
                elif dataset == 'F':
                    data_rg = np.broadcast_to(np.eye(3), (len(map_0to1_phase), 3, 3)) if not isElastic else \
                        V_e_0[map_0to1_phase]
                else:
                    data_0 = fRestart_0[path][()]
                    data_rg = data_0[map_0to1_phase, ...]

                fRestart_rg.create_dataset(path, data=data_rg)

        ### for homogenization
        for homogenization in fRestart_0['/homogenization']:
            fRestart_rg.create_group(f'/homogenization/{homogenization}')
            # not implemented ...

        ### for solver
        for dataset in fRestart_0['/solver']:
            path = f'/solver/{dataset}'
            if dataset in ['C_minMaxAvg', 'C_volAvg', 'C_volAvgLastInc', 'F_aimDot', 'P_aim']:
                data_rg = fRestart_0[path]
            elif dataset in ['F_aim', 'F_aim_lastInc']:
                data_rg = np.eye(3) if not isElastic else \
                    scipy.linalg.sqrtm(np.einsum('lij,lkj->ik', F_e_0, F_e_0) / len(F_e_0))  # MD: order correct?
            elif dataset in ['T', 'T_lastInc']:
                shape = fRestart_0[path].shape[3:]
                data_0 = fRestart_0[path][()].reshape((-1,) + shape)
                data_rg = data_0[map_0to_rg, ...].reshape(tuple(cells_rg[::-1]) + shape)
            elif dataset in ['F', 'F_lastInc']:
                if isElastic:
                    shape = fRestart_0[path].shape[3:]
                    data_0 = fRestart_0[path][()].reshape((-1,) + shape)
                    data_rg = data_0[map_0to_rg, ...].reshape(tuple(cells_rg[::-1]) + shape)
                else:
                    shape = fRestart_0[path].shape[3:]
                    data_rg = np.broadcast_to(np.eye(3), (len(map_0to_rg), 3, 3)).reshape(tuple(cells_rg[::-1]) + shape)
            else:
                print('Warning: There is restart variables that cannot be handled!')

            fRestart_rg.create_dataset(path, data=data_rg)

        def reset_cellIndex(map_0toRg_phaseBased, fRestart_0):
            NewCellIndex = np.zeros(len(map_0toRg_phaseBased))
            for phase in fRestart_0['/phase']:
                test = map_0toRg_phaseBased['label'] == phase.encode()
                phaseLength = len(map_0toRg_phaseBased[map_0toRg_phaseBased['label'] == phase.encode()]['entry'])
                NewCellIndex[test] = range(phaseLength)
            map_0toRg_phaseBased['entry'] = NewCellIndex
            return map_0toRg_phaseBased

        if not isRgHistoryExist:
            os.chdir(f'{work_dir}')
            with h5py.File('regridding.hdf5', 'w') as fRgHistory_0:
                path = '/map/0'
                fRgHistory_0.create_dataset(path, data=map_0to_rg)
                path = '/phase/0'
                # for phase in fRestart_0['/phase']:
                #     ll = len(map_0toRg_phaseBased[map_0toRg_phaseBased['label']==phase.encode()]['entry'])
                # strange this doesn't work!
                #     # map_0toRg_phaseBased[map_0toRg_phaseBased['label']==phase.encode()]['entry'][:] = range(ll)
                #     map_0toRg_phaseBased['entry'] = range(ll)

                map_0toRg_phaseBased = reset_cellIndex(map_0toRg_phaseBased, fRestart_0)

                fRgHistory_0.create_dataset(path, data=map_0toRg_phaseBased)
                print('A regridding history file is created.')
        else:
            with h5py.File('regridding.hdf5', 'a') as fRgHistory_0:
                previousHistory = np.array(fRgHistory_0[f'/map/{historyRg_idx}'])
                if np.all(previousHistory == map_0to_rg):
                    print('# WARNING: The regridding map is similar to the previous map!')
                path = f'/map/{historyRg_idx + 1}'
                fRgHistory_0.create_dataset(path, data=map_0to_rg)
                path = f'/phase/{historyRg_idx + 1}'

                # for phase in fRestart_0['/phase']:
                #     ll = len(map_0toRg_phaseBased[map_0toRg_phaseBased['label']==phase.encode()]['entry'])
                #     # map_0toRg_phaseBased[map_0toRg_phaseBased['label']==phase.encode()]['entry'] = range(ll)
                #     map_0toRg_phaseBased['entry'] = range(ll)

                map_0toRg_phaseBased = reset_cellIndex(map_0toRg_phaseBased, fRestart_0)

                fRgHistory_0.create_dataset(path, data=map_0toRg_phaseBased)
                print(f'The regridding history file is extended (index = {historyRg_idx + 1}).')
    os.chdir(f'{work_dir}')
    args = f'cp {geom_name}_{load_name}_restart_regridded_{increment_title}_material.hdf5 {regrid_geom_name}_{load_name}_restart_regridded_{increment_title}_material.hdf5'
    subprocess.run(args, shell=True, capture_output=True)
    print('------------------------\nRegridding process is completed.')
