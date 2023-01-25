from ast import arg
import imp
import os
from random import vonmisesvariate
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import subprocess

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir)

from damaskjob import DAMASK
from damask import regridding as rgg
from damask import Result
from damask import Config
from factory import DamaskLoading


class ROLLING(DAMASK):
    """
    
    """

    def __init__(self, project, job_name):
        """Create a new DAMASK type job for rolling"""
        super().__init__(project=project, job_name=job_name)

    def rolling_parameters(self, number_passes, height_reduction, rolling_speed, contact_length, increments,
                           regridding=False):
        self._number_passes = number_passes
        self._height_reduction = height_reduction
        self._rolling_speed = rolling_speed
        self._contact_length = contact_length
        self._regridding = regridding
        self._increments = increments

    def loading_discretization(self, rolltimes, filename):
        time = rolltimes * self._height_reduction / (self._rolling_speed * self._number_passes)

        load_case = Config(solver={'mechanical': 'spectral_basic'}, loadstep=[])
        dotF = [['x', 0, 0],
                [0, 0, 0],
                [0, 0, -1.0 * self._rolling_speed]]
        P = [[0, 'x', 'x'],
             ['x', 'x', 'x'],
             ['x', 'x', 'x']]
        loadstep = {'boundary_conditions': {'mechanical': {'P': P,
                                                           'dot_F': dotF}},
                    'discretization': {'t': time, 'N': self._increments * rolltimes},
                    'f_out': 5,
                    'f_restart': 5}
        load_case['loadstep'].append(loadstep)
        self._loadfilename = filename
        self._loading = load_case
        file_path = os.path.join(self.working_directory, filename + '.yaml')
        self._loading.save(file_path)
        # self.input.loading = self._loading
        print(self._loading)

    def run_rolling(self):
        print('working direction is:', self.working_directory)
        path = os.path.join(self.working_directory)
        os.makedirs(path, exist_ok=True)
        os.chdir(self.working_directory)  # cd into the working dir
        subprocess.run("rm -rf *", shell=True, capture_output=True)
        path = os.path.join(self.working_directory, 'regridrolling')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(self.working_directory, 'regridrolling/damask_job')
        os.makedirs(path, exist_ok=True)
        print('Remove all the results ...')
        self._write_material()
        self._write_geometry()
        self._rollresults = []
        self._vonMises_stress = []
        self._vonMises_strain = []
        for step in range(self._number_passes):
            print('--------------------------------------------------------------')
            print('Do the rolling-%d test ...' % (step + 1))
            if step > 0: self.savecurrentloading()
            cwd = os.getcwd()  # get the current dir
            os.chdir(self.working_directory)  # cd into the working dir
            print("Run damask simulation from ", os.getcwd())
            if step < 1:
                # for the first step
                loadfilename = 'rolling-%d' % (step + 1)
                self.loading_discretization(step + 1, loadfilename)
                print('Using %s.yaml as loading' % (self._loadfilename))
                args = "DAMASK_grid -g damask.vti -l %s.yaml > %s.log" % (self._loadfilename, self._loadfilename)
                print('Damask command is:', args)
                subprocess.run(args, shell=True, capture_output=True)
                resultfile = 'damask_%s.hdf5' % (self._loadfilename)
                filename = os.path.join(self.working_directory, resultfile)
                myresults = Result(filename)
                myresults.add_stress_Cauchy()
                myresults.add_strain()
                myresults.add_equivalent_Mises('sigma')
                myresults.add_equivalent_Mises('epsilon_V^0.0(F)')
            else:
                if self._regridding:
                    cwd = os.getcwd()  # get the current dir
                    os.chdir(self.working_directory)  # cd into the working dir
                    # do the regridding process
                    if step == 1:
                        geom_name = 'damask'  # after regridding, it is damask_regridded
                    else:
                        geom_name = self._regrid_geom_file_old
                    seed_scale = 1.0
                    load_name_old = 'rolling-%d' % (step)
                    load_name = 'rolling-%d' % (step + 1)
                    self.loading_discretization(step + 1, load_name)
                    # seeds = options.seeds
                    map_0to_rg, cells_rg, size_rg, increment_title = rgg.regrid_geom(geom_name, load_name_old,
                                                                                     seed_scale=seed_scale, seeds='nan',
                                                                                     increment='last')  # no output
                    self._grid = rgg.write_geomRegridded(geom_name, increment_title, map_0to_rg, cells_rg, size_rg)
                    # --- writing the regridded restart file ---
                    rgg.write_h5OutRestart(geom_name, load_name_old, increment_title, map_0to_rg, cells_rg,
                                           isElastic=False)
                    # here the filename rule is:
                    self._regrid_geom_file = f'{geom_name}_regridded.vti'
                    self._regrid_geom_file_old = f'{geom_name}_regridded'
                    self._regrid_hdf5file = f"{geom_name}_{load_name_old}.hdf5"
                    self._regrid_hdf5restartfile = f"{geom_name}_{load_name_old}_restart.hdf5"
                    self._regridedrestartfile = f'{geom_name}_{load_name_old}_restart_regridded.hdf5'
                    print("restart DAMASK (regridded) simulation from ", os.getcwd())

                    #############################################################
                    'damask_regridded_rolling-1.hdf5'
                    args = "cp %s %s" % (
                    self._regridedrestartfile, f"{self._regrid_geom_file_old}_{load_name_old}.hdf5")
                    subprocess.run(args, shell=True, capture_output=True)
                    # print('Do: ',args)

                    args = "DAMASK_grid -g %s -l %s.yaml > regrid.log" % (self._regrid_geom_file, load_name)
                    subprocess.run(args, shell=True, capture_output=True)
                    # print('The execute command is: ',args)
                    os.chdir(cwd)  # cd back to the notebook dir
                    print('DAMASK (regridded) restart simulation is done !')
                    # remove the stress/strain in previous simulation
                    self._regrid_result_file = f'{self._regrid_geom_file_old}_{load_name}.hdf5'
                    filename = os.path.join(self.working_directory, self._regrid_result_file)
                    myresults = Result(filename)
                    r_unprotected = myresults.view(protected=False)
                    r_unprotected.remove('sigma')
                    r_unprotected.remove('sigma_vM')
                    r_unprotected.remove('epsilon_V^0.0(F)')
                    r_unprotected.remove('epsilon_V^0.0(F)_vM')
                    myresults.add_stress_Cauchy()
                    myresults.add_strain()
                    myresults.add_equivalent_Mises('sigma')
                    myresults.add_equivalent_Mises('epsilon_V^0.0(F)')
                else:
                    # do the normal restart
                    loadfilenameold = 'rolling-%d' % (step)
                    loadfilename = 'rolling-%d' % (1)

                    # args="cp damask_%s.C_ref damask_%s.C_ref"%(loadfilenameold,loadfilename)
                    # subprocess.run(args,shell=True,capture_output=True) 

                    # args="cp damask_%s.hdf5 damask_%s.hdf5"%(loadfilenameold,loadfilename)
                    # subprocess.run(args,shell=True,capture_output=True) 

                    # args="cp damask_%s_restart.hdf5 damask_%s_restart.hdf5"%(loadfilenameold,loadfilename)
                    # subprocess.run(args,shell=True,capture_output=True) 

                    # args="cp damask_%s_restart.sta damask_%s_restart.sta"%(loadfilenameold,loadfilename)
                    # subprocess.run(args,shell=True,capture_output=True) 

                    self.loading_discretization(step + 1, loadfilename)
                    print('Restart damask simulation from increment-%d' % (step * self._increments))
                    args = "DAMASK_grid -g damask.vti -l %s.yaml -r %d > %s.log" % (
                    self._loadfilename, step * self._increments, self._loadfilename)
                    print('Damask command is:', args)
                    subprocess.run(args, shell=True, capture_output=True)
                    resultfile = 'damask_%s.hdf5' % (self._loadfilename)

                    filename = os.path.join(self.working_directory, resultfile)
                    myresults = Result(filename)
                    r_unprotected = myresults.view(protected=False)
                    r_unprotected.remove('sigma')
                    r_unprotected.remove('sigma_vM')
                    r_unprotected.remove('epsilon_V^0.0(F)')
                    r_unprotected.remove('epsilon_V^0.0(F)_vM')
                    myresults.add_stress_Cauchy()
                    myresults.add_strain()
                    myresults.add_equivalent_Mises('sigma')
                    myresults.add_equivalent_Mises('epsilon_V^0.0(F)')

            self._rollresults.append(myresults)
            stress = self.average_spatio_temporal_tensors_new(myresults, 'sigma')
            strain = self.average_spatio_temporal_tensors_new(myresults, 'epsilon_V^0.0(F)')
            # self.output.stress_von_Mises = self.average_spatio_temporal_tensors_new(myresults,'sigma_vM')
            # self.output.strain_von_Mises = self.average_spatio_temporal_tensors_new(myresults,'epsilon_V^0.0(F)_vM')
            # self.plot_stress_strain(von_mises=True)
            stress_von_Mises = self.average_spatio_temporal_tensors_new(myresults, 'sigma_vM')
            strain_von_Mises = self.average_spatio_temporal_tensors_new(myresults, 'epsilon_V^0.0(F)_vM')
            self.plot_vonmises(strain_von_Mises, stress_von_Mises)
            self._vonMises_stress.append(stress_von_Mises)
            self._vonMises_strain.append(strain_von_Mises)

        inp = open('vonMises.csv', 'w')
        inp.write("vonMises strain, vonMises stress\n")
        for i in range(len(self._vonMises_stress[0])):
            str = '%14.5e,%14.5e\n' % (self._vonMises_strain[0][i], self._vonMises_stress[0][i])
            inp.write(str)
        inp.close()

    def savecurrentloading(self):
        """
        save the old loading configuration before restart
        """
        self._loading_old = self._loading

    def updateloading(self):
        """
        save the old loading configuration before restart
        """
        self._write_loading()

    def temporal_spatial_shape(self, name):
        property_dict = self._results.get(name)
        shape_list = [len(property_dict)]
        for shape in property_dict[list(property_dict.keys())[0]].shape:
            shape_list.append(shape)
        return tuple(shape_list)

    def average_spatio_temporal_tensors(self, name):
        _shape = self.temporal_spatial_shape(name)
        temporal_spatial_array = np.empty(_shape)
        property_dict = self._results.get(name)
        i = 0
        for key in property_dict.keys():
            temporal_spatial_array[i] = property_dict[key]
            i = i + 1
        return np.average(temporal_spatial_array, axis=1)

    #################################################################
    def temporal_spatial_shape_new(self, results, name):
        property_dict = results.get(name)
        shape_list = [len(property_dict)]
        for shape in property_dict[list(property_dict.keys())[0]].shape:
            shape_list.append(shape)
        return tuple(shape_list)

    def average_spatio_temporal_tensors_new(self, results, name):
        _shape = self.temporal_spatial_shape_new(results, name)
        temporal_spatial_array = np.empty(_shape)
        property_dict = results.get(name)
        i = 0
        for key in property_dict.keys():
            temporal_spatial_array[i] = property_dict[key]
            i = i + 1
        return np.average(temporal_spatial_array, axis=1)

    def temporal_spatial_shape_regrid(self, name):
        property_dict = self._regrid_results.get(name)
        shape_list = [len(property_dict)]
        for shape in property_dict[list(property_dict.keys())[0]].shape:
            shape_list.append(shape)
        return tuple(shape_list)

    def average_spatio_temporal_tensors_regrid(self, name):
        _shape = self.temporal_spatial_shape_regrid(name)
        temporal_spatial_array = np.empty(_shape)
        property_dict = self._regrid_results.get(name)
        i = 0
        for key in property_dict.keys():
            temporal_spatial_array[i] = property_dict[key]
            i = i + 1
        return np.average(temporal_spatial_array, axis=1)

    ########################################################################
    ### for openphase
    ########################################################################
    def write_openphase_config(self, step, dt):
        """
        write the configuration file for openphase
        """
        str = """
        Standard Open Phase Input File
!!!All values in MKS (or properly scaled) units please!!!

$SimTtl         Simulation Title                        : Normal grain growth
$nSteps         Number of Time Steps                    : %8d
$FTime          Output to disk every (tSteps)           : 100
$STime          Output to screen every (tSteps)         : 100
$LUnits         Units of length                         : m
$TUnits         Units of time                           : s
$MUnits         Units of mass                           : kg
$EUnits         Energy units                            : J
$Nx             System Size in X Direction              : %d
$Ny             System Size in Y Direction              : %d
$Nz             System Size in Z Direction              : %d
$dt             Initial Time Step                       : %14.5e
$IWidth         Interface Width (in grid points)        : 4.5
$dx             Grid Spacing                            : %14.5e
$nOMP           Number of OpenMP Threads                : 12
$Restrt         Restart switch (Yes/No)                 : No
$tStart         Restart at time step                    : 0
$tRstrt         Restart output every (tSteps)           : 10000



@ChemicalProperties

$Phase_0  Name of Phase 0     :   Phase1



@BoundaryConditions

Boundary Conditions Input Parameters:

$BC0X   X axis beginning boundary condition  : Periodic
$BCNX   X axis far end boundary condition    : Periodic

$BC0Y   Y axis beginning boundary condition  : Periodic
$BCNY   Y axis far end boundary condition    : Periodic

$BC0Z   Z axis beginning boundary condition  : Periodic
$BCNZ   Z axis far end boundary condition    : Periodic



@InterfaceEnergy

$Sigma_0_0  Interface energy       : 0.24
$Eps_0_0    Interface anisotropy  : 0.0



@InterfaceMobility

$Mu_0_0      Interface mobility      : 4.0e-9
$Eps_0_0     Interface anisotropy    : 0.0
$AE_0_0      Activation energy       : 0.0
        """ % (step, self._grid.cells[0], self._grid.cells[1], self._grid.cells[2], dt, np.min(self._grid.size))

        print('min size is:', np.min(self._grid.size))
        self.openphase_config = str
        cwd = os.getcwd()  # get the current dir
        os.chdir(self.working_directory)  # cd into the working dir
        inp = open('ProjectInput.opi', 'w+')
        inp.write(self.openphase_config)
        inp.close()
        self._grid.save_ASCII('openphase.grains')
        self._grid.save('openphase.vti', compress=False)
        os.chdir(cwd)  # cd back to the notebook dir

    def run_openphase(self):
        """
        execute the open phase simulation
        """
        cwd = os.getcwd()  # get the current dir
        os.chdir(self.working_directory)  # cd into the working dir
        import subprocess
        print("running openphase from ", os.getcwd())
        args = "Recrystallization ProjectInput.opi > openphase.log"
        subprocess.run(args, shell=True, capture_output=True)
        os.chdir(cwd)  # cd back to the notebook dir
