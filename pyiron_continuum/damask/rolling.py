from ast import arg
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
import regrid as rgg
from damask import Result, YAML
from factory import DamaskLoading


class ROLLING(DAMASK):
    """ """

    def __init__(self, project, job_name):
        """Create a new DAMASK type job for rolling"""
        super().__init__(project=project, job_name=job_name)
        self.IsFirstRolling = True
        self.RollingInstance = 0

    def loading_discretization(self, rolltimes, filename):
        time = (
            rolltimes
            * self._height_reduction
            / (self._rolling_speed * self._number_passes)
        )

        self.load_case = YAML(solver={"mechanical": "spectral_basic"}, loadstep=[])
        dotF = [["x", 0, 0], [0, 0, 0], [0, 0, -1.0 * self._rolling_speed]]
        P = [[0, "x", "x"], ["x", "x", "x"], ["x", "x", "x"]]
        self.load_case["loadstep"].append(self.get_loadstep(P, dotF, time, self._increments * rolltimes))
        self._loadfilename = filename
        self._loading = self.load_case
        file_path = os.path.join(self.working_directory, filename + ".yaml")
        self._loading.save(file_path)
        # self.input.loading = self._loading
        print(self._loading)

    def executeRolling(
        self,
        reduction_height,
        reduction_speed,
        reduction_outputs,
        regrid=False,
        damask_exe="",
    ):
        if self.IsFirstRolling:
            # for the first rolling step, no regridding is required
            self.RollingInstance = 1
            self.IsFirstRolling = False
            self.ResultsFile = []

            print("working dir:", self.working_directory)
            if not os.path.exists(self.working_directory):
                os.makedirs(self.working_directory)

            # clean all the results file
            os.chdir(self.working_directory)
            args = "rm -rf *.vti *.yaml *.hdf5 *.log *.C_ref *.sta"
            subprocess.run(args, shell=True, capture_output=True)

            self._write_material()
            self._write_geometry()

            self.load_case = YAML(solver={"mechanical": "spectral_basic"}, loadstep=[])
            reduction_time = reduction_height / reduction_speed
            dotF = [["x", 0, 0], [0, 0, 0], [0, 0, -1.0 * reduction_speed]]
            P = [[0, "x", "x"], ["x", "x", "x"], ["x", "x", "x"]]
            self.load_case["loadstep"].append(self.get_loadstep(P, dotF, reduction_time, reduction_outputs))
            filename = "load"
            self._loadfilename = filename
            self._loading = self.load_case
            file_path = os.path.join(self.working_directory, filename + ".yaml")
            self._loading.save(file_path)
            print(self._loading)

            self.geom_name = "damask"
            self.load_name = "load"

            if len(damask_exe) < 11:
                args = f"DAMASK_grid -g {self.geom_name}.vti -l load.yaml -m material.yaml > FirstRolling.log"
            else:
                args = f"{damask_exe} -g {self.geom_name}.vti -l load.yaml -m material.yaml > FirstRolling.log"
            print("Start the first rolling test ...")
            os.chdir(self.working_directory)
            print("CMD=", args)
            subprocess.run(args, shell=True, capture_output=True)
            print("First rolling test is done !")
            self.ResultsFile.append(f"{self.geom_name}_{self.load_name}_material.hdf5")
        else:
            # for multiple rolling test
            self.RollingInstance += 1
            reduction_time = reduction_height / reduction_speed
            dotF = [["x", 0, 0], [0, 0, 0], [0, 0, -1.0 * reduction_speed]]
            P = [[0, "x", "x"], ["x", "x", "x"], ["x", "x", "x"]]
            self.load_case["loadstep"].append(self.get_loadstep(P, dotF, reduction_time, reduction_outputs))
            load_name = "load_rolling%d" % (self.RollingInstance)
            self.load_name_old = self.load_name
            self.load_name = load_name
            self._loading = self.load_case
            file_path = os.path.join(self.working_directory, load_name + ".yaml")
            self._loading.save(file_path)
            if regrid:
                self.load_name = self.load_name_old
                self.regridding(1.025)
                self.load_name = load_name
                self.geom_name = self.regrid_geom_name
                if len(damask_exe) < 11:
                    args = (
                        f"DAMASK_grid -g {self.regrid_geom_name}.vti -l {self.load_name}.yaml -m material.yaml > Rolling-%d.log"
                        % (self.RollingInstance)
                    )
                else:
                    args = (
                        f"{damask_exe} -g {self.regrid_geom_name}.vti -l {self.load_name}.yaml -m material.yaml > Rolling-%d.log"
                        % (self.RollingInstance)
                    )
            else:
                if len(damask_exe) < 11:
                    args = (
                        f"DAMASK_grid -g {self.geom_name}.vti -l {self.load_name}.yaml -m material.yaml > Rolling-%d.log"
                        % (self.RollingInstance)
                    )
                else:
                    args = (
                        f"{damask_exe} -g {self.geom_name}.vti -l {self.load_name}.yaml -m material.yaml > Rolling-%d.log"
                        % (self.RollingInstance)
                    )
            print("Start the rolling-%d test ..." % (self.RollingInstance))
            print("CMD=", args)
            os.chdir(self.working_directory)
            subprocess.run(args, shell=True, capture_output=True)
            print("Rolling-%d test is done !" % (self.RollingInstance))
            self.ResultsFile.append(f"{self.geom_name}_{self.load_name}_material.hdf5")

    @staticmethod
    def get_loadstep(P, dot_F, reduction_time, reduction_outputs):
        return {
            "boundary_conditions": {"mechanical": {"P": P, "dot_F": dot_F}},
            "discretization": {"t": reduction_time, "N": reduction_outputs},
            "f_out": 5,
            "f_restart": 5,
        }

    def postProcess(self):
        self._load_results(f"{self.geom_name}_{self.load_name}_material.hdf5")

    def plotStressStrainCurve(self, xmin, xmax, ymin, ymax):
        plt.plot(self.output.strain_von_Mises, self.output.stress_von_Mises)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    def regridding(self, scale):
        map_0to_rg, cells_rg, size_rg, increment_title = rgg.regrid_Geom(
            self.working_directory,
            self.geom_name,
            self.load_name,
            seed_scale=scale,
            increment="last",
        )

        self.regrid_grid, self.regrid_geom_name = rgg.write_RegriddedGeom(
            self.working_directory,
            self.geom_name,
            increment_title,
            map_0to_rg,
            cells_rg,
            size_rg,
        )

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
        """ % (
            step,
            self._grid.cells[0],
            self._grid.cells[1],
            self._grid.cells[2],
            dt,
            np.min(self._grid.size),
        )

        print("min size is:", np.min(self._grid.size))
        self.openphase_config = str
        cwd = os.getcwd()  # get the current dir
        os.chdir(self.working_directory)  # cd into the working dir
        inp = open("ProjectInput.opi", "w+")
        inp.write(self.openphase_config)
        inp.close()
        self._grid.save_ASCII("openphase.grains")
        self._grid.save("openphase.vti", compress=False)
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
