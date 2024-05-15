import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
import subprocess
import warnings

from pyiron_continuum.damask.damaskjob import DAMASK
import pyiron_continuum.damask.regrid as rgg
from damask import YAML


class ROLLING(DAMASK):
    """ """

    def __init__(self, project, job_name):
        """Create a new DAMASK type job for rolling"""
        super().__init__(project=project, job_name=job_name)
        self.input.reduction_height = None
        self.input.reduction_speed = None
        self.input.reduction_outputs = None
        self.input.regrid = False
        self.input.executable_name = ""
        self.input.RollingInstance = 1
        self.regrid_geom_name = None
        self.input.regrid_scale = 1.025
        self.output.ResultsFile = []

    def _join_path(self, path, return_str=True):
        file_path = Path(self.working_directory) / path
        if not return_str:
            file_path = str(file_path)
        return file_path

    def loading_discretization(self, rolltimes, filename):
        time = (
            rolltimes
            * self._height_reduction
            / (self._rolling_speed * self._number_passes)
        )

        self.load_case = YAML(solver={"mechanical": "spectral_basic"}, loadstep=[])
        self.load_case["loadstep"].append(
            self.get_loadstep(
                self.get_dot_F(self._rollling_speed), time, self._increments * rolltimes
            )
        )
        self.load_case.save(self._join_path(filename + ".yaml"))
        print(self.load_case)

    @property
    def reduction_time(self):
        return self.input.reduction_height / self.input.reduction_speed

    def executeRolling(
        self,
        reduction_height=None,
        reduction_speed=None,
        reduction_outputs=None,
        regrid=None,
        damask_exe=None,
    ):
        warnings.warn("`executeRolling` is deprecated; use `run`")
        if reduction_height is not None:
            self.input.reduction_height = reduction_height
        if reduction_speed is not None:
            self.input.reduction_speed = reduction_speed
        if reduction_outputs is not None:
            self.input.reduction_outputs = reduction_outputs
        if regrid is not None:
            self.input.regrid = regrid
        if damask_exe is not None:
            self.input.damask_exe = damask_exe
        self._execute_rolling()

    def write_input(self):
        if self.input.RollingInstance == 1:
            super().write_input()
            self.load_case = YAML(solver={"mechanical": "spectral_basic"}, loadstep=[])
        self.load_case["loadstep"].append(
            self.get_loadstep(
                self.get_dot_F(self.input.reduction_speed),
                self.reduction_time,
                self.input.reduction_outputs,
            )
        )
        self.load_case.save(self._join_path(self._load_name + ".yaml"))
        if self.input.regrid and self.input.RollingInstance > 1:
            self.regridding(self.input.regrid_scale)

    # To be replaced by run_static
    def _execute_rolling(self):
        if self.input.RollingInstance == 1:
            # Most useless five lines to be removed ASAP
            print("working dir:", self.working_directory)
            Path(self.working_directory).mkdir(parents=True, exist_ok=True)
            for file_path in Path(self.working_directory).glob("*"):
                if file_path.is_file():
                    file_path.unlink()

        self.write_input()
        self._execute_damask(self.input.damask_exe)
        self.collect_output()
        self.input.RollingInstance += 1

    @property
    def _log_name(self):
        if self.input.RollingInstance == 1:
            return "FirstRolling"
        return f"Rolling-{self.input.RollingInstance}"

    @property
    def _load_name(self):
        if self.input.RollingInstance == 1:
            return "load"
        return "load_rolling%d" % (self.input.RollingInstance)

    @property
    def _load_name_old(self):
        if self.input.RollingInstance == 2:
            return "load"
        return "load_rolling%d" % (self.input.RollingInstance - 1)

    @property
    def geom_name(self):
        if self.input.regrid and self.regrid_geom_name is not None:
            return self.regrid_geom_name
        return "damask"

    def _execute_damask(self, damask_exe):
        if len(damask_exe) < 11:
            damask_exe = "DAMASK_grid"
        args = f"{damask_exe} -g {self.geom_name}.vti -l {self._load_name}.yaml -m material.yaml > {self._log_name}.log"
        print("Start the rolling-%d test ..." % (self.input.RollingInstance))
        print("CMD=", args)
        os.chdir(self.working_directory)
        subprocess.run(args, shell=True, capture_output=True)
        print(f"{self._log_name} test is done !")
        self.output.ResultsFile.append(
            f"{self.geom_name}_{self._load_name}_material.hdf5"
        )

    @staticmethod
    def get_dot_F(reduction_speed):
        return [["x", 0, 0], [0, 0, 0], [0, 0, -1.0 * reduction_speed]]

    @staticmethod
    def get_loadstep(dot_F, reduction_time, reduction_outputs, P=None):
        if P is None:
            P = [[0, "x", "x"], ["x", "x", "x"], ["x", "x", "x"]]
        return {
            "boundary_conditions": {"mechanical": {"P": P, "dot_F": dot_F}},
            "discretization": {"t": reduction_time, "N": reduction_outputs},
            "f_out": 5,
            "f_restart": 5,
        }

    def postProcess(self):
        pass

    def collect_output(self):
        self._load_results(self.output.ResultsFile[-1])

    @property
    def ResultsFile(self):
        warnings.warn("Use job.output.ResultsFile instead")
        return self.output.ResultsFile

    def plotStressStrainCurve(self, xmin, xmax, ymin, ymax):
        plt.plot(self.output.strain_von_Mises, self.output.stress_von_Mises)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    def regridding(self, scale):
        regrid = rgg.Regrid(
            self.working_directory,
            self.geom_name,
            self._load_name_old,
            seed_scale=scale,
        )
        self.regrid_grid = regrid.grid
        self.regrid_geom_name = regrid.regrid_geom_name

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
