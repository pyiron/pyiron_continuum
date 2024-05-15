import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
from pyiron_base import ImportAlarm

with ImportAlarm(
    "DAMASK functionality requires the `damask` module (and its dependencies) specified as extra"
    "requirements. Please install it and try again."
) as damask_alarm:
    from damask import YAML, ConfigMaterial
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
        self.input.job_names = []
        self.input.regrid_scale = 1.025
        self.regrid_geom_name = None
        self.input.load_case = YAML(solver={"mechanical": "spectral_basic"}, loadstep=[])
        self.output.results_file = []
        self.output.job_names = []
        self.executable = (
            "DAMASK_grid -g damask.vti -l loading.yaml -m material.yaml > rolling.log"
        )

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

        self.input.load_case = YAML(solver={"mechanical": "spectral_basic"}, loadstep=[])
        self.input.load_case["loadstep"].append(
            self.get_loadstep(
                self.get_dot_F(self._rollling_speed), time, self._increments * rolltimes
            )
        )
        self.input.load_case.save(self._join_path(filename + ".yaml"))
        print(self.input.load_case)

    @property
    def reduction_time(self):
        return self.input.reduction_height / self.input.reduction_speed

    def set_rolling(
        self,
        reduction_height=None,
        reduction_speed=None,
        reduction_outputs=None,
        regrid=None,
    ):
        if reduction_height is not None:
            self.input.reduction_height = reduction_height
        if reduction_speed is not None:
            self.input.reduction_speed = reduction_speed
        if reduction_outputs is not None:
            self.input.reduction_outputs = reduction_outputs
        if regrid is not None:
            self.input.regrid = regrid

    def write_input(self):
        super().write_input()
        self.input.load_case["loadstep"].append(
            self.get_loadstep(
                self.get_dot_F(self.input.reduction_speed),
                self.reduction_time,
                self.input.reduction_outputs,
            )
        )
        self.input.load_case.save(self._join_path("loading.yaml"))
        if self.input.regrid and len(self.input.job_names) > 0:
            self.regridding(self.input.regrid_scale)

    @property
    def geom_name(self):
        if self.input.regrid and self.regrid_geom_name is not None:
            return self.regrid_geom_name
        return "damask"

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

    def collect_output(self):
        self.output.job_names.append(self.job_name)
        super().collect_output()
        self.to_hdf()

    def plotStressStrainCurve(self, xmin, xmax, ymin, ymax):
        plt.plot(self.output.strain_von_Mises, self.output.stress_von_Mises)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    def regridding(self, scale):
        regrid = rgg.Regrid(
            self.working_directory,
            self.geom_name,
            self.restart_file_list[0],
            seed_scale=scale,
        )
        self.regrid_grid = regrid.grid
        self.regrid_geom_name = regrid.regrid_geom_name

    def restart(self, job_name=None, job_type=None):
        new_job = super().restart(job_name=job_name, job_type=job_type)
        new_job.storage.input = self.storage.input.copy()
        new_job.input.job_names = self.output.job_names
        new_job.input.material = ConfigMaterial(**new_job.input.material)
        new_job.input.load_case = YAML(**self.input.load_case)
        new_job.restart_file_list.append(self._join_path("damask_loading_material.hdf5"))
        return new_job

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
