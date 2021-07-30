__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)

# To do in next pull request!
#Refactor the code according to Liams suggestion. 

from pyiron_base import GenericJob, GenericParameters, InputList, DataContainer
import numpy as np
import matplotlib.pyplot as plt
from damask import Grid
from damask import Result
from damask import seeds
import pyvista as pv
import h5py
import os
from os.path import join

__author__ = "Muhammad Hassani, Ahmed Aslam"
__version__ = "1.0"
__maintainer__ = "Ahmed Aslam"
__email__ = "hassani@mpie.de, aslam@mpie.de"

class DamaskMaterial:
    def __init__(self):
        self._material = DataContainer()
        self._path = None

    @property
    def value(self):
        return self._material

    @value.setter
    def value(self, mat):
        self._material = mat

    def read(self, path, wrap=True):
        self._material = self._material.read(file_name=path, wrap=wrap)
        self._path = path

    def config_material(self):
        pass

    def write(self, path):
        self._path = path
        self._material.write(file_name=path)


class DamaskLoad:
    def __init__(self):
        self._load = DataContainer()
        self._path = None

    @property
    def value(self):
        return self._load

    @value.setter
    def value(self, load):
        self._load = load

    def config_load(self):
        pass

    def read(self, path, wrap=True):
        self._path = path
        self._load.read(file_name=path, wrap=wrap)

    def write(self, path):
        self._path = path
        self._load.write(path)


class DamaskGeometry:
    def __init__(self):
        self._geom = DataContainer()
        self._path = None

    @property
    def value(self):
        return self._geom

    @value.setter
    def value(self, geom):
        self._geom = geom

    def config_geom(self, size, grains, grid):
        seed = seeds.from_random(size, grains)
        new_geom = Grid.from_Voronoi_tessellation(grid, size, seed)
        self._geom.size = size
        self._geom.grid = grid
        self._geom.grains = grains
        # new_geom.save_ASCII(os.path.join(self.working_directory, "damask.vtr"))
        self._path = os.path.join(self.working_directory, "damask.vtr")
        new_geom.save(self._path)

    def read(self, path, wrap=True):
        self._path = path
        self._geom.read(file_name=path, wrap=wrap)

    def write(self, path):
        self._path = path
        self._geom.write(path)


class DAMASKjob(GenericJob):
    def __init__(self, project, job_name):
        #TODO: check why we can't set self.output to a DataContainer
        super(DAMASKjob, self).__init__(project, job_name)
        self.input = DataContainer()
        self._material = None 
        self._loading = None
        self._geometry = None
        self._damask_results = None        
        self._executable_activate()
        
    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, new_mat):
        if not isinstance(new_mat, DamaskMaterial):
            raise TypeError("The given material is not of type DamaskMaterial")
        self._material = new_mat
        self.input.material = new_mat

    @property
    def loading(self):
        return self._loading
    
    @loading.setter
    def loading(self, new_load):
        if not isinstance(new_load, DamaskLoad):
            raise TypeError("The given loading is not of type DamaskLoad")
        self._loading = new_load
        self.input.loading = new_load

    def write_input(self):
        # TODO: needs path
        # TODO: _geom has its own write/config function, use that one.
        #os.chdir(self.working_directory)
        load_path = os.path.join(os.chdir(self.working_directory),'tesionX.yaml')
        self._loading.write(file_name = load_path)
#        self._geometry.config_geom(self, size, grains, grid)
#        self._geometry.write()
        self._material.write()
             
    def collect_output(self): 
        with self.project_hdf5.open("output/generic") as h5out: 
            h5out["stress"] = self.stress()
            h5out["strain"] = self.strain()

    def to_hdf(self, hdf=None, group_name=None):
        #TODO: check the function in the original function
        self.input.to_hdf(hdf=self._hdf5, group_name='input')
#        self.output.to_hdf(hdf=self._hdf5, group_name='output')

    def load_results(self, file_name):
        if self._damask_results is None:
            self._damask_results = self.output
            self._damask_results.add_stress_Cauchy()
            self._damask_results.add_strain()
            self._damask_results.add_equivalent_Mises('sigma')
            self._damask_results.add_equivalent_Mises('epsilon_V^0.0(F)')
            self._damask_results.add_calculation('avg_sigma', "np.average(#sigma_vM#)")
            self._damask_results.add_calculation('avg_epsilon', "np.average(#epsilon_V^0.0(F)_vM#)")
            self._damask_results.save_VTK(['sigma', 'epsilon_V^0.0(F)', 'sigma_vM', 'epsilon_V^0.0(F)_vM'])
        return self._damask_results
    
    
    @property
    def output(self):
        self._file_name = os.path.join(self.working_directory, "damask_tensionX.hdf5")
        return Result(self._file_name)
    
    def stress(self):
        """
        return the stress as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        output = self.output
        self.load_results(self._file_name)
        if output is not None:
            stress_path = output.get_dataset_location('avg_sigma')
            stress = np.zeros(len(stress_path))
            hdf = h5py.File(output.fname)
            for count,path in enumerate(stress_path):
                stress[count] = np.array(hdf[path])
            stress = np.array(stress)/1E6
            return stress

    def strain(self):
        """
        return the strain as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        output = self.output
        self.load_results(self._file_name)
        if output is not None:
            stress_path = output.get_dataset_location('avg_sigma')
            strain = np.zeros(len(stress_path))
            hdf = h5py.File(output.fname)
            for count,path in enumerate(stress_path):
                strain[count] = np.array(hdf[path.split('avg_sigma')[0]+ 'avg_epsilon'])
            return strain  
        
    def plot_stress_strain(self):
        """
        Plot the stress strain curve from the job file

        Parameters
        ----------
        self.stress
        self.strain
        """
        stress = self.stress()
        strain = self.strain()
        plt.plot(strain,stress,linestyle='-',linewidth='2.5')
        plt.xlabel(r'$\varepsilon_{VM} $',fontsize=18)
        plt.ylabel(r'$\sigma_{VM}$ (MPa)',fontsize=18)
     
    def mesh(self, inc=80):
        """
        Plot the stress strain curve from the job file

        Parameters
        ----------
        inc =  results per time increment
        self.stress
        self.strain
        """
        mesh = pv.read(os.path.basename(self.output_file.split('.')[0]) + f'_inc{inc}.vtr')
        #from itkwidgets import view
        #import itk
        #mesh
        #pl = pv.PlotterITK()
        #pl.add_mesh(mesh)
        #pl.show()
        return mesh
