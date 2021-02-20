from pyiron_base import Project, GenericJob, GenericParameters
import numpy as np
import matplotlib.pyplot as plt
from damask import Config
from damask import Grid
from damask import Result
from damask import seeds
import h5py
import yaml
import os


class DAMASKjob(GenericJob):
    def __init__(self, project, job_name):
        super(DAMASKjob, self).__init__(project, job_name)
        self.input = GenericParameters(table_name="input")
        self.input['C11'] = 0
        self.input['C12'] = 0
        self.input['C44'] = 0
        self.input['grid'] = np.array([16,16,16])
        self.input['size'] = np.array([1.0,1.0,1.0])
        self.input['grains'] = 20
        self.executable = "DAMASK_grid -l tensionX.load -g damask.vtr"
        self._material = None 
        self._tension = None 
        
    @property
    def material(self):
        return self._material
    
    @material.setter
    def material(self, mat):
        self._material = mat
        
    @property
    def tension(self):
        return self._tension
    
    @tension.setter
    def tension(self, tension):
        self._tension = tension
        
    def write_input(self): 
        with open(os.path.join(self.working_directory, 'material.yaml'), "w") as f:
            yaml.dump(self._material, f)
        with open(os.path.join(self.working_directory, 'tensionX.load'), "w") as f:
            f.writelines(self._tension)
        seed = seeds.from_random(self.input['size'], self.input['grains'])
        new_geom = Grid.from_Voronoi_tessellation(self.input['grid'], self.input['size'], seed)
        # new_geom.save_ASCII(os.path.join(self.working_directory, "damask.geom"))
        new_geom.save(os.path.join(self.working_directory, "damask"))
        C_matrix = [self.input['C11']*1e9, self.input['C12']*1e9, self.input['C44']*1e9]
        elasticity={}
        elasticity.update({'type': 'hooke'})
        elastic_constants = {'C_11': C_matrix[0], 'C_12': C_matrix[1], 'C_44': C_matrix[2]}
        elasticity.update(elastic_constants)
        mat = Config.load(os.path.join(self.working_directory, 'material.yaml'))
        mat['phase']['Aluminum']['elasticity'] = elasticity
        mat.save(os.path.join(self.working_directory, 'material.yaml'))
        
    def collect_output(self): 
        pass 
    
    def to_hdf(self, hdf=None, group_name=None): 
        super().to_hdf(
            hdf=hdf,
            group_name=group_name
        )
        with self.project_hdf5.open("input") as h5in:
            self.input.to_hdf(h5in)
            h5in["material"] = self._material
            h5in["tension"] = self._tension

    def from_hdf(self, hdf=None, group_name=None): 
        super().from_hdf(
            hdf=hdf,
            group_name=group_name
        )
        with self.project_hdf5.open("input") as h5in:
            self.input.from_hdf(h5in)
            self._material = h5in["material"]
            self._tension = h5in["tension"]
    
    @property
    def output_file(self):
        file_name = os.path.join(self.working_directory, "damask_tensionX.hdf5")
        if self.status.finished and os.path.exists(file_name):
            return file_name
    
    @property
    def output(self):
        file_name = self.output_file
        if file_name is not None:
            return Result(file_name)

    def eval_stress(self):
        """
        return the stress as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        file_name = self.output_file
        if file_name is not None:
            d = Result(file_name)
            stress_path = d.get_dataset_location('avg_sigma')
            stress = np.zeros(len(stress_path))
            hdf = h5py.File(d.fname)
            for count,path in enumerate(stress_path):
                stress[count] = np.array(hdf[path])
            stress = np.array(stress)/1E6
            return stress

    def eval_strain(self):
        """
        return the strain as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        file_name = self.output_file
        if file_name is not None:
            d = Result(file_name)
            stress_path = d.get_dataset_location('avg_sigma')
            strain = np.zeros(len(stress_path))
            hdf = h5py.File(d.fname)
            for count,path in enumerate(stress_path):
                strain[count] = np.array(hdf[path.split('avg_sigma')[0]     + 'avg_epsilon'])

            return strain  
        
    def plot(self):
        """
        Plot the stress strain curve from the job file

        Parameters
        ----------
        job_file : str
        Name of the job_file
        """
        file_name = self.output_file
        if file_name is not None:
            d = Result(file_name)
            stress_path = d.get_dataset_location('avg_sigma')
            stress = np.zeros(len(stress_path))
            strain = np.zeros(len(stress_path))
            hdf = h5py.File(d.fname)
            for count, path in enumerate(stress_path):
                stress[count] = np.array(hdf[path])
                strain[count] = np.array(hdf[path.split('avg_sigma')[0]     + 'avg_epsilon'])

            stress = np.array(stress)/1E6
            plt.plot(strain,stress,linestyle='-',linewidth='2.5')
            plt.xlabel(r'$\varepsilon_{VM} $',fontsize=18)
            plt.ylabel(r'$\sigma_{VM}$ (MPa)',fontsize=18)