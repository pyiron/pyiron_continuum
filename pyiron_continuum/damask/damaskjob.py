# Reference : https://github.com/jan-janssen/pyiron-damask/blob/master/damaskjob.py
from pyiron_base import Project, GenericJob, GenericParameters
from pyiron_base import InputList 
import numpy as np
import matplotlib.pyplot as plt
from damask import Config
from damask import Geom
from damask import Result
from damask import seeds
import pyvista as pv
import h5py
import yaml
import os
import warnings


class DAMASKjob(GenericJob):
    def __init__(self, project, job_name):
        super(DAMASKjob, self).__init__(project, job_name)
        self.input = InputList()
        self.executable = self.executable_activate
        self._material = None 
        self._load = None
        self._geometry = None
        self._damask_results = None
        self.input.create_group('geometry')
        self.input.create_group('material')
        self.input.create_group('load')
        self.input['cores'] = 2 # solver decompose the geometry into multiple domains that get solved in parallel, defualt is 2
        self.executable_activate()
    
    @property
    def executable_activate(self):
        return "mpiexec -np "+str(self.input['cores'])+" DAMASK_grid -l tensionX.load -g damask.geom"
        
    @property
    def material(self):
        return self._material
    
    @material.setter
    def material(self, path=None):
        with open(path) as f:
            self._material = yaml.load(f, Loader=yaml.FullLoader)
        self.material_inputlist()
        
    @property
    def load(self):
        return self._load
    
    @load.setter
    def load(self, path=None):
        with open(path) as f:
            self._load = f.readlines()
        self.load_inputlist()
    
    @property
    def geometry(self):
        return self._geometry
    
    @geometry.setter
    def geometry(self, path=None):
        with open(path) as f:
            self._geometry = f.readlines()
        self.geom_inputlist()
    
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, path):
        self._path = path
        self.check_path()
       
    def check_path(self):
        if self._path is not None:
            try:
                with open(os.path.join(self._path, 'damask.geom')) as f:
                    geometry = f.readlines()
                self.geometry = geometry
                with open(os.path.join(self._path, 'material.yaml')) as f:
                    material = yaml.load(f, Loader=yaml.FullLoader)
                self.material = material
                with open(os.path.join(self._path, 'tensionX.load')) as f:
                    load = f.readlines()
                self.load = load
            except:
                pass
        else:
            pass
                
    def load_inputlist(self):
        if isinstance(self._load, type(None)):
            raise ValueError('job.load file not specified')
        else:
            self.input.load.deformation_gradient_rate = [' '.join(self._load[ind].split()[1:10]) for ind in range(len(self._load))]
            
            self.input.load.stress = [' '.join(self._load[ind].split()[11:20]) for ind in range(len(self._load))]
            self.input.load.time = [''.join(self._load[ind].split()[21]) for ind in range(len(self._load))]
            self.input.load.incs = [''.join(self._load[ind].split()[23]) for ind in range(len(self._load))]
            self.input.load.freq = [''.join(self._load[ind].split()[25]) for ind in range(len(self._load))]
            self.input.load.restart = [''.join(self._load[ind].split()[27]) for ind in range(len(self._load))]
            
    def material_inputlist(self):
        if isinstance(self._material, type(None)):
            raise ValueError('job.material file not specified')
        else:
            #self.input.create_group('material')
            self.input.material = InputList(self._material)
    
    def geom_inputlist(self):
        if self._geometry is not None:
            self.input.geometry.Header = int(self._geometry[0].split()[0])
            for ind, value in enumerate(self._geometry[1:self.input.geometry.Header+1]):
                string = value.split()[0]
                self.input.geometry[string] = value.split()[1:] 
            self.input.geometry.seed = seeds.from_random(np.array([float(self.input.geometry.size[ind]) for ind in [1, 3, 5]]), int(self.input.geometry.microstructures[0]))
            self.input.geometry.microstructure_indices = [list(np.float_(value.split())) for value in self._geometry[self.input.geometry.Header+1:]]
                
        elif (self.input.geometry.grid is not None) and (self.input.geometry.size is not None) and (self.input.geometry.microstructures is not None):
            
            self.input.geometry.seed = seeds.from_random(np.array(self.input.geometry.size), int(self.input.geometry.microstructures))
            
        else:
            raise ValueError('Either specify all input parameter (geometry/grid, geometry/size and geometry/grains) for geometry or provide job.geometry file')
    
    def load_write(self):
        load_paras = []
        for ind in range(len(self._load)):
            load_paras.append(['fdot']+[self.input.load.deformation_gradient_rate[ind]]+['stress']+[self.input.load.stress[ind]]+['time']+[self.input.load.time[ind]]+['incs']+[self.input.load.incs[ind]]+ ['freq']+ [self.input.load.freq[ind]]+ ['restart']+[self.input.load.restart[ind]]+['\n'])
        self._load_paras = [' '.join(value) for value in load_paras]
        with open(os.path.join(self.working_directory, 'tensionX.load'), "w") as f:
            f.writelines(self._load_paras)
            
    def material_write(self):
        material = self.input.material.to_builtin()
        with open(os.path.join(self.working_directory, 'material.yaml'), "w") as f:
                    yaml.dump(material , f)
    
    def geometry_write(self):
        if self._geometry is not None:
            header = [str(self.input.geometry.Header)+'\theader\n']
            grid = ['grid '+' '.join(self.input.geometry.grid)+'\n']
            size = ['size '+' '.join(self.input.geometry.size)+'\n']
            origin = ['origin '+' '.join(self.input.geometry.origin)+'\n']
            micro_struct = ['microstructures '+' '.join(self.input.geometry.microstructures)+'\n']
            homogenize = ['homogenization '+' '.join(self.input.geometry.homogenization)+'\n']
            micro_ind  = [' '.join([str(int(pp)) for pp in a])+'\n' for a in self.input.geometry.microstructure_indices]
            geom_new = header + grid + size + origin + micro_struct + homogenize + micro_ind
            with open(os.path.join(self.working_directory, 'damask.geom'), "w") as f:
                f.writelines(geom_new)   
        elif (self.input.geometry.seeds is not None):
            new_geom = Geom.from_Voronoi_tessellation(np.array([int(self.input.geometry.grid[ind]) for ind in [1, 3, 5]]), np.array([float(self.input.geometry.size[ind]) for ind in [1, 3, 5]]), self.input.geometry.seed)
            new_geom.save_ASCII(os.path.join(self.working_directory, "damask.geom"))
        #new_geom.save(os.path.join(, "damask"))
    
    def write_input(self):
        self.load_write()
        self.geometry_write()
        self.material_write()
             
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
            h5in["load"] = self._load
        self.status.finished = True

    def from_hdf(self, hdf=None, group_name=None): 
        super().from_hdf(
            hdf=hdf,
            group_name=group_name
        )
        with self.project_hdf5.open("input") as h5in:
            self.input.from_hdf(h5in)
            self._material = h5in["material"]
            self._load = h5in["load"]
    
    def load_results(self, file_name):
        if self._damask_results is None:
            self._damask_results = Result(file_name)
            self._damask_results.add_Cauchy()
            self._damask_results.add_strain_tensor()
            self._damask_results.add_Mises('sigma')
            self._damask_results.add_Mises('epsilon_V^0.0(F)')
            self._damask_results.add_calculation('avg_sigma',"np.average(#sigma_vM#)")
            self._damask_results.add_calculation('avg_epsilon',"np.average(#epsilon_V^0.0(F)_vM#)")
            #self._damask_results.add_ipf(np.array([0,0,1]))
            self._damask_results.save_vtk(['sigma','epsilon_V^0.0(F)','sigma_vM','epsilon_V^0.0(F)_vM'])
        return self._damask_results
    
    @property
    def output_file(self):
        file_name = os.path.join(self.working_directory, "damask_tensionX.hdf5")
        self.load_results(file_name)
        if self.status.finished and os.path.exists(file_name):
            return file_name
    
    @property
    def output(self):
        file_name = self.output_file
        if file_name is not None:
            return Result(file_name)
    
    def stress(self):
        """
        return the stress as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        output = self.output
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
