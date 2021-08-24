from pyiron_base import GenericJob, DataContainer
import numpy as np
import matplotlib.pyplot as plt
from damask import Grid
from damask import Result
from damask import seeds
import damask
import pyvista as pv
import h5py
import os
import subprocess

class DAMASK(GenericJob):
    def __init__(self, project, job_name):
        super(DAMASK, self).__init__(project, job_name)
        self.input = DataContainer()
        self.output = DataContainer()
        self._material = None 
        self._loading = None
        self._geometry = None
        self._damask_results = None
        self._damask_cpus=2
        self.input.create_group('geometry')
        self.input.create_group('material')
        self.input.create_group('loading')
        self.output.create_group('stress')
        self.output.create_group('strain')
        #self._executable_activate()

        # set up some default settings
        self.input.geometry['grains']=4
        self.input.geometry['size']=1.0e-5 
        self.input.geometry['grids']=16
        self.input.geometry['vtifile']=''

        # set up default settings for load
        self.input.loading['loadfile']=''
        self.input.loading['time']=0.0
        self.input.loading['step']=1
        self.input.loading['interval']=1

        
    def generatematerialyaml(self,materialphase='Aluminum',homogenizationmethod='SX',c11=106.75e+9,c12=60.41e+9,c44=28.34e+9):
        r=damask.Rotation.from_random(self.input.geometry['grains'])
        ### config the material yaml
        config=damask.ConfigMaterial()
        config=config.material_add(O=r,phase=materialphase,homogenization=homogenizationmethod)
        config.save()
        # for homogenization
        str='homogenization:\n  %s:\n    N_constituents: 1\n    mechanical: {type: pass}\n'%(homogenizationmethod)
        fin = open("material.yaml", "rt")
        data = fin.read()
        data = data.replace('homogenization: {}',str)
        # for phase
        str='phase:\n  %s:\n    lattice: cF\n    mechanical:\n      output: [F, P, F_e, F_p, L_p, O]\n      elastic: {type: Hooke, C_11: %14.5e, C_12: %14.5e, C_44:%14.5e }\n      plastic:\n        type: phenopowerlaw\n        N_sl: [12]\n        a_sl: 2.25\n        atol_xi: 1.0\n        dot_gamma_0_sl: 0.001\n        h_0_sl-sl: 75.e+6\n        h_sl-sl: [1, 1, 1.4, 1.4, 1.4, 1.4, 1.4]\n        n_sl: 20\n        output: [xi_sl]\n        xi_0_sl: [31.e+6]\n        xi_inf_sl: [63.e+6]\n'%(materialphase,c11,c12,c44)
        data = data.replace('phase: {}',str)
        fin.close()
        fin = open("material.yaml", "wt")
        fin.write(data)
        fin.close()

        self._material=self.input.material.read('material.yaml')
    

    def Run(self):
        str="DAMASK_grid --geom %s --load %s"%(self.input.geometry['vtifile'],self.input.loading['loadfile'])
        subprocess.run(str,shell=True)

    @property
    def material(self):
        return self._material
    
    @material.setter
    def material(self, path=None):
        self._material = self.input.material.read(path)
        
    @property
    def loading(self):
        return self._loading
    
    @loading.setter
    def loading(self, path=None):
        self._loading = self.input.loading.read(path)
    
    def loading_write(self):
        self.input.loading.write('loading.yaml')
        
    def material_write(self):
        self.input.material.write('material.yaml')
   
    def generategeometryyaml(self,size=1.0e-5,grains=4,grids=16):
        self.input.geometry['size']=size
        self.input.geometry['grains']=grains
        self.input.geometry['grids']=grids
        self.input.geometry['vtifile']=''
        size=np.ones(3)*size
        cells=[grids,grids,grids]
        N_grains=grains

        # for voronoi seed
        seeds=damask.seeds.from_random(size,N_grains,cells)
        new_geom=damask.Grid.from_Voronoi_tessellation(cells,size,seeds)
        # save our geometry file to a vti file
        self.input.geometry['vtifile']=f'Polycystal_{N_grains}_{cells[0]}x{cells[1]}x{cells[2]}.vti'

        new_geom.save(f'Polycystal_{N_grains}_{cells[0]}x{cells[1]}x{cells[2]}')
        
        self.executable="DAMASK_grid --geom %s --load loading.yaml"%(self.input.geometry['vtifile'])

    def generateloadyaml(self,loadfilename='tensile.yaml',loadtype='tensile',time=1.0,step=10,interval=2):
        self.input.loading['loadfile']=loadfilename
        self.input.loading['time']=time
        self.input.loading['step']=step
        self.input.loading['interval']=interval
        load_case = damask.Config(solver={'mechanical':'spectral_basic'},
                          loadstep=[])
        if loadtype=='tensile':
            dotF = [1.0e-3,0,0, 0,'x',0,  0,0,'x']
            P = ['x' if i != 'x' else 0 for i in dotF]

            load_case['loadstep'].append({'boundary_conditions':{},
                'discretization':{'t':time,'N':step},'f_out':interval})

            load_case['loadstep'][0]['boundary_conditions']['mechanical'] = \
                    {'dot_F':[dotF[0:3],dotF[3:6],dotF[6:9]],
                     'P':[P[0:3],P[3:6],P[6:9]]}

        load_case.save(loadfilename)
        
        self.executable="DAMASK_grid --geom %s --load %s"%(self.input.geometry['vtifile'],self.input.loading['loadfile'])


    def SaveResult(self):
        geometryfile=self.input.geometry['vtifile']
        loadyaml=self.input.loading['loadfile']
        resultfile=geometryfile[:len(geometryfile)-4]+'_'+loadyaml[:len(loadyaml)-5]+'.hdf5'
        print('Result is saved to:',resultfile)
        self.resultfile=resultfile
        result=damask.Result(self.resultfile)
        #result.enable_user_function(avg_sigma)

        # add Cauchy stress
        #result.add_stress_Cauchy()
        # add pk2 stress
        #result.add_stress_second_Piola_Kirchhoff()
        # add strain
        #result.add_strain()
        # add equivalent von Mises stress
        #result.add_equivalent_Mises('sigma')
        #result.add_equivalent_Mises('epsilon_V^0.0(F)')
        #result.add_calculation('avg_sigma',"np.average(#sigma_vM#)")
        #result.add_calculation('avg_epsilon',"np.average(#epsilon_V^0.0(F)_vM#)")

        # save result to vtk file
        #result.export_VTK()
        #self.result=result
        
        #result.enable_user_function(damask.mechanics.equivalent_stress_Mises)

        result.add_stress_Cauchy()
        result.add_strain()
        result.add_equivalent_Mises('sigma')
        result.add_equivalent_Mises('epsilon_V^0.0(F)')
        
        #result.add_calculation('equivalent_stress_Mises',"np.average(#sigma_vM#)")
        #result.add_calculation('avg_epsilon',"np.average(#epsilon_V^0.0(F)_vM#)")
        #result.export_VTK(['sigma','epsilon_V^0.0(F)','sigma_vM','epsilon_V^0.0(F)_vM'])
        result.export_VTK()
        self.result=result

    def PlotStressStrain(self):
        """
        Plot the stress strain curve from the job file
        Parameters
        ----------
        ax (matplotlib axis /None): axis to plot on (created if None)
        """
        ### for stress
        stress_path = self.result.get_dataset_location('avg_sigma')
        stress = np.zeros(len(stress_path))
        hdf = h5py.File(self.result.fname)
        for count,path in enumerate(stress_path):
            stress[count] = np.array(hdf[path])
        self.output.stress = np.array(stress)/1E6

        ### for strain
        stress_path = self.result.get_dataset_location('avg_sigma')
        strain = np.zeros(len(stress_path))
        hdf = h5py.File(self.results.fname)
        for count,path in enumerate(stress_path):
            strain[count] = np.array(hdf[path.split('avg_sigma')[0]+ 'avg_epsilon'])
        self.output.strain = strain

        fig, ax = plt.subplots()

        ax.plot(self.output.strain, self.output.stress, linestyle='-', linewidth='2.5')
        ax.grid(True)
        ax.set_xlabel(r'$\varepsilon_{VM} $', fontsize=18)
        ax.set_ylabel(r'$\sigma_{VM}$ (MPa)', fontsize=18)
        return fig, ax

    def geometry_write(self):
        #seed = seeds.from_random(self.input.geometry['size'], self.input.geometry['grains'])
        #new_geom = Grid.from_Voronoi_tessellation(self.input.geometry['grids'], self.input.geometry['size'], seed)
        #seeds=damask.seeds.from_random(size,N_grains,cells)
        #new_geom=damask.Grid.from_Voronoi_tessellation(cells,size,seeds)
        i=1
        #new_geom.save(os.path.join(self.working_directory, "damask"))
    
    def write_input(self):
        os.chdir(self.working_directory)
        self.loading_write()
        self.geometry_write()
        self.material_write()
             
    def collect_output(self):
        self.load_results()
        self.stress()
        self.strain()
    
    def load_results(self, file_name="damask_tensionX.hdf5"):
        """
        Open ‘damask_tensionX.hdf5’,add the Mises equivalent of the Cauchy stress, and export it to VTK (file).
        """
        if self._damask_results is None:
            self._file_name = os.path.join(self.working_directory, file_name)
            self._damask_results = Result(self._file_name)
            self._damask_results.add_stress_Cauchy()
            self._damask_results.add_strain()
            self._damask_results.add_equivalent_Mises('sigma')
            self._damask_results.add_equivalent_Mises('epsilon_V^0.0(F)')
            self._damask_results.add_calculation('avg_sigma',"np.average(#sigma_vM#)")
            self._damask_results.add_calculation('avg_epsilon',"np.average(#epsilon_V^0.0(F)_vM#)")
            self._damask_results.save_VTK(['sigma','epsilon_V^0.0(F)','sigma_vM','epsilon_V^0.0(F)_vM'])
        return self._damask_results
    
    def stress(self):
        """
        return the stress as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        self.load_results()
        if self._damask_results is not None:
            stress_path = self._damask_results.get_dataset_location('avg_sigma')
            stress = np.zeros(len(stress_path))
            hdf = h5py.File(self._damask_results.fname)
            for count,path in enumerate(stress_path):
                stress[count] = np.array(hdf[path])
            self.output.stress = np.array(stress)/1E6

    def strain(self):
        """
        return the strain as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        self.load_results()
        if self._damask_results is not None:
            stress_path = self._damask_results.get_dataset_location('avg_sigma')
            strain = np.zeros(len(stress_path))
            hdf = h5py.File(self._damask_results.fname)
            for count,path in enumerate(stress_path):
                strain[count] = np.array(hdf[path.split('avg_sigma')[0]+ 'avg_epsilon'])
            self.output.strain = strain
    
    def plot_stress_strain(self, ax=None):
        """
        Plot the stress strain curve from the job file
        Parameters
        ----------
        ax (matplotlib axis /None): axis to plot on (created if None)
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.output.strain, self.output.stress, linestyle='-', linewidth='2.5')
        ax.grid(True)
        ax.set_xlabel(r'$\varepsilon_{VM} $', fontsize=18)
        ax.set_ylabel(r'$\sigma_{VM}$ (MPa)', fontsize=18)
        return fig, ax
    
    def load_mesh(self, inc=20):
        """
        Return the mesh for particular increment
        """
        mesh = pv.read(os.path.join(self.working_directory, self._file_name.split('.')[0] + f'_inc0{inc}.vtr'))
        return mesh
    
