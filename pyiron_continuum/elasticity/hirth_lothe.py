import numpy as np

class HirthLothe:
    def __init__(self, elastic_tensor, burgers_vector):
        self.elastic_tensor = elastic_tensor
        self.burgers_vector = burgers_vector
        self.fit_range = np.linspace(0, 1, 10)
        self._p = None
        self._Ak = None
        self._D = None
        
    def get_pmat(self, x):
        return (
            self.elastic_tensor[:,0,:,0]
            + np.einsum('...,ij->...ij', x, self.elastic_tensor[:,0,:,1]+self.elastic_tensor[:,1,:,0])
            + np.einsum('...,ij->...ij', x**2, self.elastic_tensor[:,1,:,1])
        )
    
    @property
    def p(self):
        if self._p is None:
            coeff = np.polyfit(self.fit_range, np.linalg.det(self.get_pmat(self.fit_range)), 6)
            self._p = np.roots(coeff)
            self._p = self._p[np.imag(self._p)>0]
        return self._p
    
    @property
    def Ak(self):
        if self._Ak is None:
            self._Ak = []
            for mat in self.get_pmat(self.p):
                values, vectors = np.linalg.eig(mat)
                self._Ak.append(vectors[np.absolute(values).argmin()])
            self._Ak = np.array(self._Ak)
        return self._Ak
    
    @property
    def D(self):
        if self._D is None:
            F = np.einsum('n,ij->nij', self.p, self.elastic_tensor[:,1,:,1])
            F += self.elastic_tensor[:,1,:,0]
            F = np.einsum('nik,nk->ni', F, self.Ak)
            F = np.concatenate((F, self.Ak), axis=0)
            F = np.concatenate((np.real(F), np.imag(F)), axis=-1)
            self._D = np.linalg.solve(B, np.concatenate((np.zeros(3), self.burgers_vector)))
            self._D = self._D[:3]+1j*self._D[3:]
        return self._D
    
    def get_z(self, positions):
        z = np.stack((np.ones_like(self.p), self.p), axis=-1)
        return np.einsum('nk,...k->...n', z, positions)

    def get_displacement(self, positions):
        return np.real(np.einsum('nk,n,...n->...k', self.Ak, self.D, np.log(self.get_z(positions))))/(2*np.pi)
    
    @property
    def dzdx(self):
        return np.stack((np.ones_like(self.p), self.p, np.zeros_like(self.p)), axis=-1)

    def get_strain(self, positions):
        strain = 0.5*np.real(np.einsum('ni,n,...n,nj->...ij', self.Ak, self.D, 1/self.get_z(positions), self.dzdx))
        strain = strain+np.einsum('...ij->...ji', strain)
        return strain/2/np.pi
