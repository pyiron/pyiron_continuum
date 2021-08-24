import numpy as np
from pyiron_continuum.elasticity.tools import *

class Isotropic:
    def __init__(self, poissons_ratio, shear_modulus, min_distance=0, optimize=True):
        self.poissons_ratio = poissons_ratio
        self.shear_modulus = shear_modulus
        self.min_dist = min_distance
        self.optimize = optimize
        self._A = None
        self._B = None

    @property
    def A(self):
        if self._A is None:
            self._A = (3-4*self.poissons_ratio)*self.B
        return self._A

    @property
    def B(self):
        if self._B is None:
            self._B = 1/(16*np.pi*self.shear_modulus*(1-self.poissons_ratio))
        return self._B

    def G(self, r):
        """ Green's function """
        R_inv = 1/np.linalg.norm(r, axis=-1)
        G = self.A*np.eye(3)+self.B*np.einsum(
            '...i,...j,...->...ij', r, r, R_inv**2, optimize=self.optimize
        )
        return np.einsum('...ij,...->...ij', G, R_inv)

    def G_fourier(self, k):
        """Fourier transform of the Green's function"""
        K = np.linalg.norm(k, axis=-1)
        if self.min_dist==0:
            return 4*np.pi*(
                self.A*np.einsum('...,ij->...ij', 1/K**2, np.eye(3))
                + self.B*np.einsum('...,ij->...ij', 1/K**2, np.eye(3))
                - 2*self.B*np.einsum('...,...i,...j->...ij',  1/K**3, k, k)
            )
        return 4*np.pi*(
            self.A*np.einsum('...,ij->...ij', np.cos(K*self.min_dist)/K**2, np.eye(3))
            + self.B*np.einsum(
                '...,ij->...ij', np.sin(K*self.min_dist)/(K**3*self.min_dist), np.eye(3)
            )
            + self.B*np.einsum(
                '...,...i,...j->nij',
                np.cos(K*self.min_dist)/K**4-3*np.sin(K*self.min_dist)/(K**5*self.min_dist),
                k, k, optimize=self.optimize)
        )

    def dG(self, r):
        """ first derivative of the Green's function """
        E = np.eye(3)
        R = np.linalg.norm(r, axis=-1)
        distance_condition = R<self.min_dist
        R[distance_condition] = 1
        r = np.einsum('...i,...->...i', r, 1/R)
        v = -self.A*np.einsum('ik,...j->...ijk', E, r)
        v += self.B*np.einsum('ij,...k->...ijk', E, r)
        v += self.B*np.einsum('jk,...i->...ijk', E, r)
        v -= 3*self.B*np.einsum('...i,...j,...k->...ijk', r, r, r, optimize=self.optimize)
        v = np.einsum('...ijk,...->...ijk', v, 1/R**2)
        v[distance_condition] *= 0
        return v

    def ddG(self, r):
        """ Second derivative of the Green's function """
        E = np.eye(3)
        R = np.linalg.norm(r, axis=-1)
        distance_condition = R<self.min_dist
        R[distance_condition] = 1
        r = np.einsum('...i,...->...i', r, 1/R)
        v = -self.A*np.einsum('ik,jl->ijkl', E, E)
        v = v+3*self.A*np.einsum('ik,...j,...l->...ijkl', E, r, r, optimize=self.optimize)
        v = v+self.B*np.einsum('il,jk->ijkl', E, E)
        v -= 3*self.B*np.einsum('il,...j,...k->...ijkl', E, r, r, optimize=self.optimize)
        v = v+self.B*np.einsum('ij,kl->ijkl', E, E)
        v -= 3*self.B*np.einsum('...i,...j,kl->...ijkl', r, r, E, optimize=self.optimize)
        v -= 3*self.B*np.einsum('ij,...k,...l->...ijkl', E, r, r, optimize=self.optimize)
        v -= 3*self.B*np.einsum('jk,...i,...l->...ijkl', E, r, r, optimize=self.optimize)
        v -= 3*self.B*np.einsum('jl,...i,...k->...ijkl', E, r, r, optimize=self.optimize)
        v += 15*self.B*np.einsum('...i,...j,...k,...l->...ijkl', r, r, r, r, optimize=self.optimize)
        v = np.einsum('...ijkl,...->...ijkl', v, 1/R**3)
        v[distance_condition] *= 0
        return v

    def get_greens_function(self, r, derivative=0, fourier=False):
        if fourier:
            return self.G_fourier(r)
        elif derivative == 0:
            return self.G(r)
        elif derivative == 1:
            return self.dG(r)
        elif derivative == 2:
            return self.ddG(r)
        else:
            raise ValueError('Derivative can be up to 2')

class Anisotropic:
    """
    Calculation of Green's functions (and their derivatives) for the anisotropic elasticity
    theory based on Barnett's approach. All notations follow Barnett's paper.

    [Link](https://doi.org/10.1002/pssb.2220490238)
    
    Notes:

    - In some cases this class can become extremely RAM-intensive. If possible do not keep
      the results in a variable.
    - If the medium is isotropic, use Isotropic instead, which has analytical solutions and is
      therefore much faster.
    """
    def __init__(self, elastic_constants, n_mesh=100, optimize=True):
        self.C = elastic_constants
        self.phi_range, self.dphi = np.linspace(0, np.pi, n_mesh, endpoint=False, retstep=True)
        self.optimize = optimize
        self.initialize()

    def initialize(self):
        self._zT = None
        self._F = None
        self._T = None
        self._z = None
        self._Ms = None
        self._MF = None

    @property
    def z(self):
        if self._z is None:
            self._z = np.einsum(
                'i...x,in->n...x',
                get_plane(self.T),
                [np.cos(self.phi_range), np.sin(self.phi_range)]
            )
        return self._z

    @property
    def Ms(self):
        if self._Ms is None:
            self._Ms = np.einsum(
                'ijkl,...j,...l->...ik', self.C, self.z, self.z, optimize=self.optimize
            )
            self._Ms = np.linalg.inv(self._Ms)
        return self._Ms

    @property
    def T(self):
        if self._T is None:
            self._T = normalize(self.r)
        return self._T

    @property
    def zT(self):
        if self._zT is None:
            self._zT = np.einsum('...p,...w->...pw', self.z, self.T)
            self._zT = self._zT+np.einsum('...ij->...ji', self._zT)
        return self._zT

    @property
    def F(self):
        if self._F is None:
            self._F = np.einsum(
                'jpnw,...ij,...nr,...pw->...ir', self.C, self.Ms, self.Ms, self.zT,
                optimize=self.optimize
            )
        return self._F

    @property
    def MF(self):
        if self._MF is None:
            self._MF = np.einsum('...ij,...nr->...ijnr', self.F, self.Ms)
            self._MF = self._MF+np.einsum('...ijnr->...nrij', self._MF)
        return self._MF

    @property
    def Air(self):
        Air = np.einsum('...pw,...ijnr->...ijnrpw', self.zT, self.MF)
        Air -= 2*np.einsum(
            '...ij,...nr,...p,...w->...ijnrpw',
            self.Ms, self.Ms, self.T, self.T, optimize=self.optimize
        )
        Air = np.einsum('jpnw,...ijnrpw->...ir', self.C, Air)
        return Air

    @property
    def _integrand_second_derivative(self):
        results = -2*np.einsum('...sm,...ir->...isrm', self.zT, self.F)
        results += 2*np.einsum(
            '...s,...m,...ir->...isrm', self.T, self.T, self.Ms, optimize=self.optimize
        )
        results += np.einsum(
            '...s,...m,...ir->...isrm', self.z, self.z, self.Air, optimize=self.optimize
        )
        return results

    @property
    def _integrand_first_derivative(self):
        results = np.einsum('...s,...ir->...isr', self.z, self.F)
        results -= np.einsum('...s,...ir->...isr', self.T, self.Ms)
        return results

    def get_greens_function(self, r, derivative=0, fourier=False):
        self.r = r
        if fourier:
            G = np.einsum('ijkl,...j,...l->...ik', self.C, r, r, optimize=self.optimize)
            return np.linalg.inv(G)
        self.initialize()
        if derivative == 0:
            M = np.einsum('n...ij->...ij', self.Ms)*self.dphi/(4*np.pi**2)
            return np.einsum('...ij,...->...ij', M, 1/np.linalg.norm(self.r, axis=-1))
        elif derivative == 1:
            M = np.einsum(
                'n...isr->...isr', self._integrand_first_derivative
            )/(4*np.pi**2)*self.dphi
            return np.einsum('...isr,...->...isr', M, 1/np.linalg.norm(self.r, axis=-1)**2)
        elif derivative == 2:
            M = np.einsum(
                'n...isrm->...isrm', self._integrand_second_derivative
            )/(4*np.pi**2)*self.dphi
            return np.einsum('...isrm,...->...isrm', M, 1/np.linalg.norm(self.r, axis=-1)**3)
