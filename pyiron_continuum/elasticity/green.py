import numpy as np

class Isotropic:
    def __init__(self, poissons_ratio, shear_modulus, min_distance=0):
        self.poissons_ratio = poissons_ratio
        self.shear_modulus = shear_modulus
        self.min_dist = min_distance
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
        G = self.A*np.eye(3)+self.B*np.einsum('...i,...j,...->...ij', r, r, R_inv**2)
        return np.einsum('...ij,...->...ij', G, R_inv)

    def G_fourier(self, k):
        """Fourier transform of the Green's function"""
        k = np.array(k).reshape(-1, 3)
        K = np.linalg.norm(k, axis=-1)
        if self.min_dist==0:
            return 4*np.pi*(
                self.A*np.einsum('n,ij->nij', 1/K**2, np.eye(3))
                + self.B*np.einsum('n,ij->nij', 1/K**2, np.eye(3))
                - 2*self.B*np.einsum('n,ni,nj->nij',  1/K**3, k, k)
            )
        return 4*np.pi*(
            self.A*np.einsum('n,ij->nij', np.cos(K*self.min_dist)/K**2, np.eye(3))
            + self.B*np.einsum('n,ij->nij', np.sin(K*self.min_dist)/(K**3*self.min_dist), np.eye(3))
            + self.B*np.einsum(
                'n,ni,nj->nij',
                np.cos(K*self.min_dist)/K**4-3*np.sin(K*self.min_dist)/(K**5*self.min_dist),
                k, k)
        )

    def dG(self, x):
        """ first derivative of the Green's function """
        E = np.eye(3)
        r = np.array(x).reshape(-1, 3)
        R = np.linalg.norm(r, axis=-1)
        distance_condition = R<self.min_dist
        R[distance_condition] = 1
        r = np.einsum('ni,n->ni', r, 1/R)
        v = -self.A*np.einsum('ik,nj->nijk', E, r)
        v += self.B*np.einsum('ij,nk->nijk', E, r)
        v += self.B*np.einsum('jk,ni->nijk', E, r)
        v -= 3*self.B*np.einsum('ni,nj,nk->nijk', r, r, r)
        v = np.einsum('nijk,n->nijk', v, 1/R**2)
        v[distance_condition] *= 0
        return v

    def ddG(self, x):
        """ Second derivative of the Green's function """
        E = np.eye(3)
        r = np.array(x).reshape(-1, 3)
        R = np.linalg.norm(r, axis=-1)
        distance_condition = R<self.min_dist
        R[distance_condition] = 1
        r = np.einsum('ni,n->ni', r, 1/R)
        v = -self.A*np.einsum('ik,jl->ijkl', E, E)
        v = v+3*self.A*np.einsum('ik,nj,nl->nijkl', E, r, r)
        v = v+self.B*np.einsum('il,jk->ijkl', E, E)
        v -= 3*self.B*np.einsum('il,nj,nk->nijkl', E, r, r)
        v = v+self.B*np.einsum('ij,kl->ijkl', E, E)
        v -= 3*self.B*np.einsum('ni,nj,kl->nijkl', r, r, E)
        v -= 3*self.B*np.einsum('ij,nk,nl->nijkl', E, r, r)
        v -= 3*self.B*np.einsum('jk,ni,nl->nijkl', E, r, r)
        v -= 3*self.B*np.einsum('jl,ni,nk->nijkl', E, r, r)
        v += 15*self.B*np.einsum('ni,nj,nk,nl->nijkl', r, r, r, r)
        v = np.einsum('nijkl,n->nijkl', v, 1/R**3)
        v[distance_condition] *= 0
        return v

def normalize(x):
    return (x.T/np.linalg.norm(x, axis=-1).T).T

def get_plane(T):
    x = normalize(np.random.random(T.shape))
    x = normalize(x-np.einsum('...i,...i,...j->...j', x, T, T))
    y = np.cross(T, x)
    return x,y

class Anisotropic:
    def __init__(self, elastic_constants, n_mesh=100, optimize=True):
        self.C = elstic_constants
        self.phi_range, self.dphi = np.linspace(0, np.pi, n_mesh, endpoint=False, retstep=True)
        self.optimize = optimize
        self.initialize()

    def initialize(self):
        self._zT = None
        self._F = None
        self._T = None
        self._x = None
        self._y = None
        self._z = None
        self._Ms = None

    @property
    def x(self):
        if self._x is None:
            self._x, self._y = get_plane(self.T)
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._x, self._y = get_plane(self.T)
        return self._y

    @property
    def z(self):
        if self._z is None:
            self._z = np.einsum(
                '...x,n->n...x', x, np.cos(phi_range)
            )+np.einsum('...x,n->n...x', y, np.sin(phi_range))
        return self._z

    @property
    def Ms(self):
        if self._Ms is None:
            self._Ms = np.einsum('ijkl,...j,...l->...ik', self.C, self.T, self.T)
            self._Ms = np.linalg.inv(M)
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
        MF = np.einsum('...ij,...nr->...ijnr', self.F, self.Ms)
        MF = MF+np.einsum('...ijnr->...nrij', MF)
        return self._MF

    @property
    def Air(self):
        Air = np.einsum('...pw,...ijnr->...ijnrpw', self.zT, self.MF)
        Air -= 2*np.einsum('...ij,...nr,...p,...w->...ijnrpw', self.Ms, self.Ms, self.T, self.T)
        Air = np.einsum('jpnw,...ijnrpw->...ir', self.C, Air)

    @property
    def _integrand_second_derivative(self):
        results = 2*np.einsum('...s,...m,...ir->...irsm', self.T, self.T, self.Ms)
        results -= 2*np.einsum('...sm,...ir->...irsm', self.zT, self.F)
        results += np.einsum('...s,...m,...ir->...irsm', self.z, self.z, self.Air)
        return results

    @property
    def _integrand_first_derivative(self):
        results = -np.einsum('s,ir->irs', self.T, self.Ms)
        results += np.einsum('s,ir->irs', self.z, self.F)
        return results

    def get_greens_function(r, derivative=0):
        self.initialize()
        self.r = r
        if derivative == 0:
            M = np.einsum('...nij->...ij', self.Ms)*self.phi/(4*np.pi**2)
            return np.einsum('...ij,...->...ij', M, 1/np.linalg.norm(self.r, axis=-1))
        elif derivative == 1:
            M = np.einsum('n...irs->...irs', self._integrand_first_derivative)/(4*np.pi**2)*self.dphi
            return np.einsum('...ijs,...->...ijs', M, 1/np.linalg.norm(self.r, axis=-1)**2)
        elif derivative == 2:
            M = np.einsum(
                'n...irsm->...irsm', self._integrand_second_derivative
            )/(4*np.pi**2)*self.dphi
            return np.einsum('...ijsm,...->...ijsm', M, 1/np.linalg.norm(self.r, axis=-1)**3)

def displacement_field(r, dipole_tensor, poissons_ratio, shear_modulus, min_distance=0):
    r = np.array(r)
    g_tmp = Green(
        poissons_ratio=poissons_ratio, shear_modulus=shear_modulus, min_distance=min_distance
    ).dG(r)
    if dipole_tensor.shape==(3,):
        return -np.einsum(
            'nijk,k,kj->ni', g_tmp, dipole_tensor, np.eye(3)
        ).reshape(r.shape)
    elif dipole_tensor.shape==(3,3,):
        return -np.einsum(
            'nijk,kj->ni', g_tmp, dipole_tensor
        ).reshape(r.shape)
    elif dipole_tensor.shape==(r.shape+(3,)):
        return -np.einsum(
            'nijk,nkj->ni', g_tmp, dipole_tensor.reshape(-1, 3, 3)
        ).reshape(r.shape)
    else:
        raise ValueError('dipole tensor must be a 3d vector 3x3 matrix or Nx3x3 matrix')

def strain_field(r, dipole_tensor, poissons_ratio, shear_modulus, min_distance=0):
    r = np.array(r)
    g_tmp = Green(
        poissons_ratio=poissons_ratio, shear_modulus=shear_modulus, min_distance=min_distance
    ).ddG(r)
    if dipole_tensor.shape==(3,) or dipole_tensor.shape==(3,3,):
        v = -np.einsum(
            'nijkl,kl->nij', g_tmp, dipole_tensor*np.eye(3)
        )
    elif dipole_tensor.shape==(r.shape+(3,)):
        v = -np.einsum(
            'nijkl,nkl->nij', g_tmp, dipole_tensor.reshape(-1,3,3)
        )
    else:
        raise ValueError('dipole tensor must be a 3d vector 3x3 matrix or Nx3x3 matrix')
    v = 0.5*(v+np.einsum('nij->nji', v))
    return v.reshape(r.shape+(3,))

def fourier_strain_field(
    k, dipole_tensor, poissons_ratio, shear_modulus, min_distance=0, safe_shift=1.0e-8
):
    k = np.array(k)
    if safe_shift > 0:
        k[np.linalg.norm(k, axis=-1)<safe_shift, 0] = safe_shift
    g_tmp = Green(
        poissons_ratio=poissons_ratio, shear_modulus=shear_modulus, min_distance=min_distance
    ).G_fourier(k)
    if dipole_tensor.shape==(3,) or dipole_tensor.shape==(3,3,):
        v = np.einsum(
            'nik,kl,nj,nl->nij', g_tmp, dipole_tensor*np.eye(3), k, k
        )
    elif dipole_tensor.shape==(k.shape+(3,)):
        v = np.einsum(
            'nik,nkl,nj,nl->nij', g_tmp, dipole_tensor.reshape(-1,3,3), k, k
        )
    else:
        raise ValueError('dipole tensor must be a 3d vector 3x3 matrix or Nx3x3 matrix')
    v = 0.5*(v+np.einsum('nij->nji', v))
    return v.reshape(k.shape+(3,))

