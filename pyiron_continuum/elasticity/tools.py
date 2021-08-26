import numpy as np

class tools:
    def normalize(self, x):
        return (x.T/np.linalg.norm(x, axis=-1).T).T

    def orthonormalize(self, vectors):
        x = self.normalize(vectors)
        x[1] = x[1]-np.einsum('i,i,j->j', x[0], x[1], x[0])
        x[2] = np.cross(x[0], x[1])
        if np.isclose(np.linalg.det(x), 0):
            raise ValueError('Vectors not independent')
        return self.normalize(x)

    def get_plane(self, T):
        x = self.normalize(np.random.random(T.shape))
        x = self.normalize(x-np.einsum('...i,...i,...j->...j', x, T, T))
        y = np.cross(T, x)
        return x,y

    def index_from_voigt(self, i, j):
        if i==j:
            return i
        else:
            return 6-i-j

    def C_from_voigt(self, C_in):
        C = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        C[i,j,k,l] = C_in[self.index_from_voigt(i,j), self.index_from_voigt(k,l)]
        return C

    def C_to_voigt(self, C_in):
        C = np.zeros((6, 6))
        for i in range(3):
            for j in range(i+1):
                for k in range(3):
                    for l in range(k+1):
                        C[self.index_from_voigt(i,j), self.index_from_voigt(k,l)] = C_in[i,j,k,l]
        return C

