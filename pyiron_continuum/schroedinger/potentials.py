
from pyiron_base import HasStorage
from abc import ABC, abstractmethod
from pyiron_continuum.schroedinger.mesh import RectMesh
from typing import Type
import numpy as np


class Potential(HasStorage, ABC):
    @abstractmethod
    def __call__(self, mesh: Type[RectMesh]) -> np.ndarray:
        pass


class InfiniteWell(Potential):
    def __init__(self, fraction_zero=0.5):
        super().__init__()
        self.storage.fraction_zero = fraction_zero

    @property
    def fraction_zero(self):
        return self.storage.fraction_zero

    @fraction_zero.setter
    def fraction_zero(self, fraction):
        self.storage.fraction_zero = fraction

    def __call__(self, mesh: Type[RectMesh]) -> np.ndarray:
        potential = np.ones_like(mesh.mesh) * np.nan
        lengths = np.array([np.amax(m) for m in mesh.mesh]) + mesh.steps
        mask = np.array([
            (m > 0.5 * l * (1 - self.fraction_zero)) * (m < 0.5 * l * (1 + self.fraction_zero))
            for m, l in zip(mesh.mesh, lengths)
        ])
        potential[mask] = 0
        return potential


class Sinusoidal(Potential):
    def __init__(self, n_waves=1, amplitude=1):
        super().__init__()
        self.storage.n_waves = self._clean_waves(n_waves)
        self.storage.amplitude = amplitude

    def _clean_waves(self, n_waves):
        if not np.issubdtype(type(n_waves), np.integer):
            raise TypeError(
                f'Waves must come in integers to obey periodic boundary conditions, but got {type(n_waves)}'
            )
        return n_waves

    @property
    def n_waves(self):
        return self.storage.n_waves

    @n_waves.setter
    def n_waves(self, n):
        self.storage.n_waves = self._clean_waves(n)

    @property
    def amplitude(self):
        return self.storage.amplitude

    @amplitude.setter
    def amplitude(self, a):
        self.storage.amplitude = a

    def __call__(self, mesh: Type[RectMesh]) -> np.ndarray:
        return np.prod([self.amplitude * np.sin(self.n_waves * m) for m in mesh.mesh], axis=0)
