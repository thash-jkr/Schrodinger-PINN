import numpy as np
from physics.utils.constants import hbar_meV_ps as hbar
from physics.utils.grids import SpaceGrid2D


def coefficients_to_wf(coefficients, basis, spectrum, space_grid, time_grid):
    wave_function = np.zeros((space_grid.length, time_grid.length), dtype=np.complex64)

    for t in time_grid.range:
        wave_function[:, t] = basis.dot(coefficients[t, :] * np.exp(-1j/hbar * spectrum * time_grid.span[t]))

    return wave_function


def scalar_product_1d(wf1, wf2, dx=1):
    return np.trapz(np.conj(wf1) * wf2, dx=dx)


class Wavefunction2D:
    def __init__(self, grid: SpaceGrid2D):
        self.grid = grid
        self.data = None

    def scalar_product(self, wf1, wf2):
        """
        Return the scalar product
        :param wf1:
        :param wf2:
        :return:
        """
        return np.trapz(np.trapz(np.conj(wf1) * wf2, dx=self.grid.dx), dx=self.grid.dz)


class Wavefunction2DColumn(Wavefunction2D):
    def build_from_basis(self, coefficients, basis, spectrum, t):
        self.data = np.zeros((self.grid.Nx * self.grid.Nz, 1), dtype=np.complex64)

        self.data[:] = basis.dot(coefficients[t, :] * np.exp(-1j / hbar * spectrum * t))

        return self.data

    def scalar_product(self, wf1, wf2):
        return np.trapz(np.conj(wf1) * wf2, dx=self.grid.dx * self.grid.dz)

    def to_2d_array(self):
        return self.data.reshape((self.grid.Nx, self.grid.Nz))
