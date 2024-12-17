import numpy as np
from physics.utils.constants import hbar_meV_ps as hbar


class HarmonicOscillator:
    def __init__(self, level_spacing):
        self.level_spacing = level_spacing
        self.omega = level_spacing / hbar
        self.basis = None
        self.spectrum = None

    def get_spectrum(self, n_states):
        h = hbar
        w = self.omega
        return np.array([h*w * (n + 1/2) for n in range(n_states)], dtype=np.float64)

    def get_eigenstate(self, level, mass, center, position_vector):
        n = level
        m = mass
        x0 = center
        x = position_vector
        h = hbar
        w = self.omega

        beta = np.sqrt(m * w / h)
        hval = np.polynomial.hermite.hermval(beta * (x - x0), [0]*n + [1])

        phi_n = (beta ** 2 / np.pi) ** (1 / 4) * 1 / np.sqrt(2**float(n) * np.math.factorial(n)) * np.exp(-beta ** 2 * (x - x0) ** 2 / 2) * hval

        return phi_n

    def compute_basis(self, N_states, mass, center, position_vector):
        Nx = len(position_vector)
        self.basis = np.zeros((Nx, N_states), dtype=np.complex64)

        for energy_level in range(N_states):
            self.basis[:, energy_level] = self.get_eigenstate(energy_level, mass, center, position_vector)

        self.spectrum = self.get_spectrum(N_states)

        return self.spectrum, self.basis
