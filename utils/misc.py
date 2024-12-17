import pickle
import numpy as np
import pandas as pd
from numpy import exp
from math import ceil, log10
# from numba import jit
import warnings

from physics.utils.constants import hbar, pi
from scipy.constants import Boltzmann, electron_volt
from physics.utils.grids import Grid1D

boltzmann_mev_per_kelvin = Boltzmann / (electron_volt*1e-3)


to_dB = lambda s: 20 * np.log10(np.abs(s))


def format_string_varying_number(n_range):
    """
    Ignores 0
    :param n_range:
    :return:
    """
    factor = []
    denominator = []
    for i, n in enumerate(n_range):
        if n == 1:
            factor.append("+")
        elif n == -1:
            factor.append("-")
        elif n != 0:
            if n > 0:
                factor.append("+%d" % n)
            else:
                factor.append("%s" % n)
    return factor


def find_resonance(f, S, antires=False, reflection=False):
    """
    From the given numerical data, return the amplitude of the peak, the resonance frequency, the 3dB bandwidth,
    and the quality factor.
    Make sure the given range contains a peak!
    By default, S is assumed to be a transmission, with a Lorentzian shape
    :param f: frequency in Hz
    :param S: S parameter in dB
    :param antires: boolean indicating whether we should look for an anti-resonance instead, i.e. a minimum
    :param reflection: boolean indicating whether the S parameter is a reflection,
    which leads to a different way of computing the resonance
    :return:
    """
    if len(f) != len(S):
        raise ValueError("Frequency and S parameter of different lengths!")

    # This code changes the value of S, so we need to be able to change a value by index
    if type(S) == pd.core.series.Series:
        S = S.to_numpy()

    if reflection:
        # Reflection
        A_max = np.max(S)
        f0_idx = np.argmin(S)
        A_bandwidth = A_max - 3
    elif antires:
        # Anti-resonance in transmission
        A_min = np.min(S)
        f0_idx = np.argmin(S)
        A_bandwidth = A_min + 3
        A_max = A_min   # So that the rest of the code is unchanged
    else:
        # Transmission
        A_max = np.max(S)
        f0_idx = np.argmax(S)
        A_bandwidth = A_max - 3

    f0 = f[f0_idx]
    diff = np.abs(S - A_bandwidth)
    bandwidth_bound_1 = np.argmin(diff)
    diff[bandwidth_bound_1] = np.Inf
    bandwidth_bound_2 = np.argmin(diff)
    bandwidth = np.abs(f[bandwidth_bound_1] - f[bandwidth_bound_2])
    Q = f0 / bandwidth

    return Resonance(f0, Q, A_max, bandwidth)


class Resonance:
    def __init__(self, frequency, quality_factor, amplitude, bandwidth):
        self.frequency = frequency
        self.quality_factor = quality_factor
        self.amplitude = amplitude
        self.bandwidth = bandwidth
        self.linewidth = bandwidth

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "f=%0.3f GHz, Q=%0.2f, linewidth=%0.3f MHz" % (self.frequency, self.quality_factor, self.bandwidth*1e3)


def round_to(number, step):
    n = ceil(log10(step))   # Leading digit position
    return round(round(number/step)*step, 8)


# @jit(nopython=True)
def bose_einstein_distribution(f, T):
    """
    Occupation probability at a given energy/temperature
    :param f: expressed in GHz
    :param T: expressed in Kelvin
    :return: the occupation probability
    """
    return 1/(exp((hbar * 2*pi*f * 1e-3) / (boltzmann_mev_per_kelvin * T)) - 1)


def index_of_nearest(array, value):
    return (np.abs(array - value)).argmin()


def my_arange(start, end, step):
    return np.array([start+k*step for k in range(round((end-start)/step)+1) if start+k*step <= end])


def absolute_error(a, b, epsilon=1e-8, debug=True):
    """
    Compute the absolute error between two values.
    :param a:
    :param b:
    :param epsilon_:
    :param debug: if True, display information to locate where there is disagreement
    :return:
    """
    # Compute the diff
    diff_abs = np.abs(a-b)

    # Find where there is a disagreement
    delta_abs = diff_abs < epsilon

    # Is there disagreement?
    agreement_abs = np.all(delta_abs)

    if debug:
        if not agreement_abs:
            print("Diff abs\n", diff_abs)
            print("Delta abs\n", delta_abs)
            print("a\n", a)
            print("b\n", b)

    return agreement_abs


def relative_error(a, b, epsilon=1e-3, debug=True):
    """
    Compute the relative error between two values. Don't trust this function if the arguments contain 0s!
    :param a:
    :param b:
    :param epsilon:
    :param debug: if True, display information to locate where there is disagreement
    :return:
    """
    # Check for zeros
    if np.any(a == 0) or np.any(b == 0):
        warnings.warn("There is at least one zero in the arguments, relative errors may be meaningless.")

    # Compute the diff
    diff_abs = np.abs(a-b)
    diff_rel_a = diff_abs / np.abs(a)
    diff_rel_b = diff_abs / np.abs(b)

    # Find where there is a disagreement
    delta_rel_a = diff_rel_a < epsilon
    delta_rel_b = diff_rel_b < epsilon

    # Is there disagreement?
    agreement_rel_a = np.all(delta_rel_a)
    agreement_rel_b = np.all(delta_rel_b)

    if debug:
        if not agreement_rel_a:
            print("Diff rel a\n", diff_rel_a)
            print("Delta rel a\n", delta_rel_a)
            print("a\n", a)
            print("b\n", b)

        if not agreement_rel_b:
            print("Diff rel b\n", diff_rel_b)
            print("Delta rel b\n", delta_rel_b)
            print("a\n", a)
            print("b\n", b)

    return agreement_rel_a, agreement_rel_b


def simulation_banner(min_x, max_x, dx, simulation_time, dt):
    print("[min_x, max_x] = [%0.2f, %0.2f] nm" % (min_x*1e9, max_x*1e9))
    print("x range: %0.2f nm" % ((max_x-min_x)*1e9))
    print("x resolution: %e m" % (dx))
    print("Simulation time: %0.3f ps" % (simulation_time*1e12))
    print("Time resolution: %0.3f ps" % (dt*1e12))


class LinearTrajectory:
    def __init__(self, x0, x1, t0, t1, time_grid: Grid1D = None):
        self.x0, self.x1 = x0, x1
        self.t0, self.t1 = t0, t1

        self.speed = (x1-x0) / (t1-t0)

        if time_grid is not None:
            t = time_grid.span
            self.trace = np.array([x0]*np.sum(t < t0))
            t_t0 = t[t >= t0]
            self.trace = np.concatenate([self.trace, x0 + self.speed * (t_t0[t1 > t_t0]-t0)])
            self.trace = np.concatenate([self.trace, np.array([x1]*np.sum(t >= t1))])
        else:
            self.trace = None

    def evolution(self, t):
        if t < self.t0:
            return self.x0
        elif self.t0 <= t < self.t1:
            return self.x0 + self.speed * (t - self.t0)
        elif self.t1 <= t:
            return self.x1


class NTTRealisticTrajectory:
    def __init__(self, space_grid: Grid1D, time_grid: Grid1D, f=4e9,
                 alpha_ent_barr=0.49, alpha_ent_exit_barr=0.037, alpha_exit_barr=0.48, alpha_exit_ent_bar=0.052,
                 V_ent=-0.7, V_exit=-0.7, V_ac=1.415, x_ent=0, x_exit=100, U_scr=1, L_ent=100, L_exit=100, L_scr=1):
        x = space_grid.span
        t = time_grid.span
        U_exit = -alpha_exit_barr * V_exit * (alpha_exit_barr / alpha_exit_ent_bar) ** \
                      (-abs(x - x_exit) / abs(x_ent - x_exit))
        U_upper = U_scr * exp(-(x - x_ent) / L_scr * (x > x_ent)) \
                       * exp(-(x_ent - L_ent - x) / L_scr * (x_ent - L_ent - x > 0)) \
                       + U_scr * exp(-(x - x_exit - L_exit) / L_scr * (x - x_exit - L_exit > 0)) \
                       * exp(-(x_exit - x) / L_scr * (x_exit - x > 0))

        U_ent_x = -alpha_ent_barr * (alpha_ent_barr/alpha_ent_exit_barr) ** (-abs(x - x_ent)/abs(x_exit - x_ent))
        U_ent = U_ent_x * (V_ent + V_ac * np.cos(2 * np.pi * f * t[np.newaxis].transpose() * 1e-12))
        U = U_ent + U_exit + U_upper

        self.trace = x[np.argmin(U, axis=1)]
        self.time_grid = time_grid
        self.f = f / 1e9    # Pumping frequency in GHz

    def evolution(self, t):
        return self.trace[self.time_grid.index_of(t)]


def save_simulation(simulation):
    with open(path, 'wb') as output:
        pickle.dump(simulation, output, pickle.HIGHEST_PROTOCOL)


def load_simulation(path):
    with open(path, 'wb') as input:
        simulation = pickle.load(input)
    return simulation