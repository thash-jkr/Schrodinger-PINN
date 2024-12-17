import numpy as np
from scipy.linalg import lu as lu_decomposition
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
import scipy.linalg
from physics.utils.grids import Grid1D
from physics.utils.constants import hbar_meV_ps as hbar
import matplotlib.pyplot as plt


class SchrodingerExponential:
    def __init__(self, time_grid: Grid1D, initial_vect, hamiltonian_over_hbar):
        """
        Note that this is only valid for time-independent Hamiltonians
        :param time_grid:
        :param initial_vect:
        :param hamiltonian_over_hbar: callable returning the matrix of the Hamiltonian divided by h_bar
        """
        self.time_grid = time_grid
        self.initial_vec = initial_vect
        self.hamiltonian_over_hbar = hamiltonian_over_hbar
        self.psi = np.zeros((time_grid.length, len(initial_vect)))

    def simulate(self):
        psi0 = self.initial_vec
        dt = self.time_grid.step

        for t in self.time_grid.span[1:]:
            H_over_hbar = self.hamiltonian_over_hbar(t)
            self.psi[t, :] = np.expm(-1j*H_over_hbar * dt) * self.psi[t-1, :]


class CrankNicolsonV2:
    def __init__(self, time_grid: Grid1D, initial_vec, evolution_function=None, evolution_matrices=None):
        """
        Solve a differential equation using the Crank-Nicolson method. The equation is assumed to be of the form:
        dC/dt = M(t) C(t)
        with M(t) a matrix called 'evolution matrix' and C a column vector.
        You can either provide an evolution function, return the matrix of M at a given time t, or a list of evolution
        matrices at each time t (not the time index).
        :param time_grid:
        :param initial_vec: value of the column vector at initial time
        :param evolution_function: function returning the evolution matrix at a given time
        :param evolution_matrices: list of evolution matrices at each time in the time_grid.
        """
        if (evolution_function is None) and (evolution_matrices is None):
            raise Exception("Either the evolution or the evolution matrices must be supplied")

        self.evolution_function, self.evolution_matrices = None, None

        if evolution_function is not None:
            self.evolution_function = evolution_function
        else:
            self.evolution_matrices = evolution_matrices

        self.time_grid = time_grid
        self.initial_vec = initial_vec
        self.solution = None
        self.measurements = None

    def run(self, measurements):
        """
        Run the solver and return the solution
        :param measurements: a list of operators to take the expectation value of at each time. They will be stored in
        the attribute "measurements"
        :return: list of column vectors at each time t specified by the time_grid
        """
        # Util variables
        N = len(self.initial_vec)
        Nt = self.time_grid.length
        dt = self.time_grid.step

        I = np.eye(N)
        C = np.zeros((Nt, N), dtype=np.complex64)
        C[0, :] = self.initial_vec

        # Initial values of the measurements
        n = len(measurements)
        if n > 0:
            self.measurements = np.zeros((n, Nt), dtype=np.complex64)
            for i, operator in enumerate(measurements):
                self.measurements[i, 0] = C[0, :].dot(operator * C[0, :])

        # Solve
        if self.evolution_matrices is not None:
            T = self.evolution_matrices

            for t in self.time_grid.range[1:]:
                A = I - dt/2 * T[t]
                B = I + dt/2 * T[t-1]

                C[t, :] = scipy.linalg.solve(A, B.dot(C[t-1, :]))

                for i, operator in enumerate(measurements):
                    self.measurements[i, t] = C[t, :].dot(operator * C[t, :])

        else:
            T = self.evolution_function

            for t in self.time_grid.range[1:]:
                previous_real_time, real_time = self.time_grid.span[t-1], self.time_grid.span[t]

                A = I - dt / 2 * T(real_time)
                B = I + dt / 2 * T(previous_real_time)

                C[t, :] = scipy.linalg.solve(A, B.dot(C[t - 1, :]))

                for i, operator in enumerate(measurements):
                    self.measurements[i, t] = C[t, :].dot(operator * C[t, :])

        self.solution = C
        self.measurements = np.real(self.measurements)
        return C


class CrankNicolson:
    def __init__(self, time_grid, initial_vec, evolution_function=None, evolution_matrices=None):
        if (evolution_function is None) and (evolution_matrices is None):
            raise Exception("Either the evolution or the evolution matrices must be supplied")

        if evolution_function is not None:
            self.evolution = evolution_function
        else:
            self.evolution = evolution_matrices

        self.time_step = time_grid.step
        self.time_range = time_grid.range
        self.sim_time_range = self.time_range[1:]
        self.n_time_points = len(time_grid.span)
        self.initial_vec = initial_vec
        self.solution = None

        if evolution_matrices is not None and len(evolution_matrices) < self.n_time_points:
            raise ValueError("The number of evolution matrices is less than the number of time steps")

        self._run()

    def _run(self):
        N = len(self.initial_vec)
        Nt = self.n_time_points
        dt = self.time_step
        T = self.evolution
        I = np.eye(N)
        Y = np.zeros((Nt, N), dtype=np.complex64)
        Y[0, :] = self.initial_vec

        for t in range(1, Nt):
            if t >= len(T):
                raise ValueError("The number of evolution matrices is less than the number of time steps")
            
            U1 = I - dt/2 * T[t]
            U2 = I + dt/2 * T[t-1]

            print(f"Checking at time step {t}")
            # print(f"Max of U1: {np.max(U1)}, Min of U1: {np.min(U1)}")
            # print(f"Max of U2: {np.max(U2)}, Min of U2: {np.min(U2)}")
            # print(f"Max of Y[t-1, :]: {np.max(Y[t-1, :])}, Min of Y[t-1, :]: {np.min(Y[t-1, :])}")
            # print(f"Norm of wavefunction at time step {t}: {np.linalg.norm(Y[t-1, :])}")

            if np.any(np.isnan(U1)) or np.any(np.isinf(U1)):
                raise ValueError(f"U1 contains NaN or Inf at time step {t}")
            if np.any(np.isnan(U2)) or np.any(np.isinf(U2)):
                raise ValueError(f"U2 contains NaN or Inf at time step {t}")
            if np.any(np.isnan(Y[t-1, :])) or np.any(np.isinf(Y[t-1, :])):
                raise ValueError(f"Y[t-1, :] contains NaN or Inf at time step {t}")


            Y[t, :] = scipy.linalg.solve(U1, U2.dot(Y[t-1, :]))

        self.solution = Y


class CrankNicolsonV3:
    def __init__(self, time_grid, initial_vec, evolution_function=None, evolution_matrices=None, offset=0):
        if (evolution_function is None) and (evolution_matrices is None):
            raise Exception("Either the evolution or the evolution matrices must be supplied")

        if evolution_function is not None:
            self.evolution = evolution_function
        else:
            self.evolution = evolution_matrices

        self.time_step = time_grid.step
        self.time_range = time_grid.range
        self.sim_time_range = self.time_range[1:]
        self.n_time_points = len(time_grid.span)
        self.initial_vec = initial_vec
        self.solution = None
        self.n_jobs = -1
        self.offset = offset

        if evolution_matrices is not None and len(evolution_matrices) < self.n_time_points:
            raise ValueError("The number of evolution matrices is less than the number of time steps")

        self.evolution = [csr_matrix(T_t) for T_t in self.evolution]
        self._run()

    def _run(self):
        N = len(self.initial_vec)
        Nt = self.n_time_points
        dt = self.time_step
        T = self.evolution
        I = eye(N, format='csr')  # Sparse identity matrix
        Y = np.zeros((Nt, N), dtype=np.complex128)
        Y[0, :] = self.initial_vec
        norms = []

        for t in range(1, Nt):
            U1 = I + 1j * dt/2 * T[t]
            U2 = I - 1j * dt/2 * T[t-1]

            norms.append(round(np.linalg.norm(Y[t-1, :]), 2))

            # Solve the linear system at each time step
            Y[t, :] = spsolve(U1, U2.dot(Y[t-1, :]))

        plt.figure(figsize=(13, 5))
        plt.plot(range(1, self.n_time_points), norms, label=self.offset)
        plt.xlabel("Time step")
        plt.ylabel("Norm of wavefunction")
        plt.title("Norm of wavefunction at each time step")
        plt.show()

        self.solution = Y


class RungeKutta:
    def __init__(self, time_grid, initial_vec, evolution_function=None, evolution_matrices=None, debug=False):
        if (evolution_function is None) and (evolution_matrices is None):
            raise Exception("Either the evolution function or the evolution matrices must be supplied")

        self.evolution = evolution_matrices if evolution_matrices is not None else evolution_function
        self.time_step = time_grid.step
        self.time_range = time_grid.range
        self.n_time_points = len(time_grid.span)
        self.initial_vec = initial_vec
        self.solution = None
        self.debug = debug

        if evolution_matrices is not None and len(evolution_matrices) < self.n_time_points:
            raise ValueError("The number of evolution matrices is less than the number of time steps")

        self.evolution = [csr_matrix(T_t) for T_t in self.evolution]
        self._run()

    def _run(self):
        N = len(self.initial_vec)
        Nt = self.n_time_points
        dt = self.time_step
        T = self.evolution
        Y = np.zeros((Nt, N), dtype=np.complex128)
        Y[0, :] = self.initial_vec
        norms = []

        def schrodinger_rhs(psi, H):
            return -1j / hbar * H.dot(psi)

        for t in range(1, Nt):
            H_n = T[t - 1]
            H_n_plus = T[t]
            H_n_half = (H_n + H_n_plus) / 2.0

            k1 = dt * schrodinger_rhs(Y[t - 1, :], H_n)
            k2 = dt * schrodinger_rhs(Y[t - 1, :] + 0.5 * k1, H_n_half)
            k3 = dt * schrodinger_rhs(Y[t - 1, :] + 0.5 * k2, H_n_half)
            k4 = dt * schrodinger_rhs(Y[t - 1, :] + k3, H_n_plus)

            Y[t, :] = Y[t - 1, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            norm = np.linalg.norm(Y[t, :])
            norms.append(norm)
            Y[t, :] /= norm

        plt.figure(figsize=(13, 5))
        plt.plot(range(1, self.n_time_points), norms)
        plt.xlabel("Time step")
        plt.ylabel("Norm of wavefunction")
        plt.title("Norm of wavefunction at each time step")
        plt.show()

        self.solution = Y

