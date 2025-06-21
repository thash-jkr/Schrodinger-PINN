import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import speed_of_light, elementary_charge, electron_mass, hbar as hbar_SI

me_SI = electron_mass
e_SI = elementary_charge
c_SI = speed_of_light

meV = e_SI * 1e-3
nm = 1e-9
ps = 1e-12

c = c_SI * ps / nm
hbar = hbar_SI / (meV * ps)
m = me_SI * c_SI**2 / meV / c**2
omega = 2 / hbar
vQD = 15

x_min, x_max = -75, 150
t_min, t_max = 0, 20
Nx, Nt = 5000, 5000
t0, t1 = 2, 7
omega0, omega1 = 0.5 / hbar, 2 / hbar

x_values = np.linspace(x_min, x_max, Nx)
t_values = np.linspace(t_min, t_max, Nt)

dx = x_values[1] - x_values[0]
dt = t_values[1] - t_values[0]

power_factor = (2 - 0.5) / (t1 - t0)
H_kinetic = (power_factor ** 2) / (2 * m)

def ground_state(x):
    A = (m * omega0 / (np.pi * hbar)) ** 0.25
    alpha = (m * omega0) / (2.0 * hbar)
    return A * np.exp(-alpha * (x ** 2))

def first_state(x):
    A = 1 / np.sqrt(2)
    B = ((m * omega0) / (np.pi * hbar)) ** 0.25
    C = np.exp((-m * omega0 * (x ** 2)) / (2 * hbar))
    D = 2 * np.sqrt((m * omega0) / hbar) * x
    
    return A * B * C * D

psi = ground_state(x_values)
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
psi /= norm

psi_real_analytical = np.zeros((Nx, Nt), dtype=np.float64)
psi_img_analytical = np.zeros((Nx, Nt), dtype=np.float64)
psi_real_analytical[:, 0] = np.real(psi)
psi_img_analytical[:, 0] = np.imag(psi)

I = sp.eye(Nx, format='csc')

def omega_center(t_values):
    omega_arr = np.zeros_like(t_values)
    
    for i, t in enumerate(t_values):
        if t < t0:
            omega_arr[i] = omega0
        elif t < t1:
            omega_arr[i] = omega0 + ((omega1 - omega0) / (t1 - t0)) * (t - t0)
        else:
            omega_arr[i] = omega1
            
    return omega_arr

omega_arr = omega_center(t_values)

for t_i in range(1, Nt):
    if t_i % 100 == 0:
        print(f"Step {t_i} / {Nt}")
        print(f"Norm at previous step = {norm}")

    V = 0.5 * m * omega_arr[t_i] ** 2 * x_values ** 2
    H_diag = H_kinetic + V
    H = sp.diags(H_diag, format="csc")

    A = I + 1j * dt / (2 * hbar) * H
    B = I - 1j * dt / (2 * hbar) * H

    psi = spla.spsolve(A, B @ psi)

    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    
    if norm > 0:
        psi /= norm
    else:
        print(f"❌ Zero norm at step {t_i}")
        break

    psi_real_analytical[:, t_i] = np.real(psi)
    psi_img_analytical[:, t_i] = np.imag(psi)

np.savez("Schrodinger-PINN/src/results/analytical/crank_omega.npz",
         real=psi_real_analytical,
         img=psi_img_analytical,
         x_values=x_values,
         t_values=t_values)
