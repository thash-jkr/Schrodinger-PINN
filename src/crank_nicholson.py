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
m = me_SI * c_SI ** 2 / meV / c ** 2
omega = 2 / hbar
vQD = 15

x_min, x_max = -75, 150
t_min, t_max = 0, 12.5
Nx, Nt = 5000, 5000

x_values = np.linspace(x_min, x_max, Nx)
t_values = np.linspace(t_min, t_max, Nt)

dx = x_values[1] - x_values[0]
dt = t_values[1] - t_values[0]

laplacian = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Nx, Nx), format='csc') / dx**2
H_kinetic = - (hbar ** 2 / (2 * m)) * laplacian

def ground_state(x):
    A = (m * omega / (np.pi * hbar)) ** 0.25
    alpha = (m * omega) / (2.0 * hbar)
    return A * np.exp(-alpha * (x ** 2))

def first_state(x):
    A = 1 / np.sqrt(2)
    B = ((m * omega) / (np.pi * hbar)) ** 0.25
    C = np.exp((-m * omega * (x ** 2)) / (2 * hbar))
    D = 2 * np.sqrt((m * omega) / hbar) * x
    
    return A * B * C * D

psi = ground_state(x_values)
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
psi /= norm

psi_real_analytical = np.zeros((Nx, Nt), dtype=np.float64)
psi_img_analytical = np.zeros((Nx, Nt), dtype=np.float64)
psi_real_analytical[:, 0] = np.real(psi)
psi_img_analytical[:, 0] = np.imag(psi)

I = sp.eye(Nx, format='csc')

def quantum_center(t_values):
    xqd_arr = np.zeros_like(t_values)
    x0, x1, t1 = 0, 75, 2
    for i, t in enumerate(t_values):
        if t < t1:
            xqd_arr[i] = x0
        elif t < t1 + (x1 - x0) / vQD:
            xqd_arr[i] = x0 + vQD * (t - t1)
        else:
            xqd_arr[i] = x1
    return xqd_arr

xqd_arr = quantum_center(t_values)

for t_i in range(1, Nt):
    if t_i % 100 == 0:
        print(f"Step {t_i} / {Nt}")

    V = 0.5 * m * omega ** 2 * (x_values - xqd_arr[t_i]) ** 2
    H = H_kinetic + sp.diags(V, format='csc')

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

np.savez("Schrodinger-PINN/src/results/analytical/crank_t12.npz",
         real=psi_real_analytical,
         img=psi_img_analytical,
         x_values=x_values,
         t_values=t_values)
