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
y_min, y_max = -75, 150
x0, y0 = 0, 0
x1, y1 = 75, 75

t1 = 2
t2 = t1 + (x1 - x0) / vQD

t_min, t_max = t1, t2
Nx, Ny, Nt = 2000, 2000, 2000

x_values = np.linspace(x_min, x_max, Nx)
y_values = np.linspace(y_min, y_max, Ny)
t_values = np.linspace(t_min, t_max, Nt)

dx = (x_max - x_min) / (Nx)
dy = (y_max - y_min) / (Ny)
dt = (t_max - t_min) / (Nt)

laplacian_x = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Nx, Nx), format='csc') / dx ** 2
laplacian_y = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Ny, Ny), format='csc') / dy ** 2

Ix = sp.eye(Nx)
Iy = sp.eye(Ny)

laplacian = sp.kron(Iy, laplacian_x) + sp.kron(laplacian_y, Ix)
H_kinetic = - (hbar ** 2 / (2 * m)) * laplacian

def ground_state_x(x):
    A = (m * omega / (np.pi * hbar)) ** 0.25
    alpha = (m * omega) / (2.0 * hbar)
    return A * np.exp(-alpha * (x ** 2))

def ground_state_y(y):
    A = (m * omega / (np.pi * hbar)) ** 0.25
    alpha = (m * omega) / (2.0 * hbar)
    return A * np.exp(-alpha * (y ** 2))

def ground_state(x, y):
    return np.outer(ground_state_x(x), ground_state_y(y))

psi = ground_state(x_values, y_values)
norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy)
psi /= norm

psi_real_analytical = np.zeros((Nx // 2, Ny // 2, Nt // 2))
psi_img_analytical = np.zeros((Nx // 2, Ny // 2, Nt // 2))

psi_real_analytical[:, :, 0] = np.real(psi[::2, ::2])
psi_img_analytical[:, :, 0] = np.imag(psi[::2, ::2])

I = sp.eye(Nx * Ny, format='csc')

def quantum_center_x(t_values):
    xqd_arr = np.zeros_like(t_values)
    
    for i, t in enumerate(t_values):
        if t < t1:
            xqd_arr[i] = x0
        elif t < t1 + (x1 - x0) / vQD:
            xqd_arr[i] = x0 + vQD * (t - t1)
        else:
            xqd_arr[i] = x1
            
    return xqd_arr

def quantum_center_y(t_values):
    yqd_arr = np.zeros_like(t_values)
    
    for i, t in enumerate(t_values):
        if t < t1:
            yqd_arr[i] = y0
        elif t < t1 + (y1 - y0) / vQD:
            yqd_arr[i] = y0 + vQD * (t - t1)
        else:
            yqd_arr[i] = y1
            
    return yqd_arr

xqd_arr = quantum_center_x(t_values)
yqd_arr = quantum_center_y(t_values)

for t_i in range(1, Nt):
    print(f"Step {t_i} / {Nt}")

    Vx = 0.5 * m * omega ** 2 * (x_values.reshape(Nx, 1) - xqd_arr[t_i]) ** 2
    Vy = 0.5 * m * omega ** 2 * (y_values.reshape(1, Ny) - yqd_arr[t_i]) ** 2
    V = Vx + Vy
    
    V_flat = V.flatten()
    H = H_kinetic + sp.diags(V_flat, 0, format='csc')

    A = I + 1j * dt / (2 * hbar) * H
    B = I - 1j * dt / (2 * hbar) * H

    psi_flat = spla.spsolve(A, B @ psi.flatten())
    psi = psi_flat.reshape((Nx, Ny))

    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    
    if norm > 0:
        psi /= norm
        print(f"Norm at step {t_i} = {norm}]")
    else:
        print(f"Zero norm at step {t_i}")
        break

    if t_i % 2 == 0:
        psi_real_analytical[:, :, t_i // 2] = np.real(psi[::2, ::2])
        psi_img_analytical[:, :, t_i // 2] = np.imag(psi[::2, ::2])

np.savez("Schrodinger-PINN/src/results/analytical/crank_2d_2.npz",
         real=psi_real_analytical,
         img=psi_img_analytical,
         x_values=x_values[::2],
         y_values=y_values[::2],
         t_values=t_values[::2])
