import numpy as np
import torch
import torch.nn as nn
from scipy.constants import pi, speed_of_light, elementary_charge, electron_mass, hbar

me_SI = electron_mass
hbar_SI = hbar
e_SI = elementary_charge
c_SI = speed_of_light

meV = e_SI * 1e-3
nm = 1e-9
ps = 1e-12

c = c_SI * ps / nm
hbar_meV_ps = hbar_SI / (meV * ps)
me = me_SI * c_SI**2 / meV / c**2

hbar = hbar_meV_ps
m = me
omega = 2 / hbar
vQD = 15

x_min = -75
x_max = 150
t_min = 0
t_max = 20

x0 = 0
x1 = 75
t0 = 0
t1 = 2

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend for Apple GPU acceleration!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
else:
    device = torch.device("cpu")
    print("Using CPU instead.")

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for units in layers[1:-1]:
            self.hidden_layers.append(nn.Linear(in_features=layers[0], out_features=units))
            layers[0] = units
        
        self.output_layer = nn.Linear(layers[-2], layers[-1])
    
    def forward(self, inputs):
        x, t = inputs
        X = torch.stack((x, t), dim=1)
        activation = nn.SiLU()
    
        for layer in self.hidden_layers:
            X = layer(X)
            X = activation(X)
    
        output = self.output_layer(X)
        psi_real = output[:, 0]
        psi_img = output[:, 1]
        
        return psi_real, psi_img

layers = [2, 512, 512, 512, 512, 512, 512, 2]

n_collocation = 3000
n_initial = 1000
n_boundary = 1000

# Define dataset generator
def generator():
    t_collocation = np.random.uniform(t_min, t_max, n_collocation)
    x_qd_collocation = np.where(t_collocation < t1, x0, np.where(t_collocation < t1 + (x1 - x0) / vQD, x0 + vQD * (t_collocation - t1), x1))
    x_collocation = np.random.normal(loc=x_qd_collocation, scale=25.0, size=n_collocation)

    x_initial = np.random.normal(loc=x0, scale=25.0, size=n_initial)
    t_initial = np.full(n_initial, t_min)

    x_boundary = np.concatenate([np.full(n_boundary // 2, x_min), np.full(n_boundary // 2, x_max)])
    t_boundary = np.random.uniform(t_min, t_max, n_boundary)

    x_collocation_torch = torch.from_numpy(x_collocation).float().to(device)
    t_collocation_torch = torch.from_numpy(t_collocation).float().to(device)

    x_initial_torch  = torch.from_numpy(x_initial).float().to(device)
    t_initial_torch  = torch.from_numpy(t_initial).float().to(device)

    x_boundary_torch = torch.from_numpy(x_boundary).float().to(device)
    t_boundary_torch = torch.from_numpy(t_boundary).float().to(device)

    return x_collocation_torch, t_collocation_torch, x_initial_torch, t_initial_torch, x_boundary_torch, t_boundary_torch

# Model setup
model = PINN(layers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.98
steps = 5000

def exp_decay(step):
    return decay_rate ** (step / steps)

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000)

# Loss function
def loss_function(x_collocation_torch, t_collocation_torch, x_initial_torch, t_initial_torch, x_boundary_torch, t_boundary_torch):
    #pde loss
    x_collocation_torch = x_collocation_torch.clone().requires_grad_(True)
    t_collocation_torch = t_collocation_torch.clone().requires_grad_(True)
    
    u, v = model((x_collocation_torch, t_collocation_torch))
    
    du_dt = torch.autograd.grad(u, t_collocation_torch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x_collocation_torch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_collocation_torch, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    
    dv_dt = torch.autograd.grad(v, t_collocation_torch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x_collocation_torch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx, x_collocation_torch, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0]
    
    xqd_arr = torch.where(t_collocation_torch < t1, x0, torch.where(t_collocation_torch < t1 + (x1 - x0) / vQD, x0 + vQD * (t_collocation_torch - t1), x1))
    
    real = -hbar * dv_dt + ((hbar ** 2) / (2 * m)) * d2u_dx2 - 0.5 * m * (omega ** 2) * ((x_collocation_torch - xqd_arr) ** 2) * u
    img = hbar * du_dt + ((hbar ** 2) / (2 * m)) * d2v_dx2 - 0.5 * m * (omega ** 2) * ((x_collocation_torch - xqd_arr) ** 2) * v
    
    physics_loss = torch.mean(real ** 2 + img ** 2)
    
    
    
    
    #initial condition loss
    u_i, v_i = model((x_initial_torch, t_initial_torch))
    
    psi_initial_actual = (((m * omega) / (np.pi * hbar)) ** 0.25) * torch.exp(((-m * omega) / (2 * hbar)) * (x_initial_torch ** 2))
    initial_condition_loss = torch.mean((u_i - psi_initial_actual) ** 2) + torch.mean((v_i - 0) ** 2)
    
    
    
    
    #boundary condition loss
    u_b, v_b = model((x_boundary_torch, t_boundary_torch))
    boundary_condition_loss = torch.mean(u_b ** 2) + torch.mean(v_b ** 2)
    
    return physics_loss, initial_condition_loss, boundary_condition_loss

# Training loop
epochs = 150000
history = []

for epoch in range(1, epochs+1):
    optimizer.zero_grad()

    physics_loss, initial_condition_loss, boundary_condition_loss = loss_function(*generator())
    total_loss = 10 * physics_loss + initial_condition_loss + boundary_condition_loss

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    history.append({"total_loss": total_loss.item(), "physics_loss": physics_loss.item(), "initial_condition_loss": initial_condition_loss.item(), "boundary_condition_loss": boundary_condition_loss.item()})

    if epoch % 15000 == 0:
        print(f"Epoch {epoch}/{epochs}")
        print(f"Total loss: {total_loss.item():.4e}")
        print(f"Physics loss: {physics_loss.item():.4e}")
        print(f"Initial condition loss: {initial_condition_loss.item():.4e}")
        print(f"Boundary condition loss: {boundary_condition_loss.item():.4e}")
        print("-" * 50)

#saving
import os
import json

os.makedirs("results", exist_ok=True)

torch.save(model.state_dict(), "results/trained_model.pth")

with open("results/training_history.json", "w") as f:
    json.dump(history, f)
