import numpy as np
import torch
import json
import torch.nn as nn
from scipy.constants import speed_of_light, elementary_charge, electron_mass, hbar

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

x_min, x_max = -75, 150
y_min, y_max = -75, 150
x0, y0 = 0, 0
x1, y1 = 75, 75
dx = x1 - x0
dy = y1 - y0

t1 = 2
dist = np.sqrt(dx ** 2 + dy ** 2)
t2 = t1 + dist / vQD

t_min = t1
t_max = t2

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
    def __init__(self, layers, t_min, t_max):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        for units in layers[1:-1]:
            self.hidden_layers.append(nn.Linear(in_features=layers[0], out_features=units))
            layers[0] = units
        
        self.output_layer = nn.Linear(layers[-2], layers[-1])

        self.n_collocation = 5000
        self.n_initial = 1000
        self.n_boundary = 1000

        self.t_min = t_min
        self.t_max = t_max
    
    def forward(self, inputs):
        x, y, t = inputs
        X = torch.stack((x, y, t), dim=1)
        activation_1 = nn.Tanh()
        activation_2 = nn.SiLU()
        i = 1
    
        for layer in self.hidden_layers:
            X = layer(X)
            X = activation_1(X) if i < 2 else activation_2(X)
            i += 1
    
        output = self.output_layer(X)
        psi_real = output[:, 0]
        psi_img = output[:, 1]
        
        return psi_real, psi_img

    def generator(self, T_min, T_max):
        #Collocation Points
        t_collocation = np.random.uniform(T_min, T_max, self.n_collocation)
        
        prog = np.clip((t_collocation - t1) * vQD / dist, 0, 1)
        x_qd_collocation = x0 + prog * dx
        y_qd_collocation = y0 + prog * dy
        
        x_collocation = np.random.normal(loc=x_qd_collocation, scale=25.0, size=self.n_collocation)
        y_collocation = np.random.normal(loc=y_qd_collocation, scale=25.0, size=self.n_collocation)
    
        #Initial condition points
        x_c = 0
        y_c = 0
        x_initial = np.random.normal(loc=x_c, scale=25.0, size=self.n_initial)
        y_initial = np.random.normal(loc=y_c, scale=25.0, size=self.n_initial)
        t_initial = np.full(self.n_initial, T_min)
        
        #Boundary condition points
        n_set = self.n_boundary // 4
        
        x_boundary_x = np.concatenate([np.full(n_set, x_min), np.full(n_set, x_max)])
        y_boundary_x = np.random.uniform(y_min, y_max, n_set * 2)
        
        x_boundary_y = np.random.uniform(x_min, x_max, n_set * 2)
        y_boundary_y = np.concatenate([np.full(n_set, y_min), np.full(n_set, y_max)])
        
        x_boundary = np.concatenate([x_boundary_x, x_boundary_y])
        y_boundary = np.concatenate([y_boundary_x, y_boundary_y])
        t_boundary = np.random.uniform(T_min, T_max, self.n_boundary)
        
        #Converting to torch
        x_collocation_torch = torch.from_numpy(x_collocation).float().to(device)
        y_collocation_torch = torch.from_numpy(y_collocation).float().to(device)
        t_collocation_torch = torch.from_numpy(t_collocation).float().to(device)
        
        x_initial_torch = torch.from_numpy(x_initial).float().to(device)
        y_initial_torch = torch.from_numpy(y_initial).float().to(device)
        t_initial_torch = torch.from_numpy(t_initial).float().to(device)
        
        x_boundary_torch = torch.from_numpy(x_boundary).float().to(device)
        y_boundary_torch = torch.from_numpy(y_boundary).float().to(device)
        t_boundary_torch = torch.from_numpy(t_boundary).float().to(device)
    
        return x_collocation_torch, y_collocation_torch, t_collocation_torch, x_initial_torch, y_initial_torch, t_initial_torch, x_boundary_torch, y_boundary_torch, t_boundary_torch

    def loss_function(self, initial_condition, x_collocation_torch, y_collocation_torch, t_collocation_torch, x_initial_torch, y_initial_torch, t_initial_torch, x_boundary_torch, y_boundary_torch, t_boundary_torch):
        #pde loss
        x_collocation_torch = x_collocation_torch.clone().requires_grad_(True)
        y_collocation_torch = y_collocation_torch.clone().requires_grad_(True)
        t_collocation_torch = t_collocation_torch.clone().requires_grad_(True)
        
        u, v = self((x_collocation_torch, y_collocation_torch, t_collocation_torch))
        
        du_dt = torch.autograd.grad(u, t_collocation_torch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x_collocation_torch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dy = torch.autograd.grad(u, y_collocation_torch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_collocation_torch, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        d2u_dy2 = torch.autograd.grad(du_dy, y_collocation_torch, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]
        
        dv_dt = torch.autograd.grad(v, t_collocation_torch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dx = torch.autograd.grad(v, x_collocation_torch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dy = torch.autograd.grad(v, y_collocation_torch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        d2v_dx2 = torch.autograd.grad(dv_dx, x_collocation_torch, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0]
        d2v_dy2 = torch.autograd.grad(dv_dy, y_collocation_torch, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0]
        
        dist = torch.sqrt(torch.tensor(dx ** 2 + dy ** 2, device=device))
        prog = torch.clamp((t_collocation_torch - t1) * vQD / dist, min=0.0, max=1.0)
        xqd_arr = x0 + prog * dx
        yqd_arr = y0 + prog * dy
        
        real = -hbar * dv_dt + ((hbar ** 2) / (2 * m)) * d2u_dx2 + ((hbar ** 2) / (2 * m)) * d2u_dy2 - 0.5 * m * (omega ** 2) * ((x_collocation_torch - xqd_arr) ** 2) * u - 0.5 * m * (omega ** 2) * ((y_collocation_torch - yqd_arr) ** 2) * u
        img = hbar * du_dt + ((hbar ** 2) / (2 * m)) * d2v_dx2 + ((hbar ** 2) / (2 * m)) * d2v_dy2 - 0.5 * m * (omega ** 2) * ((x_collocation_torch - xqd_arr) ** 2) * v - 0.5 * m * (omega ** 2) * ((y_collocation_torch - yqd_arr) ** 2) * v
        
        physics_loss = torch.mean(real ** 2 + img ** 2)
        
        
        
        
        #initial condition loss
        u_i, v_i = self((x_initial_torch, y_initial_torch, t_initial_torch))
        
        psi_initial_real, psi_initial_img = initial_condition(x_initial_torch, y_initial_torch)
        initial_condition_loss = torch.mean((u_i - psi_initial_real) ** 2) + torch.mean((v_i - psi_initial_img) ** 2)
        
        
        
        
        #boundary condition loss
        u_b, v_b = self((x_boundary_torch, y_boundary_torch, t_boundary_torch))
        boundary_condition_loss = torch.mean(u_b ** 2) + torch.mean(v_b ** 2)
        
        return physics_loss, initial_condition_loss, boundary_condition_loss

    def train_model(self, optimizer, scheduler, initial_condition, epochs):
        history = []
        
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            
            physics_loss, initial_condition_loss, boundary_condition_loss = self.loss_function(initial_condition, *self.generator(self.t_min, self.t_max))
            total_loss = 16 * physics_loss + initial_condition_loss + boundary_condition_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        
            history.append(
                {
                    "total_loss": total_loss.item(),
                    "physics_loss": physics_loss.item(),
                    "initial_condition_loss": initial_condition_loss.item(),
                    "boundary_condition_loss": boundary_condition_loss.item(),
                }
            )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"Total loss: {total_loss.item():.4e}")
                print(f"Physics loss: {physics_loss.item():.4e}")
                print(f"Initial condition loss: {initial_condition_loss.item():.4e}")
                print(f"Boundary condition loss: {boundary_condition_loss.item():.4e}")
                print("-" * 50)

        return history

layers = [3, 512, 512, 512, 512, 512, 512, 2]

# Model setup
model = PINN(layers, t1, t2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 4000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

def ground_state_x(x):
    A = (m * omega / (np.pi * hbar)) ** 0.25
    alpha = (m * omega) / (2.0 * hbar)
    return A * torch.exp(-alpha * (x ** 2))

def ground_state_y(y):
    A = (m * omega / (np.pi * hbar)) ** 0.25
    alpha = (m * omega) / (2.0 * hbar)
    return A * torch.exp(-alpha * (y ** 2))

def ground_state(x, y):
    return ground_state_x(x) * ground_state_y(y), 0

history = model.train_model(optimizer, scheduler, ground_state, 300000)

torch.save(model.state_dict(), "Schrodinger-PINN/src/results/movement/model_16.pth")

with open("Schrodinger-PINN/src/results/movement/history_14.json", "w") as f:
    json.dump(history, f)
