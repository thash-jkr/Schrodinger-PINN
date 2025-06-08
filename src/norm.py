import numpy as np
import torch
import json
import torch.nn as nn
from scipy.constants import speed_of_light, elementary_charge, electron_mass, hbar
from scipy.integrate import simpson

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

x0 = 0
x1 = 75
t0 = 0
t1 = 2
t2 = t1 + (x1 - x0) / vQD

x_min = -75
x_max = 150
t_min = 0
t_max = 20

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
        self.n_initial = 500
        self.n_boundary = 500
        self.n_norm = 1000

        self.t_min = t_min
        self.t_max = t_max
    
    def forward(self, inputs):
        x, t = inputs
        X = torch.stack((x, t), dim=1)
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
        t_collocation = np.random.uniform(T_min, T_max, self.n_collocation)
        x_qd_collocation = np.where(t_collocation < t1, x0, np.where(t_collocation < t1 + (x1 - x0) / vQD, x0 + vQD * (t_collocation - t1), x1))
        x_collocation = np.random.normal(loc=x_qd_collocation, scale=25.0, size=self.n_collocation)
    
        x_c = 0
        x_initial = np.random.normal(loc=x_c, scale=25.0, size=self.n_initial)
        t_initial = np.full(self.n_initial, T_min)
        
        x_boundary = np.concatenate([np.full(self.n_boundary // 2, x_min), np.full(self.n_boundary // 2, x_max)])
        t_boundary = np.random.uniform(T_min, T_max, self.n_boundary)
        
        # x_norm = np.random.uniform(x_min, x_max, self.n_norm)
        # x_norm = np.random.normal(loc=x1, scale=25.0, size=self.n_norm)
        x_norm = np.linspace(x_min, x_max, self.n_norm)
        
        x_collocation_torch = torch.from_numpy(x_collocation).float().to(device)
        t_collocation_torch = torch.from_numpy(t_collocation).float().to(device)
        
        x_initial_torch  = torch.from_numpy(x_initial).float().to(device)
        t_initial_torch  = torch.from_numpy(t_initial).float().to(device)
        
        x_boundary_torch = torch.from_numpy(x_boundary).float().to(device)
        t_boundary_torch = torch.from_numpy(t_boundary).float().to(device)
        
        x_norm_torch = torch.from_numpy(x_norm).float().to(device).repeat(20)
        t_norm_torch = torch.arange(1, 20.1, device=device).float().repeat_interleave(self.n_norm)
    
        return x_collocation_torch, t_collocation_torch, x_initial_torch, t_initial_torch, x_boundary_torch, t_boundary_torch, x_norm_torch, t_norm_torch

    def loss_function(self, initial_condition, x_collocation_torch, t_collocation_torch, x_initial_torch, t_initial_torch, x_boundary_torch, t_boundary_torch, x_norm_torch, t_norm_torch, norm_ready):
        #pde loss
        x_collocation_torch = x_collocation_torch.clone().requires_grad_(True)
        t_collocation_torch = t_collocation_torch.clone().requires_grad_(True)
        
        u, v = self((x_collocation_torch, t_collocation_torch))
        
        du_dt = torch.autograd.grad(u, t_collocation_torch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x_collocation_torch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_collocation_torch, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        
        dv_dt = torch.autograd.grad(v, t_collocation_torch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dx = torch.autograd.grad(v, x_collocation_torch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        d2v_dx2 = torch.autograd.grad(dv_dx, x_collocation_torch, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0]
        
        xqd_arr = torch.where(t_collocation_torch < t1, x0, torch.where(t_collocation_torch < t1 + (x1 - x0) / vQD, x0 + vQD * (t_collocation_torch - t1), x1))
        
        real = -hbar * dv_dt + ((hbar ** 2) / (2 * m)) * d2u_dx2 - 0.5 * m * (omega ** 2) * ((x_collocation_torch - xqd_arr) ** 2) * u
        img = hbar * du_dt + ((hbar ** 2) / (2 * m)) * d2v_dx2 - 0.5 * m * (omega ** 2) * ((x_collocation_torch - xqd_arr) ** 2) * v
        
        # physics_loss = torch.mean(real ** 2 + img ** 2)
        
        cumulative_loss = 0
        physics_loss = 0
        segments = 20
        width = 20 / segments
        
        for k in range(segments):
            t_start = self.t_min + k * width
            t_end = t_start + width
            mask = (t_collocation_torch >= t_start) & (t_collocation_torch < t_end)
            
            # if mask.sum() == 0:
            #     continue
            
            loss = torch.mean(real[mask] ** 2 + img[mask] ** 2)
            cumulative_loss += loss
            physics_loss += cumulative_loss
            
        physics_loss /= segments
        
        
        
        
        #initial condition loss
        u_i, v_i = self((x_initial_torch, t_initial_torch))
        
        psi_initial_real, psi_initial_img = initial_condition(x_initial_torch, t_initial_torch)
        initial_condition_loss = torch.mean((u_i - psi_initial_real) ** 2) + torch.mean((v_i - psi_initial_img) ** 2)
        
        
        
        
        #boundary condition loss
        u_b, v_b = self((x_boundary_torch, t_boundary_torch))
        boundary_condition_loss = torch.mean(u_b ** 2) + torch.mean(v_b ** 2)
        
        
        
        
        #normalization loss
        if norm_ready:
            u_n, v_n = self((x_norm_torch, t_norm_torch))
            psi_sq = u_n ** 2 + v_n ** 2
            psi_sq = psi_sq.view(20, self.n_norm)
            
            integrals = psi_sq.mean(dim=1) * (x_max - x_min)
            normalization_loss = ((integrals - 1.0) ** 2).mean()
        else:
            normalization_loss = torch.tensor(0)
        
        return physics_loss, initial_condition_loss, boundary_condition_loss, normalization_loss

    def train_model(self, optimizer, scheduler, initial_condition, epochs):
        history = []
        
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            
            if epoch < 250000: 
                norm_ready = False
            else:
                norm_ready = True
            
            physics_loss, initial_condition_loss, boundary_condition_loss, normalization_loss = self.loss_function(initial_condition, *self.generator(self.t_min, self.t_max), norm_ready)
            total_loss = 16 * physics_loss + initial_condition_loss + boundary_condition_loss + normalization_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        
            history.append(
                {
                    "total_loss": total_loss.item(),
                    "physics_loss": physics_loss.item(),
                    "initial_condition_loss": initial_condition_loss.item(),
                    "boundary_condition_loss": boundary_condition_loss.item(),
                    "normalization_loss": normalization_loss.item(),
                }
            )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"Total loss: {total_loss.item():.4e}")
                print(f"Physics loss: {physics_loss.item():.4e}")
                print(f"Initial condition loss: {initial_condition_loss.item():.4e}")
                print(f"Boundary condition loss: {boundary_condition_loss.item():.4e}")
                print(f"Normalization loss: {normalization_loss.item():.4e}")
                print("-" * 50)

        return history

layers = [2, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 2]

# Model setup
model = PINN(layers, 0, 20).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 4000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

def ground_state(x, t):
    return (((m * omega) / (np.pi * hbar)) ** 0.25) * torch.exp(((-m * omega) / (2 * hbar)) * (x ** 2)), 0

history = model.train_model(optimizer, scheduler, ground_state, 350000)

torch.save(model.state_dict(), "Schrodinger-PINN/src/results/norm/model_13.pth")

with open("Schrodinger-PINN/src/results/norm/history_13.json", "w") as f:
    json.dump(history, f)
