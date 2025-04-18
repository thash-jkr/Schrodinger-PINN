import numpy as np
import torch
import json
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
vQD = 1

x0 = 0
x1 = 75
t0 = 0
t1 = 2
t2 = t1 + (x1 - x0) / vQD

x_min = -75
x_max = 150
t_min = 0
t_max = 100

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

        self.n_collocation = 4000
        self.n_initial = 2000
        self.n_boundary = 1000

        self.t_min = t_min
        self.t_max = t_max
    
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

    def generator(self, T_min, T_max):
        t_collocation = np.random.uniform(T_min, T_max, self.n_collocation)
        x_qd_collocation = np.where(t_collocation < t1, x0, np.where(t_collocation < t1 + (x1 - x0) / vQD, x0 + vQD * (t_collocation - t1), x1))
        x_collocation = np.random.normal(loc=x_qd_collocation, scale=25.0, size=self.n_collocation)
    
        x_c = np.where(T_min < t1, x0, np.where(T_min < t1 + (x1 - x0) / vQD, x0 + vQD * (T_min - t1), x1))
        x_initial = np.random.normal(loc=x_c, scale=25.0, size=self.n_initial)
        t_initial = np.full(self.n_initial, T_min)
        
        x_boundary = np.concatenate([np.full(self.n_boundary // 2, x_min), np.full(self.n_boundary // 2, x_max)])
        t_boundary = np.random.uniform(T_min, T_max, self.n_boundary)
        
        x_collocation_torch = torch.from_numpy(x_collocation).float().to(device)
        t_collocation_torch = torch.from_numpy(t_collocation).float().to(device)
        
        x_initial_torch  = torch.from_numpy(x_initial).float().to(device)
        t_initial_torch  = torch.from_numpy(t_initial).float().to(device)
        
        x_boundary_torch = torch.from_numpy(x_boundary).float().to(device)
        t_boundary_torch = torch.from_numpy(t_boundary).float().to(device)
    
        return x_collocation_torch, t_collocation_torch, x_initial_torch, t_initial_torch, x_boundary_torch, t_boundary_torch

    def loss_function(self, initial_condition, x_collocation_torch, t_collocation_torch, x_initial_torch, t_initial_torch, x_boundary_torch, t_boundary_torch):
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
        
        physics_loss = torch.mean(real ** 2 + img ** 2)
        
        
        
        
        #initial condition loss
        u_i, v_i = self((x_initial_torch, t_initial_torch))
        
        psi_initial_real, psi_initial_img = initial_condition(x_initial_torch, t_initial_torch)
        initial_condition_loss = torch.mean((u_i - psi_initial_real) ** 2) + torch.mean((v_i - psi_initial_img) ** 2)
        
        
        
        
        #boundary condition loss
        u_b, v_b = self((x_boundary_torch, t_boundary_torch))
        boundary_condition_loss = torch.mean(u_b ** 2) + torch.mean(v_b ** 2)
        
        return physics_loss, initial_condition_loss, boundary_condition_loss

    def train_model(self, optimizer, scheduler, initial_condition, epochs):
        history = []
        
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            
            physics_loss, initial_condition_loss, boundary_condition_loss = self.loss_function(initial_condition, *self.generator(self.t_min, self.t_max))
            total_loss = 10 * physics_loss + initial_condition_loss + boundary_condition_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        
            history.append({"total_loss": total_loss.item() ,"physics_loss": physics_loss.item(), "initial_condition_loss": initial_condition_loss.item(), "boundary_condition_loss": boundary_condition_loss.item()})

        return history


def ground_state(x, t):
    return (((m * omega) / (np.pi * hbar)) ** 0.25) * torch.exp(((-m * omega) / (2 * hbar)) * (x ** 2)), 0

print("Starting model 1 training")

# Model 1
layers_1 = [2, 512, 512, 512, 512, 512, 512, 2]
model_1 = PINN(layers_1, 0, t1).to(device)

optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 2000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

history_1 = model_1.train_model(optimizer, scheduler, ground_state, 100000)

torch.save(model_1.state_dict(), "Schrodinger-PINN/src/special_results/vqd1/model_1.pth")

with open("Schrodinger-PINN/src/special_results/vqd1/history_1.json", "w") as f:
    json.dump(history_1, f)


print("Finished model 1 training")
print("Starting model 2 training")


# Model 2
layers_2 = [2, 512, 512, 512, 512, 512, 512, 2]
model_2 = PINN(layers_2, 0, t2).to(device)

optimizer = torch.optim.Adam(model_2.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 2000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

history_2 = model_2.train_model(optimizer, scheduler, ground_state, 100000)

torch.save(model_2.state_dict(), "Schrodinger-PINN/src/special_results/vqd1/model_2.pth")

with open("Schrodinger-PINN/src/special_results/vqd1/history_2.json", "w") as f:
    json.dump(history_2, f)


print("Finished model 2 training")
print("Starting model 3 training")


# Model 3
layers_3 = [2, 512, 512, 512, 512, 512, 512, 2]
model_3 = PINN(layers_3, 0, 100).to(device)

optimizer = torch.optim.Adam(model_3.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 2000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

history_3 = model_3.train_model(optimizer, scheduler, ground_state, 100000)

torch.save(model_3.state_dict(), "Schrodinger-PINN/src/special_results/vqd1/model_3.pth")

with open("Schrodinger-PINN/src/special_results/vqd1/history_3.json", "w") as f:
    json.dump(history_3, f)


print("Finished model 3 training")
print("Starting model 4 training")


# Model 4
layers_4 = [2, 512, 512, 512, 512, 512, 512, 2]
model_4 = PINN(layers_4, t1, 100).to(device)

optimizer = torch.optim.Adam(model_4.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 2000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

def ic(x, t):
    times = torch.full(x.size(), t1).to(device)
    return model_1((x, times))

history_4 = model_4.train_model(optimizer, scheduler, ic, 100000)

torch.save(model_4.state_dict(), "Schrodinger-PINN/src/special_results/vqd1/model_4.pth")

with open("Schrodinger-PINN/src/special_results/vqd1/history_4.json", "w") as f:
    json.dump(history_4, f)


print("Finished model 4 training")
print("Starting model 5.1 training")


# Model 5
layers_5_1 = [2, 512, 512, 512, 512, 512, 512, 2]
model_5_1 = PINN(layers_5_1, t1, t2).to(device)

optimizer = torch.optim.Adam(model_5_1.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 2000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

def ic(x, t):
    times = torch.full(x.size(), t1).to(device)
    return model_1((x, times))

history_5_1 = model_5_1.train_model(optimizer, scheduler, ic, 100000)

torch.save(model_5_1.state_dict(), "Schrodinger-PINN/src/special_results/vqd1/model_5_1.pth")

with open("Schrodinger-PINN/src/special_results/vqd1/history_5_1.json", "w") as f:
    json.dump(history_5_1, f)


print("Finished model 5.1 training")
print("Starting model 5.2 training")


layers_5_2 = [2, 512, 512, 512, 512, 512, 512, 2]
model_5_2 = PINN(layers_5_2, t2, 100).to(device)

optimizer = torch.optim.Adam(model_5_2.parameters(), lr=0.001, betas=(0.9, 0.9))

decay_rate = 0.9
steps = 2000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

def ic(x, t):
    times = torch.full(x.size(), t2).to(device)
    return model_2((x, times))

history_5_2 = model_5_2.train_model(optimizer, scheduler, ic, 100000)

torch.save(model_5_2.state_dict(), "Schrodinger-PINN/src/special_results/vqd1/model_5_2.pth")

with open("Schrodinger-PINN/src/special_results/vqd1/history_5_2.json", "w") as f:
    json.dump(history_5_2, f)


print("Finished model 5.2 training")