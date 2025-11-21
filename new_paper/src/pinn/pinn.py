import numpy as np
import torch
import json
import torch.nn as nn
from scipy.constants import speed_of_light, elementary_charge, electron_mass, hbar
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

me_SI = electron_mass
hbar_SI = hbar
e_SI = elementary_charge
c_SI = speed_of_light

meV = e_SI * 1e-3
nm = 1e-9
ps = 1e-12

c = c_SI * ps / nm
hbar_meV_ps = hbar_SI / (meV * ps)
me = me_SI * c_SI ** 2 / meV / c ** 2

hbar = hbar_meV_ps
m = me

m = m * 0.98 
omega = 1 / hbar
alpha_ent_barr = 0.4900
alpha_ent_exit_barr = 0.0370
alpha_exit_barr = 0.4800
alpha_exit_ent_bar = 0.0520
V_ent = -0.7000
V_exit = -0.7000
V_amp = 1.4150
f = 4
x_ent = 0
x_exit = 100
U_scr = 1
L_ent = 100
L_exit = 100
L_scr = 1

x_min, x_max = -50, 150
t_min, t_max = 0, 7

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
    

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        for units in layers[1:-1]:
            self.hidden_layers.append(nn.Linear(in_features=layers[0], out_features=units))
            layers[0] = units
        
        self.output_layer = nn.Linear(layers[-2], layers[-1])

        self.n_collocation = 5000
        self.n_initial = 5000
        self.n_boundary = 500
    
    def forward(self, inputs):
        x, t = inputs
        X = torch.stack((x, t), dim=1)
        activation_1 = nn.Tanh()
        activation_2 = nn.SiLU()
        i = 1
    
        for layer in self.hidden_layers:
            X = layer(X)
            X = activation_1(X) # if i < 2 else activation_2(X)
            i += 1
    
        output = self.output_layer(X)
        psi_real = output[:, 0]
        psi_img = output[:, 1]
        
        return psi_real, psi_img

    def generator(self):
        t_collocation = np.random.uniform(t_min, t_max, self.n_collocation)
        x_collocation = np.random.uniform(x_min, x_max, self.n_collocation)
    
        x_initial = np.random.uniform(x_min, x_max, self.n_initial)
        t_initial = np.full(self.n_initial, t_min)
        
        x_boundary = np.concatenate([np.full(self.n_boundary // 2, x_min), np.full(self.n_boundary // 2, x_max)])
        t_boundary = np.random.uniform(t_min, t_max, self.n_boundary)
        
        x_collocation_torch = torch.from_numpy(x_collocation).float().to(device)
        t_collocation_torch = torch.from_numpy(t_collocation).float().to(device)
        
        x_initial_torch  = torch.from_numpy(x_initial).float().to(device)
        t_initial_torch  = torch.from_numpy(t_initial).float().to(device)
        
        x_boundary_torch = torch.from_numpy(x_boundary).float().to(device)
        t_boundary_torch = torch.from_numpy(t_boundary).float().to(device)
    
        return x_collocation_torch, t_collocation_torch, x_initial_torch, t_initial_torch, x_boundary_torch, t_boundary_torch
    
    def theta(self, arr):
        return torch.where(arr > 0, 1, 0)
    
    def complex_potential(self, x, t):
        V_ac = V_amp * torch.cos(torch.tensor(2 * torch.pi * f * t * 1e-3, device=device))
        A_ent = -alpha_ent_barr * (V_ent + V_ac)
        B_ent = (alpha_ent_barr / alpha_ent_exit_barr) ** (-torch.abs(x - x_ent) / torch.abs(torch.tensor(x_exit - x_ent, device=device)))
        U_ent = A_ent * B_ent
        
        A_exit = -alpha_exit_barr * V_exit
        B_exit = (alpha_exit_barr / alpha_exit_ent_bar) ** (-torch.abs(x - x_exit) / torch.abs(torch.tensor(x_exit - x_ent, device=device)))
        U_exit = A_exit * B_exit
        
        A_upper = U_scr * torch.exp(-((x - x_ent) / L_scr) * self.theta(x - x_ent)) * torch.exp(-((x_ent - L_ent - x) / L_scr) * self.theta(x_ent - L_ent - x))
        B_upper = U_scr * torch.exp(-((x - x_exit - L_exit) / L_scr) * self.theta(x - x_exit - L_exit)) * torch.exp(-((x_exit - x) / L_scr) * self.theta(x_exit - x))
        U_upper = A_upper + B_upper
        
        return U_ent + U_exit + U_upper

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
        
        real = -hbar * dv_dt + ((hbar ** 2) / (2 * m)) * d2u_dx2 - self.complex_potential(x_collocation_torch, t_collocation_torch) * u
        img = hbar * du_dt + ((hbar ** 2) / (2 * m)) * d2v_dx2 - self.complex_potential(x_collocation_torch, t_collocation_torch) * v
        
        physics_loss = torch.mean(real ** 2 + img ** 2)
        
        
        
        
        #initial condition loss
        u_i, v_i = self((x_initial_torch, t_initial_torch))
        
        psi_initial_real, psi_initial_img = initial_condition
        initial_condition_loss = torch.mean((u_i - torch.abs(psi_initial_real)) ** 2) + torch.mean((v_i - torch.abs(psi_initial_img)) ** 2)
        
        
        
        
        #boundary condition loss
        u_b, v_b = self((x_boundary_torch, t_boundary_torch))
        boundary_condition_loss = torch.mean(u_b ** 2) + torch.mean(v_b ** 2)
        
        return physics_loss, initial_condition_loss, boundary_condition_loss

    def train_model(self, optimizer, scheduler, initial_condition, epochs):
        history = []
        
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            
            physics_loss, initial_condition_loss, boundary_condition_loss = self.loss_function(initial_condition, *self.generator())
            total_loss = physics_loss + initial_condition_loss + boundary_condition_loss
            
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
    
layers = [2, 512, 512, 512, 512, 512, 512, 2]

model = PINN(layers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.9))
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

decay_rate = 0.9
steps = 4000

def exp_decay(step):
    return decay_rate ** (step / steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_decay)

x_values = np.linspace(x_min, x_max, 5000)
dx = x_values[1] - x_values[0]
ground_potential = model.complex_potential(torch.tensor(x_values, device=device), 0)
laplacian = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(5000, 5000), format='csc') / dx ** 2
H_kinetic = - (hbar ** 2 / (2 * m)) * laplacian
H = H_kinetic + sp.diags(ground_potential.cpu().numpy(), format="csc")
w, v = eigsh(H, k=1, which="SA")
psi0 = v[:, 0]
psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)
initial_condition = (torch.tensor(psi0, device=device), torch.zeros_like(torch.tensor(psi0, device=device)))

history = model.train_model(optimizer, scheduler, initial_condition, 50000)

torch.save(model.state_dict(), "Schrodinger-PINN/src/results/complex_potential/model_3.pth")

with open("Schrodinger-PINN/src/results/complex_potential/history_3.json", "w") as f:
    json.dump(history, f)
