# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from physics.utils.constants import hbar_meV_ps as hbar, e as e_charge, me as m_electron, mt as m_eff

Lx = 50
Lz = 50
T = 1

qd_x_start = 15
qd_x_end = 25
v_qd = (qd_x_end - qd_x_start) / T

Ez = 5e-3

omega = 2 / hbar

# Define the potential function
def potnetial_function(x, z, t):
  x_qd = qd_x_start + v_qd * t

  V_qd = 0.5 * m_eff * omega**2 * (x - x_qd)**2
  V_electric = - e_charge * Ez * z

  V_total = V_qd + V_electric

  return V_total

# Neural Network Architecture
class PINN(tf.keras.Model):
  def __init__(self, layers):
    super(PINN, self).__init__()
    self.hidden_layers = [tf.keras.layers.Dense(units, activation='tanh', dtype='float64') for units in layers[:-1]]
    self.output_layer = tf.keras.layers.Dense(layers[-1], activation=None, dtype='float64')

  def call(self, inputs):
    x, z, t = inputs
    X = tf.stack([x, z, t], axis=1)
    
    for layer in self.hidden_layers:
        X = layer(X)

    output = self.output_layer(X)
    psi_real = output[:, 0:1]
    psi_imag = output[:, 1:2]
    return psi_real, psi_imag

# Define the layers: [input_size, hidden1, hidden2, ..., output_size]
layers = [3, 50, 50, 50, 50, 2]  # 3 inputs (x, z, t), several hidden layers, 2 outputs (Re[ψ], Im[ψ])

# Generate collocation points
# Number of collocation points
n_collocation = 10000
n_initial = 2000
n_boundary = 1000

x_min, x_max = 0, Lx
z_min, z_max = 0, Lz
t_min, t_max = 0, T

x_collocation = np.random.uniform(x_min, x_max, n_collocation)
z_collocation = np.random.uniform(z_min, z_max, n_collocation)
t_collocation = np.random.uniform(t_min, t_max, n_collocation)

x_initial = np.random.uniform(x_min, x_max, n_initial)
z_initial = np.random.uniform(z_min, z_max, n_initial)
t_initial = np.full(n_initial, t_min)

x_boundary_x = np.concatenate([np.full(n_boundary // 2, x_min), np.full(n_boundary // 2, x_max)])
z_boundary_x = np.random.uniform(z_min, z_max, n_boundary)
t_boundary_y = np.random.uniform(t_min, t_max, n_boundary)

x_boundary_z = np.random.uniform(x_min, x_max, n_boundary)
z_boundary_z = np.concatenate([np.full(n_boundary // 2, z_min), np.full(n_boundary // 2, z_max)])
t_boundary_z = np.random.uniform(t_min, t_max, n_boundary)

x_boundary_total = np.concatenate([x_boundary_x, x_boundary_z])
z_boundary_total = np.concatenate([z_boundary_x, z_boundary_z])
t_boundary_total = np.concatenate([t_boundary_y, t_boundary_z])

# Convert numpy arrays to tensors and ensure correct data type
x_collocation_tf = tf.convert_to_tensor(x_collocation, dtype=tf.float64)
z_collocation_tf = tf.convert_to_tensor(z_collocation, dtype=tf.float64)
t_collocation_tf = tf.convert_to_tensor(t_collocation, dtype=tf.float64)

x_initial_tf = tf.convert_to_tensor(x_initial, dtype=tf.float64)
z_initial_tf = tf.convert_to_tensor(z_initial, dtype=tf.float64)
t_initial_tf = tf.convert_to_tensor(t_initial, dtype=tf.float64)

x_boundary_tf = tf.convert_to_tensor(x_boundary_total, dtype=tf.float64)
z_boundary_tf = tf.convert_to_tensor(z_boundary_total, dtype=tf.float64)
t_boundary_tf = tf.convert_to_tensor(t_boundary_total, dtype=tf.float64)

# Normalize inputs to [-1, 1]
def normalize(value, min_value, max_value):
    return 2.0 * (value - min_value) / (max_value - min_value) - 1.0

x_collocation_norm = normalize(x_collocation_tf, x_min, x_max)
z_collocation_norm = normalize(z_collocation_tf, z_min, z_max)
t_collocation_norm = normalize(t_collocation_tf, t_min, t_max)

x_initial_norm = normalize(x_initial_tf, x_min, x_max)
z_initial_norm = normalize(z_initial_tf, z_min, z_max)
t_initial_norm = normalize(t_initial_tf, t_min, t_max)

x_boundary_norm = normalize(x_boundary_tf, x_min, x_max)
z_boundary_norm = normalize(z_boundary_tf, z_min, z_max)
t_boundary_norm = normalize(t_boundary_tf, t_min, t_max)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Initialize the model
model = PINN(layers)

# Define the loss function
@tf.function
def loss_fn():
    # Physics loss (PDE residual)
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x_collocation_norm, z_collocation_norm, t_collocation_norm])
        psi_real, psi_imag = model((x_collocation_norm, z_collocation_norm, t_collocation_norm))
        psi = tf.complex(psi_real, psi_imag)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x_collocation_norm, z_collocation_norm, t_collocation_norm])
        psi_real, psi_imag = model((x_collocation_norm, z_collocation_norm, t_collocation_norm))
        psi = tf.complex(psi_real, psi_imag)
        psi_x = tape2.gradient(psi, x_collocation_norm) / ((x_max - x_min) / 2)
        psi_z = tape2.gradient(psi, z_collocation_norm) / ((z_max - z_min) / 2)
        psi_t = tape2.gradient(psi, t_collocation_norm) / ((t_max - t_min) / 2)
    psi_xx = tape1.gradient(psi_x, x_collocation_norm) / ((x_max - x_min) / 2)
    psi_zz = tape1.gradient(psi_z, z_collocation_norm) / ((z_max - z_min) / 2)
    del tape1, tape2  # Free up resources

    # Rescale second derivatives due to normalization
    psi_xx = psi_xx / ((x_max - x_min) / 2)
    psi_zz = psi_zz / ((z_max - z_min) / 2)
    psi_t = psi_t / ((t_max - t_min) / 2)

    # Compute potential at collocation points
    x_collocation_actual = x_collocation_tf  # in nm
    z_collocation_actual = z_collocation_tf  # in nm
    t_collocation_actual = t_collocation_tf  # in ps
    V = potential_function(x_collocation_actual, z_collocation_actual, t_collocation_actual)
    V = tf.cast(V, dtype=tf.complex128)

    # Schrödinger equation residual
    residual = (1j * hbar * psi_t + (hbar**2 / (2 * mt)) * (psi_xx + psi_zz) - V * psi)
    physics_loss = tf.reduce_mean(tf.square(tf.abs(residual)))

    # Boundary loss
    psi_real_b, psi_imag_b = model((x_boundary_norm, z_boundary_norm, t_boundary_norm))
    psi_b = tf.complex(psi_real_b, psi_imag_b)
    boundary_loss = tf.reduce_mean(tf.square(tf.abs(psi_b)))

    # Initial condition loss
    psi_real_i, psi_imag_i = model((x_initial_norm, z_initial_norm, t_initial_norm))
    psi_i = tf.complex(psi_real_i, psi_imag_i)
    # Define initial wave function (e.g., ground state of initial potential)
    x_initial_actual = x_initial_tf
    z_initial_actual = z_initial_tf
    # Ground state of harmonic oscillator in x, plane wave in z
    alpha = mt * omega / hbar  # in 1/nm^2
    psi_initial = tf.exp(-alpha * (x_initial_actual - qd_x_start)**2)
    psi_initial = tf.cast(psi_initial, dtype=tf.complex128)
    initial_loss = tf.reduce_mean(tf.square(tf.abs(psi_i - psi_initial)))

    # Total loss
    total_loss = physics_loss + boundary_loss + initial_loss

    return total_loss

# Training loop
epochs = 10000  # Adjust as needed
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_value = loss_fn()
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Save loss history
    loss_history.append(loss_value.numpy())

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")

# Plot loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.show()

# Validation and Visualization
# Create a grid for evaluation
x_eval = np.linspace(x_min, x_max, 100)
z_eval = np.linspace(z_min, z_max, 100)
t_eval = np.linspace(t_min, t_max, 5)  # Evaluate at 5 time snapshots

X_eval, Z_eval, T_eval = np.meshgrid(x_eval, z_eval, t_eval, indexing='ij')

# Flatten and normalize inputs
x_eval_flat = X_eval.flatten()
z_eval_flat = Z_eval.flatten()
t_eval_flat = T_eval.flatten()

x_eval_norm = normalize(x_eval_flat, x_min, x_max)
z_eval_norm = normalize(z_eval_flat, z_min, z_max)
t_eval_norm = normalize(t_eval_flat, t_min, t_max)

# Evaluate the model
psi_real_eval, psi_imag_eval = model((x_eval_norm.astype(np.float64),
                                      z_eval_norm.astype(np.float64),
                                      t_eval_norm.astype(np.float64)))
psi_eval = psi_real_eval.numpy().flatten() + 1j * psi_imag_eval.numpy().flatten()
psi_eval = psi_eval.reshape(X_eval.shape)

# Compute probability density
prob_density = np.abs(psi_eval)**2

# Plot probability density at different times
for i in range(t_eval.shape[0]):
    plt.figure()
    plt.contourf(X_eval[:, :, i], Z_eval[:, :, i], prob_density[:, :, i], levels=50, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.xlabel('x (nm)')
    plt.ylabel('z (nm)')
    plt.title(f'Probability Density at t = {t_eval[i]:.2f} ps')
    plt.show()
