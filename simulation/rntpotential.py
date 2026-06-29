import numpy as np

# Initialise parameters

L = 2 * np.pi # length of ring
w = 1.5 # self-propulsion speed
gamma = 0.5 # tumbling rate
D = 0.05 # diffusion constant
dt = 0.001 # time step
T = 50.0 # total simulation time
n_steps = int(T/dt)

# Defining the potential
# Want to make a variable shape here
# tests out all possible shapes

nu = 1.0 # strength of potential
def U(x):
    return 1

def dU_dx(x):
    return 0

# REPLACE ABOVE SECTION WITH VARIABLE DEFINITION OF POTENTIAL

def phi(x):
    return nu*U(x)

def dphi_dx(x):
    return nu*dU_dx(x)


def update_orientation(u, gamma, dt, rng):
    if rng.random() < gamma * dt:
        return -u
    return u

def step_x_with_unwrap(x, x_unwrapped, u, dt, D, rng):
    drift = -dphi_dx(x) + w * u
    noise = np.sqrt(2 * D * dt) * rng.normal()

    x_new_unwrapped = x_unwrapped + drift * dt + noise
    x_new = x_new_unwrapped % L
    return x_new, x_new_unwrapped

def simulate_trajectory(w_val, gamma_val, D_val, T, dt, rng_seed=123):
    global w, gamma, D
    w, gamma, D = w_val, gamma_val, D_val

    n_steps = int(T / dt)

    # Initial state conditions
    x = 0.0
    x_unwrapped = 0.0
    u = 1 # u(t) = 1 or -1

    rng = np.random.default_rng(rng_seed)

    # Optionally record, but for current we only really need x_unwrapped
    for n in range(n_steps):
        u = update_orientation(u, gamma, dt, rng)
        x, x_unwrapped = step_x_with_unwrap(x, x_unwrapped, u, dt, D, rng)

    # Net displacement and average velocity
    net_displacement = x_unwrapped  # since we started at 0
    avg_velocity = net_displacement / T  # J ~ ⟨\dot x⟩

    return avg_velocity

def estimate_current(w_val, gamma_val, D_val, T=50.0, dt=0.001, rng_seed=123):
    return simulate_trajectory(w_val, gamma_val, D_val, T, dt, rng_seed=rng_seed)

# Example: scan over w for fixed gamma, D
w_values = np.linspace(0.5, 3.0, 11)
currents = []

for w_test in w_values:
    J_est = estimate_current(w_test, gamma, D, T=100.0, dt=0.001, rng_seed=123)
    currents.append(J_est)
    print(f"w = {w_test:.2f}, J ≈ {J_est:.4f}")

# Find approximate maximiser
max_idx = int(np.argmax(currents))
w_opt = w_values[max_idx]
J_opt = currents[max_idx]
print(f"\nApproximate optimal w ≈ {w_opt:.3f} with J ≈ {J_opt:.4f}")

gamma_values = np.linspace(0.1, 2.0, 10)
currents_gamma = []

for gamma_test in gamma_values:
    J_est = estimate_current(w, gamma_test, D, T=100.0, dt=0.001, rng_seed=123)
    currents_gamma.append(J_est)
    print(f"gamma = {gamma_test:.2f}, J ≈ {J_est:.4f}")