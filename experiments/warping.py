import math
import h5py
import mujoco
import mediapy
import warp as wp
import numpy as np
from tqdm import tqdm
import mujoco_warp as mjw

# Initialize Warp device
wp.init()
device = "cuda" # Ensure running on GPU

### SETUP ###
NWORLD = 512
new_file_path = "configs/stable/gobox.h5"
mujoco_xml = "configs/mujoco_/unitree_go2/box_scene.xml"
start_idx = 1
end_idx = 2
ctrl_n = 4
tau_action = 0.25
tau_sim = 0.01
action_steps = math.ceil(tau_action / tau_sim)
num_generations = 10
noise_std = 10.
lam = 1.  # MPPI temperature

geoms_in_cost_names = [
    "FL", "FR",
    "RL", "RR",
    "box_marker_0", "box_marker_1", "box_marker_2"
]

geom_weights = wp.array([
    1., 1.,     # Feet
    1., 1.,     # Hands
    10., 10., 10.  # Box
], dtype=wp.float32, device=device)

mj_model = mujoco.MjModel.from_xml_path(mujoco_xml)
mj_model.opt.timestep = tau_sim
mj_model.opt.ccd_iterations = 200
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetData(mj_model, mj_data)

geom_indices_np = np.array([mj_model.geom(name).id for name in geoms_in_cost_names], dtype=np.int32)
geom_indices = wp.array(geom_indices_np, dtype=wp.int32, device=device)

model = mjw.put_model(mj_model)
data = mjw.put_data(mj_model, mj_data, nworld=NWORLD, nconmax=250, njmax=250)

file = h5py.File(new_file_path, 'r')
stable_configs = file["qpos"]
stable_configs_ctrl = file["ctrl"]
initial_ctrl_np = np.array(stable_configs_ctrl[start_idx], dtype=np.float32)
initial_ctrl = wp.array(initial_ctrl_np, dtype=wp.float32, device=device)

# Get target xpos
mj_data.qpos[:] = stable_configs[end_idx]
mujoco.mj_forward(mj_model, mj_data)
target_geom_xpos_np = np.array([mj_data.geom_xpos[idx] for idx in geom_indices_np], dtype=np.float32)
target_geom_xpos = wp.array(target_geom_xpos_np, dtype=wp.vec3, device=device)

# Nominal control sequence: (ctrl_n, nu)
U_np = np.tile(initial_ctrl_np, (ctrl_n, 1)).astype(np.float32)
for i in range(ctrl_n):
    alpha = (i+1) / ctrl_n
    U_np[i] = (1 - alpha) * stable_configs_ctrl[start_idx] + alpha * stable_configs_ctrl[end_idx]
U = wp.array(U_np, dtype=wp.float32, device=device)
ctrl_low = wp.array(mj_model.actuator_ctrlrange[:, 0], dtype=wp.float32, device=device)
ctrl_high = wp.array(mj_model.actuator_ctrlrange[:, 1], dtype=wp.float32, device=device)

### WARP KERNELS ###
@wp.kernel
def interpolate_ctrl_kernel(
    U: wp.array(dtype=wp.float32, ndim=2),          # Explicitly 2D (ctrl_n, nu)
    noise: wp.array(dtype=wp.float32, ndim=3),      # Explicitly 3D (NWORLD, ctrl_n, nu)
    initial_ctrl: wp.array(dtype=wp.float32, ndim=1),# Explicitly 1D (nu,)
    target_ctrl_idx: int,
    alpha: float,
    out_ctrl: wp.array(dtype=wp.float32, ndim=2)    # Explicitly 2D (NWORLD, nu)
):
    world, nu = wp.tid()
    
    # Extract control state A
    if target_ctrl_idx == 0:
        ctrl_a = initial_ctrl[nu]
    else:
        ctrl_a = U[target_ctrl_idx - 1, nu] + noise[world, target_ctrl_idx - 1, nu]
        
    # Extract control state B
    ctrl_b = U[target_ctrl_idx, nu] + noise[world, target_ctrl_idx, nu]
    
    # Linearly interpolate execution steps directly into the simulation buffers
    out_ctrl[world, nu] = (1.0 - alpha) * ctrl_a + alpha * ctrl_b

@wp.kernel
def running_cost_kernel(
    qvel: wp.array(dtype=wp.float32, ndim=2),
    out_cost: wp.array(dtype=wp.float32, ndim=1)
):
    world = wp.tid()
    cost = float(0.0)
    for i in range(qvel.shape[1]):
        v = qvel[world, i]
        cost += v * v
    out_cost[world] = cost * 0.001

@wp.kernel
def compute_cost_kernel(
    geom_xpos: wp.array(dtype=wp.vec3, ndim=2),
    target_geom_xpos: wp.array(dtype=wp.vec3, ndim=1),
    geom_weights: wp.array(dtype=wp.float32, ndim=1),
    geom_indices: wp.array(dtype=wp.int32, ndim=1),
    out_cost: wp.array(dtype=wp.float32, ndim=1)
):
    world = wp.tid()
    sum_sq_err = float(0.0)
    num_geoms = geom_indices.shape[0]
    
    for i in range(num_geoms):
        g_idx = geom_indices[i]
        weight = geom_weights[i]
        
        # Extract the vec3 structures
        p_curr = geom_xpos[world, g_idx]
        p_targ = target_geom_xpos[i]
        
        # Compute differences component-wise
        dx = p_curr[0] - p_targ[0]
        dy = p_curr[1] - p_targ[1]
        dz = p_curr[2] - p_targ[2]
        
        sum_sq_err += (dx * dx + dy * dy + dz * dz) * (weight * weight)
        
    out_cost[world] = sum_sq_err


@wp.kernel
def update_u_kernel(
    U: wp.array(dtype=wp.float32, ndim=2),
    noise: wp.array(dtype=wp.float32, ndim=3),
    weights: wp.array(dtype=wp.float32, ndim=1),
    ctrl_low: wp.array(dtype=wp.float32, ndim=1),
    ctrl_high: wp.array(dtype=wp.float32, ndim=1)
):
    t_idx, nu_idx = wp.tid()
    
    delta = float(0.0)  # Explicitly declared as a mutable dynamic float
    n_world = weights.shape[0]
    for w in range(n_world):
        delta += weights[w] * noise[w, t_idx, nu_idx]
        
    new_val = U[t_idx, nu_idx] + delta
    
    # Native clamping bounds evaluation
    low = ctrl_low[nu_idx]
    high = ctrl_high[nu_idx]
    if new_val < low:
        new_val = low
    elif new_val > high:
        new_val = high
    
    U[t_idx, nu_idx] = new_val


### MPPI LOOP SETUP ###
best_cost = float('inf')
best_U = wp.array(U_np, dtype=wp.float32, device=device)

qpos_init = wp.array(np.tile(stable_configs[start_idx], (NWORLD, 1)), dtype=wp.float32, device=device)
ctrl_init = wp.array(np.tile(stable_configs_ctrl[start_idx], (NWORLD, 1)), dtype=wp.float32, device=device)
terminal_cost = wp.zeros(shape=(NWORLD,), dtype=wp.float32, device=device)
running_cost = wp.zeros(shape=(NWORLD,), dtype=wp.float32, device=device)
cost = wp.zeros(shape=(NWORLD,), dtype=wp.float32, device=device)
qvel_dim = data.qvel.shape[-1]

### MPPI LOOP ###
for i in tqdm(range(num_generations), total=num_generations):
    # Reset all parallel worlds to starting config
    cost *= 0.
    wp.copy(data.qpos, qpos_init)
    wp.copy(data.ctrl, ctrl_init)
    mjw.forward(model, data)

    # Sample random perturbations using NumPy and shift directly to Warp
    noise_np = np.random.normal(0.0, noise_std, size=(NWORLD, ctrl_n, mj_model.nu)).astype(np.float32)
    noise = wp.array(noise_np, dtype=wp.float32, device=device)

    # Parallel Rollout Simulation
    for target_ctrl_idx in range(ctrl_n):
        for t in range(action_steps):
            alpha = float(t) / float(action_steps - 1) if action_steps > 1 else 0.0
            
            wp.launch(
                kernel=interpolate_ctrl_kernel,
                dim=(NWORLD, mj_model.nu),
                inputs=[U, noise, initial_ctrl, target_ctrl_idx, alpha, data.ctrl],
                device=device
            )
            mjw.step(model, data)

        wp.launch(
            kernel=running_cost_kernel,
            dim=(NWORLD, qvel_dim),
            inputs=[data.qvel, running_cost],
            device=device
        )
        cost += running_cost

    # Execute GPU cost updates
    wp.launch(
        kernel=compute_cost_kernel,
        dim=(NWORLD,),
        inputs=[data.geom_xpos, target_geom_xpos, geom_weights, geom_indices, terminal_cost],
        device=device
    )
    cost += terminal_cost
    
    # Pull lightweight cost scalar properties back to Host for quick evaluation
    cost_np = cost.numpy()
    min_cost = float(np.min(cost_np))

    # MPPI weight updates using NumPy
    beta = min_cost
    weights_np = np.exp(-(cost_np - beta) / lam)
    weights_np /= np.sum(weights_np)
    weights = wp.array(weights_np, dtype=wp.float32, device=device)

    # Update nominal parameter paths via Warp
    wp.launch(
        kernel=update_u_kernel,
        dim=(ctrl_n, mj_model.nu),
        inputs=[U, noise, weights, ctrl_low, ctrl_high],
        device=device
    )

    # Track overall optimized control trajectory
    if min_cost < best_cost:
        best_cost = min_cost
        wp.copy(best_U, U)
    else:
        wp.copy(U, best_U)

    print(f"Cost in generation {i+1}/{num_generations}: {min_cost:.4f} (best: {best_cost:.4f})")

### RENDER BEST SOLUTION ###
best_ctrl = best_U.numpy()  # Extract the tracking array to local memory space for plotting

# Reset local single-instance tracker to start config
mujoco.mj_resetData(mj_model, mj_data)
mj_data.qpos[:] = stable_configs[start_idx]
mj_data.ctrl[:] = stable_configs_ctrl[start_idx]
mujoco.mj_forward(mj_model, mj_data)

# Replay optimal parameters within the standard MuJoCo context engine
with mujoco.Renderer(mj_model) as renderer:
    frames = []
    for target_ctrl_idx in range(ctrl_n):
        ctrl_a = np.array(best_ctrl[target_ctrl_idx - 1]) if target_ctrl_idx > 0 else initial_ctrl_np
        ctrl_b = np.array(best_ctrl[target_ctrl_idx])
        for t in range(action_steps):
            alpha = t / (action_steps - 1) if action_steps > 1 else 0.0
            mj_data.ctrl[:] = (1 - alpha) * ctrl_a + alpha * ctrl_b
            mujoco.mj_step(mj_model, mj_data)
            renderer.update_scene(mj_data)
            frames.append(renderer.render())

mediapy.write_video("best_solution.mp4", frames, fps=int(1 / tau_sim))
print("Saved best_solution.mp4")
print(f"Final cost: {best_cost:.4f}")
