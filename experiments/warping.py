import jax
import math
import h5py
import mujoco
import mediapy
import warp as wp
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import mujoco_warp as mjw

### SETUP ###
NWORLD = 512
new_file_path = "configs/stable/humanoid_box_grasps.h5"
mujoco_xml = "configs/mujoco_/unitree_g1/table_box_scene.xml"
start_idx = 1
end_idx = 12217
ctrl_n = 4
tau_action = 0.5
tau_sim = 0.005
action_steps = math.ceil(tau_action / tau_sim)
num_generations = 25
noise_std = .5
lam = 5.  # MPPI temperature

geoms_in_cost_names = [
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_rubber_hand_0", "right_rubber_hand_0",
    "box_marker_0", "box_marker_1", "box_marker_2"
]
geom_weights = jnp.array([
    1., 1.,     # Feet
    2., 2.,     # Hands
    4., 4., 4.  # Box
])

mj_model = mujoco.MjModel.from_xml_path(mujoco_xml)
mj_model.opt.timestep = tau_sim
mj_model.opt.ccd_iterations = 200
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetData(mj_model, mj_data)
geom_indices = jnp.array([mj_model.geom(name).id for name in geoms_in_cost_names])

model = mjw.put_model(mj_model)
data = mjw.put_data(mj_model, mj_data, nworld=NWORLD, nconmax=250, njmax=250)

file = h5py.File(new_file_path, 'r')
stable_configs = file["qpos"]
stable_configs_ctrl = file["ctrl"]
initial_ctrl = jnp.array(stable_configs_ctrl[start_idx])

# Get target xpos
mj_data.qpos[:] = stable_configs[end_idx]
mujoco.mj_forward(mj_model, mj_data)
target_geom_xpos = jnp.array(mj_data.geom_xpos[geom_indices])  # (n_geoms, 3)

key = jax.random.key(0)

# Nominal control sequence: (ctrl_n, nu)
U = jnp.tile(initial_ctrl, (ctrl_n, 1))
ctrl_low  = jnp.array(mj_model.actuator_ctrlrange[:, 0])
ctrl_high = jnp.array(mj_model.actuator_ctrlrange[:, 1])

### MPPI LOOP ###
best_cost = jnp.inf
best_U = U

qpos_init = wp.array(np.tile(stable_configs[start_idx], (NWORLD, 1)), dtype=wp.float32)
ctrl_init = wp.array(np.tile(stable_configs_ctrl[start_idx], (NWORLD, 1)), dtype=wp.float32)
initial_ctrl_tiled = jnp.tile(initial_ctrl, (NWORLD, 1))

@jax.jit
def compute_ctrls(ctrl_a, ctrl_b):
    return ctrl_a[None] + alphas[:, None, None] * (ctrl_b - ctrl_a)[None]  # (action_steps, NWORLD, nu)

@jax.jit
def compute_cost(geom_xpos, target_geom_xpos, geom_weights, geom_indices):
    e = geom_xpos[:, geom_indices, :] - target_geom_xpos
    e = e * geom_weights[None, :, None]
    return jnp.sum(e ** 2, axis=(1, 2))

@jax.jit
def update_U(U, weights, noise):
    return U + jnp.einsum("i,ijk->jk", weights, noise)

alphas = jnp.linspace(0, 1, action_steps)  # precompute once outside the generation loop
for i in tqdm(range(num_generations), total=num_generations):
    key, subkey = jax.random.split(key)

    # reset all worlds to start config
    wp.copy(data.qpos, qpos_init)
    wp.copy(data.ctrl, ctrl_init)
    mjw.forward(model, data)

    # sample noise: (NWORLD, ctrl_n, nu)
    noise = jax.random.normal(subkey, shape=(NWORLD, ctrl_n, mj_model.nu)) * noise_std

    # perturbed sequences: (NWORLD, ctrl_n, nu)
    ctrl_sequences = U[None, :, :] + noise

    # rollout

    for target_ctrl_idx in range(ctrl_n):
        ctrl_a = ctrl_sequences[:, target_ctrl_idx - 1, :] if target_ctrl_idx > 0 else initial_ctrl_tiled
        ctrl_b = ctrl_sequences[:, target_ctrl_idx, :]
        
        ctrls = compute_ctrls(ctrl_a, ctrl_b)
        
        for t in range(action_steps):
            data.ctrl.assign(wp.from_jax(ctrls[t]))
            mjw.step(model, data)

    # compute cost
    geom_xpos = wp.to_jax(data.geom_xpos)                 # (NWORLD, ngeom, 3)
    cost = compute_cost(geom_xpos, target_geom_xpos, geom_weights, geom_indices)

    # MPPI weight update
    beta = jnp.min(cost)
    weights = jnp.exp(-(cost - beta) / lam)  # (NWORLD,)
    weights = weights / jnp.sum(weights)      # normalize

    # update nominal: weighted average of sampled noise
    U = update_U(U, weights, noise)
    U = jnp.clip(U, ctrl_low, ctrl_high)

    # track best overall
    if cost.min() < best_cost:
        best_cost = cost.min()
        best_U = U

    print(f"Cost in generation {i+1}/{num_generations}: {cost.min():.4f} (best: {best_cost:.4f})")

### RENDER BEST SOLUTION ###
best_ctrl = best_U  # (ctrl_n, nu) — best overall nominal sequence

# reset to start config
mujoco.mj_resetData(mj_model, mj_data)
mj_data.qpos[:] = stable_configs[start_idx]
mj_data.ctrl[:] = stable_configs_ctrl[start_idx]
mujoco.mj_forward(mj_model, mj_data)

# replay with renderer
with mujoco.Renderer(mj_model) as renderer:
    frames = []
    for target_ctrl_idx in range(ctrl_n):
        ctrl_a = np.array(best_ctrl[target_ctrl_idx - 1]) if target_ctrl_idx > 0 else np.array(initial_ctrl)
        ctrl_b = np.array(best_ctrl[target_ctrl_idx])
        for t in range(action_steps):
            alpha = t / (action_steps - 1)
            mj_data.ctrl[:] = (1 - alpha) * ctrl_a + alpha * ctrl_b
            mujoco.mj_step(mj_model, mj_data)
            renderer.update_scene(mj_data)
            frames.append(renderer.render())

mediapy.write_video("best_solution.mp4", frames, fps=int(1 / tau_sim))
print("Saved best_solution.mp4")
print(f"Final cost: {best_cost:.4f}")
