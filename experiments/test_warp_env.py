import numpy as np
from omegaconf import OmegaConf

from explore.env.mujoco_warp_sim import MjSim
from explore.env.SGRL_env import StableConfigsEnv

# cfg = OmegaConf.create({
#     "parallel_sims": 10,
#     "xml_path": "configs/mujoco_/fingerRamp.xml",
#     "verbose": 1,
#     "geoms_in_cost": [
#         "l_fing", "obj"
#     ],
#     "geoms_in_cost_weights": [
#         1., 1., 1.,
#         4., 4., 4.
#     ],
# })

# sim = MjSim(cfg)
# print(sim.getCustomState())
# print(sim.getCustomStateScaled())

# sim.step(10, sim.data.ctrl.numpy().copy())
# print(sim.getCustomState())
# print(sim.getCustomStateScaled())

cfg = OmegaConf.create({
    "min_cost": 0.01,
    "max_steps": 64,
    "use_schedule": True,
    "schedule_alpha_end_step": 100000,
    "schedule_alpha_block": 5000,
    "use_csrl": True,
    "verbose": 1,

    "tau_action": 0.2,
    "stepsize": 0.1,
    
    "stable_configs_path": "configs/stable/fingerRamp_onRamp.h5",
    
    "q": [0,3],
    "q_dot": [0,3],
    "q_obj_dot": 3,
    "P": ["obj"],
    "G": ["obj"],
    
    "obs_pos_scale": 1.0,
    "obs_vel_scale": 0.05,
    "obs_ref_err_scale": 20.0,

    "q_weight": 0.2,

    "sim_interface": {    
        "parallel_sims": 10,
        "xml_path": "configs/mujoco_/fingerRamp.xml",
        "verbose": 1
    }
})

env = StableConfigsEnv(cfg)
states, _ = env.reset(options={"alpha": 1.0})
print("After Reset:")
print("q: ", states[0, :3])
print("q_dot: ", states[0, 3:6])
print("q_obj_dot: ", states[0, 6:12])
print("r_q: ", states[0, 12:15])
print("P: ", states[0, 15:18])
print("G* - G: ", states[0, 18:21])

actions = np.random.uniform(-env.stepsize, env.stepsize, (10, 3))
next_states, rewards, terminated, truncated, info = env.step(actions)
print("After First Action:")
print("q: ", next_states[0, :3])
print("q_dot: ", next_states[0, 3:6])
print("q_obj_dot: ", next_states[0, 6:12])
print("r_q: ", next_states[0, 12:15])
print("P: ", next_states[0, 15:18])
print("G* - G: ", next_states[0, 18:21])
