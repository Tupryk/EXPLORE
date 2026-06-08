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
    
    "sim_interface": {    
        "parallel_sims": 10,
        "xml_path": "configs/mujoco_/fingerRamp.xml",
        "verbose": 1,
        "geoms_in_cost": [
            "l_fing", "obj"
        ],
        "geoms_in_cost_weights": [
            1., 1., 1.,
            4., 4., 4.
        ]
    }
})

env = StableConfigsEnv(cfg)
env.reset()

actions = np.random.random((10, 3))
env.step(actions)
print(env.getState())
