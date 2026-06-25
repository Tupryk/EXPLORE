import numpy as np
from omegaconf import OmegaConf

from explore.env.StaGE_env import StaGEEnv


cfg = OmegaConf.create({
    "min_cost": 0.01,
    "max_steps": 64,
    "max_manifold_size": 100,
    "sparse_reward": True,
    "verbose": 1,

    "tau_action": 0.05,
    "stepsize": 0.05,
    
    "stable_configs_path": "configs/stable/double_sphere.h5",

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
        "parallel_sims": 100,
        "xml_path": "configs/mujoco_/doubleSphere.xml",
        "tau_sim": 0.005,
        "verbose": 1,

        "camera": "fixed_cam",
        "render_w": 640,
        "render_h": 480
    }
})

env = StaGEEnv(cfg)
states, info = env.reset()
for s in states:
    print("q: ", s[:3])
    print("q_dot: ", s[3:6])
    print("q_obj_dot: ", s[6:12])
    print("r_q: ", s[12:15])
    print("P: ", s[15:18])
    print("G* - G: ", s[18:21])
print(info)
