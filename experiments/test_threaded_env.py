import numpy as np
from omegaconf import OmegaConf

from explore.env.mujoco_threaded_sim import MjSim

cfg = OmegaConf.create({
    "parallel_sims": 10,
    "xml_path": "configs/mujoco_/fingerRamp.xml",
    "verbose": 1
})

sim = MjSim(cfg)
print(sim.getState())

sim.step(10, sim.data_ctrl.copy())
print(sim.getState())
