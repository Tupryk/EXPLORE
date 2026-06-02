import time
from omegaconf import OmegaConf
from explore.env.mujoco_sim import MjSim

cfg = OmegaConf.create({
    "xml_path": "configs/mujoco_/unitree_go2/table_box_scene.xml",
    "verbose": 1,
    "geoms_in_cost": [
        "FL", "FR", "RL", "RR",
        "box_marker0", "box_marker1", "box_marker2"
    ],
    "geoms_in_cost_weights": [
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        4., 4., 4., 4., 4., 4., 4., 4., 4.
    ],
    "vels_geoms_in_cost": True
})

sim = MjSim(cfg, view=True)
print(sim.getCustomState())
print(sim.getCustomStateScaled())

sim.step(10, view=1.)
print(sim.getCustomState())
print(sim.getCustomStateScaled())
