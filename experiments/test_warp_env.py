from omegaconf import OmegaConf
from explore.env.mujoco_warp_sim import MjSim

cfg = OmegaConf.create({
    "parallel_sims": 10,
    "xml_path": "configs/mujoco_/fingerRamp.xml",
    "verbose": 1,
    "geoms_in_cost": [
        "FL", "FR", "RL", "RR",
        "box_marker0", "box_marker1", "box_marker2"
    ],
    "geoms_in_cost_weights": [
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        4., 4., 4., 4., 4., 4., 4., 4., 4.
    ],
})

sim = MjSim(cfg)
print(sim.getCustomState())
print(sim.getCustomStateScaled())

sim.step(10, sim.data.ctrl.numpy().copy())
print(sim.getCustomState())
print(sim.getCustomStateScaled())

