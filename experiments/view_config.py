import time

from explore.env.mujoco_sim import MjSim
from explore.utils.mj import explain_qpos


mujoco_xml = "configs/mujoco_/unitree_g1/g1_single.xml"

sim = MjSim(mujoco_xml, view=True, verbose=1)
explain_qpos(sim.model)
sim.step(60, view=1.)
