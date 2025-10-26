import time

from explore.env.mujoco_sim import MjSim


mujoco_xml = "configs/fingerRamp.xml"

sim = MjSim(mujoco_xml, view=True, verbose=1)
sim.step(60, view=1.)
