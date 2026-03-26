import h5py
import numpy as np
from tqdm import trange

from explore.env.mujoco_sim import MjSim
from explore.utils.mj import explain_qpos
from bbo_spline_utils import optimize_stand_with_box


mujoco_xml = "configs/mujoco_/unitree_g1/table_box_scene.xml"

sim = MjSim(mujoco_xml, view=False, verbose=1)
explain_qpos(sim.model)


def random_quaternion():
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()

    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])

    return q


def sample_uniform_qpos():
    qpos = np.zeros_like(sim.data.qpos)
    
    qpos[0:2] = np.random.uniform(-1.5, 1.5, (2,))
    qpos[2] = np.random.uniform(1., 1.5)
    qpos[3:7] = [1, 0, 0, 0]

    ranges = sim.model.actuator_ctrlrange[:, 1] - sim.model.actuator_ctrlrange[:, 0]
    qpos[7:-7] = ranges * np.random.uniform(size=ranges.shape) + sim.model.actuator_ctrlrange[:, 0]

    if np.random.uniform() < 0.75:
        qpos[-7:-4] = qpos[0:3] + np.random.randn(3) * 0.2
    else:
        qpos[-7:-5] = np.random.uniform(-1.5, 1.5, (2,))
        qpos[-5] = np.random.uniform(1., 1.5)
    qpos[-4:] = random_quaternion()

    ctrl = qpos[7:-7]
    
    return qpos, ctrl


stable_configs = []
stable_configs_ctrl = []

for i in trange(10000):
    
    while True:
        target_qpos, target_ctrl = sample_uniform_qpos()
        sim.pushConfig(target_qpos, target_ctrl)
        if sim.data.ncon == 0: break
    
    sim.step(10, view=0.)
    # sim.step(1, view=1.)
    start_qpos = np.copy(sim.data.qpos[:])
    start_ctrl = np.copy(sim.data.ctrl[:])

    # Optimize
    end_qpos, end_ctrl = optimize_stand_with_box(sim, start_qpos, start_ctrl, target_qpos)
    sim.pushConfig(end_qpos, end_ctrl)
    # sim.step(10, view=1.)

    stable_configs.append(end_qpos)
    stable_configs_ctrl.append(end_ctrl)

    qpos_array = np.array(stable_configs)
    ctrl_array = np.array(stable_configs_ctrl)

    with h5py.File("./experiments/stable_configs.h5", "w") as f:
        f.create_dataset("qpos", data=qpos_array)
        f.create_dataset("ctrl", data=ctrl_array)
