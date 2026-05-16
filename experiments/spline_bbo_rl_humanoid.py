import cma
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

from explore.env.mujoco_sim import MjSim
from scipy.interpolate import make_interp_spline


start_idx = 1
end_idx = 12217
ctrl_n = 10

new_file_path = "configs/stable/humanoid_box_grasps.h5"
mujoco_xml = "configs/mujoco_/unitree_g1/table_box_scene.xml"

class BSpline:
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, degree: int=6):
        self.start_time = 0.
        self.end_time = 2.
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.degree = degree
        
    def compute(self, points: np.ndarray):
        points = np.concatenate((self.start_pos.reshape(1, -1), points))
        times = np.linspace(self.start_time, self.end_time, points.shape[0])
        
        self.splines = []
        for dim in range(points.shape[1]):
            s = make_interp_spline(times, points[:, dim], k=self.degree)
            self.splines.append(s)
        
    def eval(self, t: float):
        point = np.array([s(t) for s in self.splines])
        return point

file = h5py.File(new_file_path, 'r')
stable_configs = file["qpos"]
stable_configs_ctrl = file["ctrl"]

start_ctrl = stable_configs_ctrl[start_idx]
end_ctrl = stable_configs_ctrl[end_idx]
spline = BSpline(start_ctrl, end_ctrl)

sim = MjSim(mujoco_xml, view=False, verbose=0)

def eval_spline(points: np.ndarray, sim: MjSim, vis: bool=False) -> float:

    sim.pushConfig(stable_configs[start_idx], stable_configs_ctrl[start_idx])
    tau_action = (spline.end_time - spline.start_time) / 100
    view = tau_action if vis else 0

    spline.compute(points)
    ts = np.linspace(spline.start_time, spline.end_time, 100)
    for t in ts[1:]:
        ctrl = spline.eval(t)
        sim.step(tau_action, ctrl, view)
    
    final_state = sim.getState()[1]
    e = (final_state - stable_configs[end_idx])
    result = e.T @ e
    return result

def eval_splines(candidates: np.ndarray, sim: MjSim) -> list[float]:

    results = []
    for c in candidates:
        v = eval_spline(c.reshape(-1, start_ctrl.shape[0]), sim, vis=True)
        results.append(v)
    
    return results

initial_guess_points = np.array([np.linspace(start_ctrl[dim], end_ctrl[dim], ctrl_n) for dim in range(start_ctrl.shape[0])]).T
# initial_guess_times = np.linspace(0., 1., ctrl_n)
# initial_guess = np.concatenate((initial_guess_times, initial_guess_points.flatten()))
initial_guess = initial_guess_points.flatten()
initial_guess += np.random.randn(initial_guess.shape[0]) * .01
print("Decision variables: ", initial_guess.shape)

es = cma.CMAEvolutionStrategy(initial_guess, 1.5, {
    "popsize": 2,
    "maxfevals": 1,
    "verbose": -1
})

while not es.stop():
    candidates = es.ask()

    results = eval_splines(candidates, sim)

    es.tell(candidates, results)
    es.disp()

print(f"Done! with cost {es.result.fbest}")

r = es.result.xbest
points = np.concatenate((spline.start_pos.reshape(1, -1), r.reshape(-1, start_ctrl.shape[0])))
times = np.linspace(spline.start_time, spline.end_time, points.shape[0])

spline.compute(points)

del sim
sim = MjSim(mujoco_xml, view=True, verbose=0, joints_are_same_as_ctrl=True)
sim.pushConfig(stable_configs[start_idx])
time.sleep(3)
sim.pushConfig(stable_configs[end_idx])
time.sleep(3)
eval_spline(initial_guess.reshape(-1, start_ctrl.shape[0]), sim, vis=True)
time.sleep(3)
