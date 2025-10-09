import cma
import time
import h5py
import numpy as np

from explore.env.mujoco_sim import MjSim
from scipy.interpolate import make_interp_spline


start_idx = 1
end_idx = 0
ctrl_n = 7


class BSpline:
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, degree: int=6):
        self.start_time = 0.
        self.end_time = 1.
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.degree = degree
        
    def compute(self, points: np.ndarray, times: np.ndarray):
        points = np.concatenate((self.start_pos.reshape(1, -1), points))
        times = np.array([self.start_time, *times])
        times.sort()
        
        self.splines = []
        for dim in range(points.shape[1]):
            # print(times)
            # print(points[:, dim])
            s = make_interp_spline(times, points[:, dim], k=self.degree)
            self.splines.append(s)
        
    def eval(self, t: float):
        point = np.array([s(t) for s in self.splines])
        return point


new_file_path = "configs/new_rnd_twoFingers.h5"

file = h5py.File(new_file_path, 'r')
stable_configs = file["qpos"]
stable_configs_ctrl = file["ctrl"]

start_ctrl = stable_configs_ctrl[start_idx]
end_ctrl = stable_configs_ctrl[end_idx]
spline = BSpline(start_ctrl, end_ctrl)

sim = MjSim("configs/twoFingers.xml", view=False, verbose=0)

def eval_spline(points: np.ndarray, times: np.ndarray, sim: MjSim, vis: bool=False) -> float:

    if vis:
        sim = MjSim("configs/twoFingers.xml", view=True, verbose=0)
    
    sim.pushConfig(stable_configs[start_idx], stable_configs_ctrl[start_idx])
    tau_action = (spline.start_time - spline.end_time) / 100
    view = tau_action if vis else 0

    spline.compute(points, times)
    ts = np.linspace(spline.start_time, spline.end_time, 100)
    for t in ts:
        ctrl = spline.eval(t)
        sim.step(tau_action, ctrl, view)
    
    final_state = sim.getState()[1]
    e = final_state - stable_configs[end_idx]
    result = e.T @ e
    return result

def eval_splines(candidates: np.ndarray, sim: MjSim) -> list[float]:

    results = []
    for c in candidates:
        v = eval_spline(c[ctrl_n:].reshape(-1, 6), c[:ctrl_n], sim)
        results.append(v)
    
    return results

initial_guess_points = np.array([np.linspace(start_ctrl[dim], end_ctrl[dim], ctrl_n) for dim in range(start_ctrl.shape[0])]).T
initial_guess_times = np.linspace(0., 1., ctrl_n)
initial_guess = np.concatenate((initial_guess_times, initial_guess_points.flatten()))
print("Decision variables: ", initial_guess.shape)

es = cma.CMAEvolutionStrategy(initial_guess, 0.3, {
    "popsize": 32,
    "maxfevals": 10_000,
    "verbose": -1
})

i = 0
while not es.stop():
    i += 1
    print(f"Starting Iter {i}")
    candidates = es.ask()

    results = eval_splines(candidates, sim)
    print(results)

    es.tell(candidates, results)
    es.disp()

print(f"Done! with cost {es.result.fbest}")
r = es.result.xbest
eval_spline(r[ctrl_n:].reshape(-1, 6), r[:ctrl_n], sim, vis=True)
