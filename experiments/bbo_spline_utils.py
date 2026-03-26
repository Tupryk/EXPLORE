import cma
import numpy as np

from explore.env.mujoco_sim import MjSim
from scipy.interpolate import make_interp_spline


class BSpline:
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, degree: int=2):
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


def eval_spline(points: np.ndarray, sim: MjSim, start_qpos: np.ndarray, start_ctrl: np.ndarray, target_qpos: np.ndarray, vis: bool=False) -> float:

    spline = BSpline(start_ctrl, points[-1])
    sim.pushConfig(start_qpos, start_ctrl)
    tau_action = (spline.end_time - spline.start_time) / 100
    view = tau_action if vis else 0

    spline.compute(points)
    ts = np.linspace(spline.start_time, spline.end_time, 100)
    for t in ts[1:]:
        ctrl = spline.eval(t)
        sim.step(tau_action, ctrl, view)

    e = sim.data.qpos - target_qpos
    
    result = 10 * e[2]**2 + 100 * e[-5]**2
    result += sim.data.qvel.T @ sim.data.qvel
    return result


def eval_splines(candidates: np.ndarray, sim: MjSim, start_qpos: np.ndarray, start_ctrl: np.ndarray, target_qpos: np.ndarray) -> list[float]:

    results = []
    for i, c in enumerate(candidates):
        v = eval_spline(c.reshape(-1, start_ctrl.shape[0]), sim, start_qpos, start_ctrl, target_qpos, vis=False)
        results.append(v)
    
    return results


def optimize_stand_with_box(
        sim: MjSim, start_qpos: np.ndarray, start_ctrl: np.ndarray, target_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    knots = 5
    initial_guess_points = np.array([np.linspace(start_ctrl[dim], start_ctrl[dim], knots) for dim in range(start_ctrl.shape[0])]).T
    initial_guess = initial_guess_points.flatten()
    print("Decision variables: ", initial_guess.shape)

    es = cma.CMAEvolutionStrategy(initial_guess, 0.5, {
        "popsize": 128,
        "maxfevals": 10000,
        "verbose": -1
    })

    while not es.stop():
        candidates = es.ask()

        results = eval_splines(candidates, sim, start_qpos, start_ctrl, target_qpos)

        es.tell(candidates, results)
        es.disp()

    print(f"Done! with cost {es.result.fbest}")

    points = es.result.xbest.reshape(knots, -1)
    spline = BSpline(start_ctrl, points[-1])
    sim.pushConfig(start_qpos, start_ctrl)
    tau_action = (spline.end_time - spline.start_time) / 100

    spline.compute(points)
    ts = np.linspace(spline.start_time, spline.end_time, 100)
    for t in ts[1:]:
        ctrl = spline.eval(t)
        sim.step(tau_action, ctrl)
    
    return np.copy(sim.data.qpos[:]), np.copy(sim.data.ctrl[:])
