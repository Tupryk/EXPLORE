import time
import pickle
import numpy as np
import robotic as ry
import gymnasium as gym
from gymnasium import spaces
from omegaconf import DictConfig

from explore.env.MujocoSim import MjSim
from explore.datasets.rnd_configs import RndConfigs


def getFeasibleTransitionPairs(
        trees: list[list[dict]], feasible_thresh: float=5e-2
        )-> tuple[list[tuple[int, int]], list[int]]:
    
    top_nodes = []
    min_costs = []
    
    tree_count = len(trees)
    for i in range(tree_count):

        tree_min_costs = [float("inf") for _ in range(tree_count)]
        tree_top_nodes = [-1 for _ in range(tree_count)]
        
        for n, node in enumerate(trees[i]):
            for j in range(tree_count):
                if node["costs"][j] < tree_min_costs[j]:
                    tree_min_costs[j] = node["costs"][j]
                    tree_top_nodes[j] = n
        
        top_nodes.append(tree_top_nodes)
        min_costs.append(tree_min_costs)
    
    feasible_pairs = []
    end_nodes = []
    for start_idx in range(tree_count):
        for end_idx in range(tree_count):
            if start_idx != end_idx and min_costs[start_idx][end_idx] <= feasible_thresh:
                feasible_pairs.append((start_idx, end_idx))
                end_nodes.append(top_nodes[start_idx][end_idx])

    return feasible_pairs, end_nodes

class FingerBallsEnv(gym.Env):

    def __init__(self, cfg: DictConfig):
        
        super().__init__()

        state_n = 22 if cfg.use_vel else 9
        self.observation_space = spaces.Box(low=-2., high=2., shape=(state_n,), dtype=np.float32)
        self.action_space = spaces.Box(low=-cfg.stepsize, high=cfg.stepsize, shape=(6,), dtype=np.float32)

        self.max_steps = cfg.max_steps
        self.stepsize = cfg.stepsize
        self.tau = cfg.tau
        self.actions_noise_sigma = cfg.actions_noise_sigma
        self.use_vel = cfg.use_vel
        self.guiding = cfg.guiding
        self.reward = None

        # Setup sim
        self.start_config_idx = cfg.start_config_idx
        self.reset()
        
        # Get target cost to compute cost
        De = RndConfigs("configs/twoFingers.g", "configs/rnd_twoFingers.h5")
        De.set_config(cfg.target_config_idx)
        sim_ = MjSim(open("configs/twoFingers.xml", 'r').read(), De.C, view=False, verbose=0)
        self.target_state = sim_.getState()[0][self.relevant_frames_idxs, :3].flatten()
        del sim_

        self.guiding_path = []
        if self.guiding:
            
            trees = []
            for i in range(100):
                data_path = f"data/{cfg.dataset}/{i}_{cfg.dataset}.pkl"
                with open(data_path, "rb") as f:
                    tree: list[list[dict]] = pickle.load(f)
                    trees.append(tree)

            cfg_pair = (cfg.start_config_idx, cfg.target_config_idx)
            pairs, end_nodes = getFeasibleTransitionPairs(trees)
            if not (cfg_pair in pairs):
                raise Exception(f"Config pair not feasible! Feasible pairs: {pairs}")

            pair_idx = pairs.index(cfg_pair)
            tree = trees[cfg.start_config_idx]
            node = tree[end_nodes[pair_idx]]

            self.guiding_path = []
            while True:
                self.guiding_path.append(node["action"])
                if node["parent"] == -1: break
                node = tree[node["parent"]]
            
            self.guiding_path.reverse()
            assert self.guiding_path[0] == tree[0]["action"]

            self.max_steps = int(len(self.guiding_path) * 1.5)

    def getState(self) -> np.ndarray:
        self.state = self.sim.getState()[0][self.relevant_frames_idxs, :3].flatten()
        if self.use_vel:
            self.state = np.concatenate((self.sim.getState()[1], self.state))
        return self.state

    def reset(self, *, seed: int=None, options: dict=None):
        super().reset(seed=seed)

        self.Ds = RndConfigs("configs/twoFingers.g", "configs/rnd_twoFingers.h5")
        self.Ds.set_config(self.start_config_idx)

        self.sim = MjSim(open("configs/twoFingers.xml", 'r').read(), self.Ds.C, view=False, verbose=0)

        self.iter = 0

        # Relevant frame indices for cost computing
        relevant_frames = ["obj", "l_fing", "r_fing"]
        self.relevant_frames_idxs = [self.Ds.C.getFrameNames().index(rf) for rf in relevant_frames]

        return self.getState(), {}

    def step(self, action: np.ndarray):
        
        if self.actions_noise_sigma != -1:
            action += np.random.randn(action.shape[-1]) * self.actions_noise_sigma
        action = self.sim.getState()[1][:6] + action
        
        time_offset = self.tau * self.iter
        self.sim.resetSplineRef(time_offset)
        self.sim.setSplineRef(action.reshape(1,-1), [self.tau], append=False)
        self.sim.step([], self.tau, ry.ControlMode.spline, .0)
        
        self.iter += 1

        self.getState()

        goal_cost_scaler = .1 if self.iter < len(self.guiding_path) else 1.
        if self.use_vel:
            self.reward = -goal_cost_scaler * np.linalg.norm(self.state[13:] - self.target_state)
        else:
            self.reward = -goal_cost_scaler * np.linalg.norm(self.state - self.target_state)
        
        if self.guiding and self.iter < len(self.guiding_path):
            guiding_step = self.guiding_path[self.iter].reshape(-1)
            self.reward += -1. * np.linalg.norm(action - guiding_step)

        self.reward = -1. * (self.reward**2)

        truncated = self.iter >= self.max_steps
        terminated = truncated
        info = {}

        return self.state, self.reward, terminated, truncated, info

    def render(self, mode: str="none") -> np.ndarray:

        print("Iter: ", self.iter, "Reward: ", self.reward)
        
        if mode == "human":
            if self.iter == 0:
                self.Ds.C.view(True)
            elif self.iter >= self.max_steps-1:
                self.Ds.C.view(True)
            elif (self.iter+1) % 5 == 0:
                self.Ds.C.view(True)
            else:
                self.Ds.C.view()
                time.sleep(.1)
        
        elif mode == "human0":
            if self.iter == 0:
                self.Ds.C.view(True)
            elif self.iter >= self.max_steps-1:
                self.Ds.C.view(True)
            else:
                self.Ds.C.view()
                time.sleep(.1)
        
        elif mode == "none":
            pass

        else:
            raise Exception(f"Render mode '{mode}' not implemented yet!")

        f = self.Ds.C.addFrame("camera")
        f.setPose([-1.68869923, 1.57007084, 0.8082533, 0.27602618, -0.28676833, 0.70882871, -0.58235327])
        f.setAttributes({
            'focalLength': 2.,
            'width': 200,
            'height': 200
        })
        cam = ry.CameraView(self.Ds.C)
        cam.setCamera(self.Ds.C.getFrame("camera"))
        img, _ = cam.computeImageAndDepth(self.Ds.C)

        return img

    def get_config(self) -> ry.Config:
        return self.Ds.C
