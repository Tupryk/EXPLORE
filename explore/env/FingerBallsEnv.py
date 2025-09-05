import gym
import time
import numpy as np
import robotic as ry
from gym import spaces
from omegaconf import DictConfig

from explore.env.MujocoSim import MjSim
from explore.datasets.rnd_configs import RndConfigs


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
            data_path = f"./data/results/{cfg.start_config_idx}_{cfg.target_config_idx}.npy"
            trajectory_data = np.load(data_path, allow_pickle=True)
            self.guiding_path = [t.action for t in trajectory_data]
            self.max_steps = int(len(self.guiding_path) * 1.5)

    def getState(self) -> np.ndarray:
        self.state = self.sim.getState()[0][self.relevant_frames_idxs, :3].flatten()
        if self.use_vel:
            self.state = np.concatenate((self.sim.getState()[1], self.state))
        return self.state

    def reset(self, *, seed=None, options=None):
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

    def render(self, mode="human"):
        print("Iter: ", self.iter, "Reward: ", self.reward)
        if self.iter == 0:
            self.Ds.C.view(True)
        elif self.iter >= self.max_steps-1:
            self.Ds.C.view(True)
        elif (self.iter+1) % 5 == 0:
            self.Ds.C.view(True)
        else:
            self.Ds.C.view()
            time.sleep(.1)

    def get_config(self) -> ry.Config:
        return self.Ds.C
