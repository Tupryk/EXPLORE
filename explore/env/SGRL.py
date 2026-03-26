import math
import numpy as np
from explore.datasets.workspace import Workspace
from explore.models.TD7.buffer import CloneBuffer
from data_tools.video_gen import VideoGenerator
from sim_wrappers import MujocoGym
import matplotlib.pyplot as plt

class TrajCloneBuffer:
    observation: np.array
    action: np.array
    Return: np.array
    Gamma: np.array
    next_observation: np.array

class SGRL:
    def __init__(self, W: Workspace):
        self.W = W

    def update_environment(self, env: MujocoGym, t, td7, cfg):
        # phase
        alpha = (t+cfg.T_block) / (cfg.T_schedule+1)
        alpha = min(alpha, 1.)
        env.cfg.time_limit = 2.*alpha*cfg.nominal_time_limit

        # sample new start/goals for this phase
        n = env.starts_goals_counter
        if n == 0:
            n = cfg.T_block * env.num_scenes / (0.2 * cfg.nominal_time_limit/env.cfg.tau_step) # heuristic: enough goals if episodes would last only 20% of nominal time limit
            if n>1e6:
                n = 1e6
            n = int(n)
        else:
            n *= 2 # heuristic: twice as many goals as needed in the previous block
        # n = int(alpha*100_000)
        starts, goals = self.W.getStartsGoals(n=n, alpha=alpha, mode=cfg.start_goal_mode)
        env.set_starts_goals(starts, goals)

        # enable BC?
        if t < cfg.T_cloning:
            traj_clone_buffer = self.get_TrajCloneBuffer(alpha)
            td7.clone_buffer = CloneBuffer(traj_clone_buffer.observation,
                                                traj_clone_buffer.action)
            cloneInfo = td7.clone_buffer.state.shape[0]
        else:
            td7.clone_buffer = None
            cloneInfo = 0

        print(f'=== alpha: {alpha}, mode: {cfg.start_goal_mode}, starts/goals: {env.starts_q.shape[0]}, clone: {cloneInfo}, time_limit: {env.cfg.time_limit}, T_block: {cfg.T_block}')


    def evaluate_policy(self, env: MujocoGym, td7, num_episodes=20, view=False, make_video=False, collect_histograms=False, verbose=2):
        
        num_scenes = env.num_scenes
        F_goals = []
        for f in env.sim.C.getFrames():
            if 'obj_goal' in f.name:
                F_goals.append(f)

        if view or make_video:
            env.sim.view_speed = .5
            # env.verbose = 3
        if make_video:
            env.sim.C.view(False)
            env.sim.view_speed = .5
            env.sim.C.get_viewer().getRgb()
            env.sim.save_images = True
            V = VideoGenerator(framerate=50)
        if collect_histograms:
            actions = []
            obs = []

        n_succ = 0.
        steps_succ = 0.
        steps_succ_sqr = 0.
        num_ep = 0
        ep_reward = np.zeros((num_scenes))
        ep_steps = np.zeros((num_scenes))

        while num_ep<num_episodes:
            state, info = env.auto_reset()

            # reposition goal frames for illustration
            goal_glob = env.goal_feat.copy()
            for th in range(len(F_goals)):
                F_goals[th].setRelativePosition(goal_glob[th])

            action = td7.agent.select_action(state, False, use_exploration=False)
            state, reward, terminated, truncated, info = env.step(action)
            ep_finished = np.logical_or(terminated, truncated)
            ep_reward += reward
            ep_steps += 1.

            for k in range(num_scenes):
                if ep_finished[k]:
                    if verbose>1:
                        print(f'ep {num_ep} reward: {ep_reward[k]} steps: {ep_steps[k]}')
                    num_ep += 1
                    if ep_reward[k]>0.:
                        n_succ += 1
                    steps_succ += ep_steps[k]
                    steps_succ_sqr += ep_steps[k]*ep_steps[k]
                    
                    if make_video and ep_reward[k]>0. and ep_steps[k]>=5:
                        env.sim.C.view(False, 'END', offscreen=True)
                        rgb = env.sim.C.get_viewer().getRgb()
                        env.sim.saved_images.append([rgb] * 10)
                        V.add(env.sim.saved_images)

                    env.sim.saved_images = []
                    ep_reward[k] = 0.
                    ep_steps[k] = 0.

            if collect_histograms:
                actions.append(action)
                obs.append(state)

        if n_succ>0:
            steps_succ /= n_succ
            steps_succ_sqr /= n_succ
            len_std = math.sqrt(steps_succ_sqr - steps_succ*steps_succ)
        else:
            len_std = 0.
        print(f'== #ep: {num_episodes}  succ_rate: {n_succ/num_episodes}  avg succ length: {steps_succ}+={len_std}')

        if collect_histograms:
            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

            n = len(actions)*num_scenes
            actions = np.stack(actions).reshape(n,-1)
            ax0.hist(actions, 21, histtype='step', stacked=True, fill=False)
            ax0.set_title('actions')
            ax0.legend(labels=[f'{i}' for i in range(actions.shape[1])], prop={'size': 7})

            obs = np.stack(obs).reshape(n,-1)
            ax1.hist(obs, 21, histtype='step', stacked=True, fill=False)
            ax1.set_title('observations')
            ax1.legend(labels=[f'{i}' for i in range(obs.shape[1])], prop={'size': 7})

            fig.tight_layout()
            plt.show()

        return n_succ/num_episodes

    def get_TrajCloneBuffer(self, phase) -> TrajCloneBuffer:
        if not hasattr(self.W, 'traj_data'):
            self.W.load_TrajectoryData()

        S = self.W.traj_data.state.shape
        T = S[1]
        if phase<=1.:
            T0 = int((1.-phase)*T)
        else:
            T0 = 0

        clone_buffer = TrajCloneBuffer()
        time_to_go = self.W.traj_data.time_to_go[:,T0:,:]
        o_dim = self.W.traj_data.observation.shape[2]

        clone_buffer.observation = self.W.traj_data.observation[:,T0:,:]
        clone_buffer.action = self.W.traj_data.action[:,T0:,:]
        clone_buffer.Return = np.empty((S[0], T-T0, 1))
        clone_buffer.Gamma = np.zeros((S[0], T-T0, 1)) # ZERO!
        clone_buffer.next_observation = np.empty((S[0], T-T0, o_dim))

        for i in range(S[0]):
            Ti = clone_buffer.observation.shape[1]
            
            # set next_state_obs to final state
            final_obs = clone_buffer.observation[i,-1]
            clone_buffer.next_observation[i,:] = final_obs

            for t in range(Ti):
                clone_buffer.Return[i,t,0] = 1. - self.W.gym.cfg.cost_const * time_to_go[i,t,0]

        for key, x in clone_buffer.__dict__.items():
            clone_buffer.__dict__[key] = x.reshape(-1, x.shape[-1])

        print('-- created clone_buffer:')
        for key, x in clone_buffer.__dict__.items():
            print('  ', key, x.shape)

        return clone_buffer  
