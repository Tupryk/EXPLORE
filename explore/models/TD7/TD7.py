import time

import numpy as np
import gymnasium as gym

from .agent import *
from .writer import *


class TD7:
    def __init__(self, env: gym.Env, cfg):
        self.env = env

        state_dim = env.observation_space.shape[-1]
        action_dim = env.action_space.shape[-1]
        max_action = float(np.max(env.action_space.high))
        self.agent = TD7_Agent(state_dim, action_dim, max_action, offline=False, hp=cfg)

        self.num_envs = 1

        self.clone_buffer = None
        self.start_time = time.time()
        self.train_time = 0.0
        self.writer = Writer(f"TD7_{cfg.tag}")

        self.t = 0
        self.next_report_t = 0

        #stats for writer
        self.act_hist = np.zeros((20))
        self.obs_hist = np.zeros((20))
        self.hist_count = 0.
        self.ep_reward = np.zeros((self.num_envs))
        self.ep_Q = np.zeros((self.num_envs))
        self.ep_steps = np.zeros((self.num_envs))

    def learn(self, total_timesteps):
        
        t_stop = self.t+total_timesteps
        terminated = False
        state, info = self.env.reset()
        while self.t < t_stop:
            allow_train = self.agent.replay_buffer.size >= self.agent.hp.timesteps_before_training

            # -- print info, log model
            self.report_callback(self.env)

            # -- reset
            if terminated:
                state, info = self.env.reset()

            # -- select action
            if allow_train:
                action = self.agent.select_action(np.atleast_2d(state))
            else:
                action = self.env.action_space.sample().reshape(1, -1)

            # -- single step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            ep_finished = np.logical_or(terminated, truncated)

            # add to replay buffer
            self.agent.replay_buffer.add(state, action, next_state, reward, terminated)  # done)

            # add to histograms
            self.act_hist += np.histogram(action.reshape(-1), bins=20, range=(-1.,1.))[0]/action.size
            self.obs_hist += np.histogram(state.reshape(-1), bins=20, range=(-2.,2.))[0]/state.size
            self.hist_count += 1.

            state = next_state

            # -- train
            self.train_time -= time.time()
            if allow_train and not self.agent.hp.use_checkpoints:
                self.agent.train(self.writer, self.clone_buffer, multi=self.agent.hp.training_steps)
            self.train_time += time.time()

            # -- data for the writer
            self.ep_reward += reward
            self.ep_steps += 1.0

            # -- if truncated, evaluate Q function
            if truncated:
                self.ep_Q[0] += self.agent.evaluate_critic(np.atleast_2d(state)).item()

            # -- write data
            if ep_finished:
                self.writer.add("reward_sum_ep", self.ep_reward[0])
                self.writer.add("steps_ep", self.ep_steps[0])
                self.writer.add("reward_Q_ep", self.ep_reward[0]+self.ep_Q[0])
                self.ep_reward[0] = 0.0
                self.ep_steps[0] = 0.0
                self.ep_Q[0] = 0.0

            # -- increment step
            self.t += self.agent.hp.training_steps

        self.report_callback(self.env)


    def report_callback(self, env):
        if self.t >= self.next_report_t:
            self.next_report_t += self.agent.hp.eval_freq

            self.agent.save(filename=f"data/tmp/{self.writer.tag}.torch")

            print("---------------------------------------")
            print(f"Report after {self.t} training steps, tag {self.agent.hp.tag}")
            time_passed = time.time() - self.start_time
            print(
                f"Total time passed: {round((time_passed)/60.,2)} min(s), training time: {round(self.train_time/60.,2)} min(s) ({round(self.train_time/time_passed*100.,1)}%)"
            )

            print(f'ReplayBuffer size: {self.agent.replay_buffer.size}')
            if self.hist_count>0:
                np.set_printoptions(linewidth=np.inf)
                print(f'action hist [-1.,1.]: {(1000.*self.act_hist/self.hist_count).round().astype(np.int16)}')
                print(f'observ hist [-2.,2.]: {(1000.*self.obs_hist/self.hist_count).round().astype(np.int16)}')
                self.act_hist = np.zeros((20))
                self.obs_hist = np.zeros((20))
                self.hist_count = 0.
                print("Current alpha: ", self.env.schedule_alpha)

            if self.writer is not None:

                # self.writer.add('reward_sum_eval', avg_total_reward)
                # self.writer.add('return_dis_eval', total_return.mean())
                self.writer.write(self.t, verbose=1)

            print("---------------------------------------")
