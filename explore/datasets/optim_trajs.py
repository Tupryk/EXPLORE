import jax
import math
import h5py
import torch
import mujoco
import warp as wp
import numpy as np
from tqdm import tqdm
import mujoco_warp as mjw
from evosax.algorithms import CMA_ES


class OptimPair:
    def __init__(self):
        self.nworld = 256

        new_file_path = "configs/stable/humanoid_box_grasps.h5"
        mujoco_xml = "configs/mujoco_/unitree_g1/table_box_scene.xml"

        self.start_idx = 1
        end_idx = 12217
        self.ctrl_n = 10
        tau_action = 0.1
        tau_sim = 0.005
        self.action_steps = math.ceil(tau_action / tau_sim)

        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_xml)
        self.mj_model.opt.timestep = tau_sim
        self.mj_model.opt.ccd_iterations = 200
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.put_data(self.mj_model, self.mj_data, nworld=self.nworld, nconmax=250, njmax=250)

        self.ctrl_torch = torch.zeros((self.nworld, self.mj_model.nu), device="cuda")
        self.ctrl_wp = wp.from_torch(self.ctrl_torch)

        file = h5py.File(new_file_path, 'r')
        self.stable_configs = file["qpos"]
        self.stable_configs_ctrl = file["ctrl"]

        self.data.qpos = wp.array(np.tile(self.stable_configs[self.start_idx], (self.nworld, 1)), dtype=wp.float32)
        self.data.ctrl = wp.array(np.tile(self.stable_configs_ctrl[self.start_idx], (self.nworld, 1)), dtype=wp.float32)
        mjw.forward(self.model, self.data)

    def eval_ctrl_sequences(self, ctrl_sequences: jax.Array):
        for target_ctrl_idx in tqdm(range(1, len(ctrl_sequence))):
            
            for t in range(self.action_steps):

                alpha = t / (self.action_steps - 1)

                self.ctrl_torch[:] = (1 - alpha) * ctrl_sequence[target_ctrl_idx-1] + alpha * ctrl_sequence[target_ctrl_idx]

                self.data.ctrl.assign(self.ctrl_wp)
                mjw.step(self.model, self.data)


    def run(self):
        es = CMA_ES(population_size=32, solution=dummy_solution)
        params = es.default_params

        # Initialize state
        key = jax.random.key(0)
        state = es.init(key, dummy_solution, params)

        # Ask-Eval-Tell loop
        for i in range(num_generations):
            key, key_ask, key_eval = jax.random.split(key, 3)

            # Generate a set of candidate solutions to evaluate
            population, state = es.ask(key_ask, state, params)

            # Evaluate the fitness of the population
            fitness = ...

            # Update the evolution strategy
            state, metrics = es.tell(key, population, fitness, state, params)

        # Get best solution
        state.best_solution, state.best_fitness

        ctrl_sequence = [torch.tensor(self.stable_configs_ctrl[self.start_idx], device="cuda")]
        for _ in range(self.ctrl_n-1):
            ctrl_target = torch.rand((self.nworld, self.mj_model.nu), device="cuda") * 2 - 1
            ctrl_sequence.append(ctrl_target)
