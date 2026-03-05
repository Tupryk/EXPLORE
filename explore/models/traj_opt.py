import time

import jax

jax.config.update("jax_compilation_cache_dir", "tmp/jax_cache") # hack to avoid requiring sudo access to /tmp folder

import jax.numpy as jnp

import os

os.environ['MUJOCO_GL'] = 'egl'   # or 'osmesa'

import mujoco



import mujoco.viewer

import numpy as np

from mujoco import mjx

import copy

from hydrax.alg_base import Trajectory, SamplingBasedController

import tqdm

from functools import partial

from pathlib import Path

import matplotlib.pyplot as plt

import imageio





class TrajectoryOptimizer:



    def __init__(

        self,

        name: str,

        controller: SamplingBasedController,

        mj_model: mujoco.MjModel,

        mj_data: mujoco.MjData,

    ):





        self.warm_up = False

        self.mj_model = mj_model

        self.mj_data = mj_data

        self.controller = controller

        

        

        mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        mjx_data = mjx_data.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)

        self.mjx_data = mjx_data

        

        self.viewer = None

        self.controller_name = name



        self.max_eval = 200



        if controller is not None:



            # initialize the controller

            jit_optimize = jax.jit(partial(controller.optimize))

            self.jit_optimize = jit_optimize

            

            # logging

            print(

                f"Trajectory Optimization with {controller.num_knots} steps "

                f"over a {controller.ctrl_steps * controller.task.dt} "

                f"second horizon."

            )



            print(f'task.dt:{controller.task.dt}; controller.dt:{controller.dt}; '

                f'task.model.opt.timestep: {controller.task.model.opt.timestep}; '

                f'task.mj_model.opt.timestep: {controller.task.mj_model.opt.timestep}; '

                f'simulator mj_model.opt.timestep: {mj_model.opt.timestep}'

                ) 



    def reset_mjx_data(self):

        """Enhanced reset method that ensures complete state reset"""

        # Create a fresh mjx_data from the original mj_data

        mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        mjx_data = mjx_data.replace(

            mocap_pos=self.mj_data.mocap_pos, 

            mocap_quat=self.mj_data.mocap_quat,

            time=self.mj_data.time,  # Ensure time is also reset

            qpos = self.mj_data.qpos,

            qvel = self.mj_data.qvel

        )

        self.mjx_data = mjx_data



    def optimize(

        self,

        max_iteration: int = 100,

        seed: int = 1

    ) -> list[list, list, Trajectory]:



        knots_list = [] 

        

        self.reset_mjx_data()

        

        policy_params = self.controller.init_params(seed=seed)

        mean_knots = policy_params.mean 



        knots_list.append(mean_knots)

        

        for _ in tqdm.tqdm(range(max_iteration)):

            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)

            

            self.reset_mjx_data()

            mean_knots = policy_params.mean

            knots_list.append(mean_knots)

        if self.max_eval < max_iteration:

            n_it = max_iteration // self.max_eval

            cost_list = []

            for i in range(n_it):

                cost_list_tmp, controls_tmp = self.get_cost_control_list(knots_list[i*self.max_eval:(i+1)*self.max_eval])

                cost_list += cost_list_tmp

            # only keep last controls

            controls = controls_tmp



        else:

            cost_list, controls = self.get_cost_control_list(knots_list)



        return cost_list, controls



    def get_cost_control_list(

        self,

        knots_list: list,

    ) -> list:



        ctrl = self.controller

        task = self.controller.task



        knots = jnp.array(knots_list)      



        controls = self.knots2ctrls(knots)



        state = self.mjx_data

        _, rollouts = ctrl.eval_rollouts(task.model, state, controls, knots)



        costs = jnp.sum(rollouts.costs, axis=-1)



        return list(costs), controls

        

    def knots2ctrls(self,

                    knots: jax.Array

        )-> jax.Array:

        # This function follow exactly how spline interpolation was done in Hydrax: https://github.com/vincekurtz/hydrax/blob/main/hydrax/alg_base.py/#L208



        ctrl = self.controller



        tk = (

            jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + self.mjx_data.time

        )



        tq = jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps)

        controls = ctrl.interp_func(tq, tk, knots)



        return controls

    

    def plot_solution(self, cost_list, controls):

        """

        Plot the sequence of control vectors.

        

        Args:

            controls: array of shape (horizon, Nu), where horizon is

                    the number of time steps and Nu is the number of controls.

        """



        ctrls = np.asarray(controls)

        horizon, Nu = ctrls.shape

        

        dt = float(self.controller.dt)

        t = np.arange(horizon) * dt

        

        ub = getattr(self.controller.task, 'ub', None)

        lb = getattr(self.controller.task, 'lb', None)



        # create one subplot per control dimension

        fig, axes = plt.subplots(Nu, 1, sharex=True, figsize=(8, 2*Nu))

        if Nu == 1:

            axes = [axes]  # make it iterable

        

        for i, ax in enumerate(axes):

            ax.plot(t, ctrls[:, i], lw=1.5)



            if ub is not None and lb is not None:

                # plot upper/lower bounds as horizontal dashed lines

                ax.axhline(y=ub[i], color='r', linestyle='--', label="ub" if i==0 else None)

                ax.axhline(y=lb[i], color='r', linestyle='--', label="lb" if i==0 else None)

            

            ax.set_ylabel(f"$u_{i+1}$")

            ax.grid(True)

        

        task_name = self.controller.task.__class__.__name__



        axes[-1].set_xlabel("Time [s]")

        fig.tight_layout()

        plt.show(block=False)

        plt.savefig("Figures/" + task_name + "_controls.png")





        fig = plt.figure()

        plt.plot(np.array(cost_list))

        plt.grid(True)

        plt.show(block=False)

        plt.savefig("Figures/" + task_name + "_costs.png")



    def visualize_solution(self, controls):

        

        self.__create_temporary_viewer()



        i = 0

        horizon = controls.shape[0]

        dt = float(self.mj_model.opt.timestep)



        i = 0

        while self.viewer.is_running():

            t_start = time.time()



            # apply control and step

            self.tmp_mj_data.ctrl[:] = controls[i]

            mujoco.mj_step(self.mj_model, self.tmp_mj_data)

            self.viewer.sync()



            # sleep the remainder of dt to approximate real time

            elapsed = time.time() - t_start

            to_sleep = dt - elapsed

            if to_sleep > 0:

                time.sleep(to_sleep)



            i += 1

            if i == horizon:

                i = 0

                self.__reset_tmp_data()



        





    def __create_temporary_viewer(self):

        if self.viewer is None:

            self.tmp_mj_data = copy.copy(self.mj_data)

            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.tmp_mj_data)



    def __create_recorder(self):

        self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)

        self.frames = []



    def __reset_tmp_data(self):

        self.tmp_mj_data.qpos[:] = self.mj_data.qpos

        self.tmp_mj_data.qvel[:] = self.mj_data.qvel



    def savegif(self, controls, show_reference=True):



        # Create tmp data (no viewer)

        self.tmp_mj_data = copy.copy(self.mj_data)



        self.__create_recorder()



        if show_reference:

            reference = self.controller.task.goal

            ref_data = mujoco.MjData(self.mj_model)

            ref_data.qpos[:] = reference

            mujoco.mj_forward(self.mj_model, ref_data)

            # Create a separate renderer for the ghost

            ghost_renderer = mujoco.Renderer(self.mj_model, height=480, width=640)

            # if camera_id >= 0:

            #     ghost_renderer.enable_camera_id = camera_id

            

            # Simply update the ghost renderer with the ghost data

            ghost_renderer.update_scene(ref_data)

            ghost_img = ghost_renderer.render().copy()

            ghost_renderer.close()

                    

        horizon = controls.shape[0]

        dt = float(self.mj_model.opt.timestep)



        for i in range(horizon):



            # apply control and step

            self.tmp_mj_data.ctrl[:] = controls[i]

            mujoco.mj_step(self.mj_model, self.tmp_mj_data)

            

            # Render off-screen frame

            self.renderer.update_scene(self.tmp_mj_data)

            frame = self.renderer.render().copy()



            if show_reference:

                alpha = 0.3  # Ghost transparency

                blended_img = (1 - alpha) * frame.astype(np.float32) + alpha * ghost_img.astype(np.float32)

                img = np.clip(blended_img, 0, 255).astype(np.uint8)

            else:

                img = frame.copy()

            self.frames.append(img)









        # Save GIF at the end

        task_name = self.controller.task.__class__.__name__

        imageio.mimsave("Figures/" + task_name + "_simulation.gif", self.frames, fps=int(1/dt))