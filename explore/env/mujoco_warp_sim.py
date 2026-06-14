import math
import mujoco
import warp as wp
import numpy as np
import mujoco_warp as mjw
from omegaconf import DictConfig

from explore.utils.mj import explain_qpos


class MjSim:

    def __init__(self, cfg: DictConfig):

        self.verbose = cfg.get("verbose", 0)
        self.tau_sim = cfg.get("tau_sim", 1e-3)

        ### MJ MODEL AND DATA ###
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.mj_model.opt.timestep = self.tau_sim
        self.mj_data = mujoco.MjData(self.mj_model)

        if self.verbose:
            print(f"Loaded config '{cfg.xml_path}' with position values:")
            print(self.mj_data.qpos)
            explain_qpos(self.mj_model)

        ### WARP MODEL AND DATA ###
        njmax = cfg.get("njmax", -1)
        nconmax = cfg.get("nconmax", -1)
        ccd_iterations = cfg.get("ccd_iterations", -1)
        
        if ccd_iterations != -1: self.mj_model.opt.ccd_iterations = ccd_iterations

        self.model = mjw.put_model(self.mj_model)

        if njmax != -1:
            self.data = mjw.put_data(
                self.mj_model,
                self.mj_data,
                nworld=cfg.parallel_sims,
                njmax=njmax,
                nconmax=nconmax
            )
        else:
            self.data = mjw.put_data(self.mj_model, self.mj_data, nworld=cfg.parallel_sims)
        
        self.nworld = cfg.parallel_sims

        ### COST COMPUTATION ###
        self.q_mask = np.array(cfg.get("q_mask", []))
        self.dist_weight = cfg.get("dist_weight", 0.1)
        self.dist_max = cfg.get("dist_max", 0.2)
        self.vel_weight = cfg.get("velocity_weight", 0.0)
        
        ### RENDERING ###
        self.frame_dt = 1.0 / cfg.get("fps", 24.0)
        self.next_frame_time = 0.0
        
        render_w = cfg.get("render_w", 640)
        render_h = cfg.get("render_h", 480)
        self.camera = cfg.get("camera", "fixed_cam")
        
        self.renderer = mujoco.Renderer(self.mj_model, render_h, render_w)

    def gen_numpy_dict(self):
        """GPU to CPU"""
        self.numpy_dict = {
            "qpos": self.data.qpos.numpy().copy(),
            "qvel": self.data.qvel.numpy().copy(),
            "ctrl": self.data.ctrl.numpy().copy(),
            "geom_xpos": self.data.geom_xpos.numpy().copy(),
        }

    def pushConfig(self, joint_state: np.ndarray, ctrl_state: np.ndarray, indices: np.ndarray = None):
        """
        Reset worlds to the given state and run a forward pass.
        Args:
            joint_state: [n, nq] where n == nworld if indices is None, else n == len(indices)
            ctrl_state:  [n, nu]
            indices:     optional 1-D array of world indices to reset; if None, resets all
        """
        if indices is None:
            self.next_frame_time = 0.0
            self.data.time.assign(wp.zeros(self.nworld, dtype=wp.float32))
            self.data.qpos.assign(wp.array(joint_state, dtype=wp.float32))
            self.data.qvel.assign(wp.zeros_like(self.data.qvel))
            self.data.ctrl.assign(wp.array(ctrl_state, dtype=wp.float32))
        else:
            if 0 in indices: self.next_frame_time = 0.0
            time_np = self.data.time.numpy()
            qpos_np = self.data.qpos.numpy()
            qvel_np = self.data.qvel.numpy()
            ctrl_np = self.data.ctrl.numpy()

            time_np[indices] = 0.0
            qpos_np[indices] = joint_state
            qvel_np[indices] = 0.0
            ctrl_np[indices] = ctrl_state

            self.data.time.assign(wp.array(time_np, dtype=wp.float32))
            self.data.qpos.assign(wp.array(qpos_np, dtype=wp.float32))
            self.data.qvel.assign(wp.array(qvel_np, dtype=wp.float32))
            self.data.ctrl.assign(wp.array(ctrl_np, dtype=wp.float32))

        mjw.forward(self.model, self.data)

    def setState(
        self,
        time: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ):
        """
        Args:
            time:  [nworld]
            qpos:  [nworld, nq]
            qvel:  [nworld, nv]
            ctrl:  [nworld, nu]
        """
        self.data.time.assign(wp.array(time, dtype=wp.float32))
        self.data.qpos.assign(wp.array(qpos, dtype=wp.float32))
        self.data.qvel.assign(wp.array(qvel, dtype=wp.float32))
        self.data.ctrl.assign(wp.array(ctrl, dtype=wp.float32))
        mjw.forward(self.model, self.data)

    def getState(self):
        """
        Returns:
            time:  [nworld]  float32 numpy array
            qpos:  [nworld, nq]
            qvel:  [nworld, nv]
            ctrl:  [nworld, nu]
        """
        return (
            self.data.time.numpy().copy(),
            self.data.qpos.numpy().copy(),
            self.data.qvel.numpy().copy(),
            self.data.ctrl.numpy().copy(),
        )
    
    def render_state(self, qpos: np.ndarray) -> np.ndarray:
        self.mj_data.qpos[:] = qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data, self.camera)
        return self.renderer.render()

    def step(self, tau_action: float, ctrl_target: np.ndarray, render: bool=False) -> list[np.ndarray]:
        """
        Args:
            tau_action:   duration to simulate
            ctrl_target:  [nworld, nu] target control at end of window
        """
        steps = math.ceil(tau_action / self.tau_sim)
        prev_ctrl = self.data.ctrl.numpy().copy()   # [nworld, nu]
        frames = []

        for k in range(steps):
            perc = (k + 1) / steps
            interpolated_ctrl = prev_ctrl * (1 - perc) + ctrl_target * perc
            self.data.ctrl.assign(wp.array(interpolated_ctrl, dtype=wp.float32))
            mjw.step(self.model, self.data)
            
            if render and self.data.time.numpy()[0] >= self.next_frame_time:
                
                mjw.get_data_into(self.mj_data, self.mj_model, self.data)
                self.renderer.update_scene(self.mj_data, self.camera)
                frames.append(self.renderer.render())
                
                self.next_frame_time += self.frame_dt
            
        return frames
            