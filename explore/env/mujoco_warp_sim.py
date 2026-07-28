import math
import mujoco
import warp as wp
import numpy as np
import mujoco_warp as mjw
from omegaconf import DictConfig

from explore.utils.mj import explain_qpos


@wp.kernel
def compute_spline_coeffs_kernel(
    ctrl: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    ctrl_target: wp.array2d(dtype=wp.float32),
    joint_vel_start: int,
    lmbda: float,
    prev_ctrl_out: wp.array2d(dtype=wp.float32),
    v0_out: wp.array2d(dtype=wp.float32),
    accel_coef_out: wp.array2d(dtype=wp.float32),
):
    """One thread per (world, actuator). Reads ctrl/qvel straight off the
    live GPU buffers -- no CPU roundtrip -- and writes the three spline
    coefficients needed to evaluate r(dt) for the rest of the window."""
    w, i = wp.tid()

    pc = ctrl[w, i]
    v = qvel[w, joint_vel_start + i]
    tgt = ctrl_target[w, i]

    action = 2.0 * (tgt - pc)
    accel = (action - 2.0 * lmbda * v) / (2.0 * lmbda * lmbda)

    prev_ctrl_out[w, i] = pc
    v0_out[w, i] = v
    accel_coef_out[w, i] = accel


@wp.kernel
def eval_spline_kernel(
    prev_ctrl: wp.array2d(dtype=wp.float32),
    v0: wp.array2d(dtype=wp.float32),
    accel_coef: wp.array2d(dtype=wp.float32),
    dt: float,
    ctrl_out: wp.array2d(dtype=wp.float32),
):
    """One thread per (world, actuator). Pure GPU-side polynomial eval,
    writes directly into data.ctrl."""
    w, i = wp.tid()
    ctrl_out[w, i] = prev_ctrl[w, i] + v0[w, i] * dt + accel_coef[w, i] * dt * dt


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

        ### SPLINE ACTION STATE ###
        # joint_vel_ids: [start, end) slice into qvel for the actuated joints,
        # matched 1:1 with the nu ctrl channels (same convention as the
        # threaded CPU implementation).
        self.joint_vel_ids = cfg.joint_vel_ids
        self.nu = self.mj_model.nu

        assert self.joint_vel_ids[1] - self.joint_vel_ids[0] == self.nu, (
            "joint_vel_ids span must match nu (one velocity per actuator)"
        )

        wp_device = self.data.ctrl.device
        self.prev_ctrl = wp.zeros((self.nworld, self.nu), dtype=wp.float32, device=wp_device)
        self.v0 = wp.zeros((self.nworld, self.nu), dtype=wp.float32, device=wp_device)
        self.accel_coef = wp.zeros((self.nworld, self.nu), dtype=wp.float32, device=wp_device)

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
            "time": self.data.time.numpy().copy(),
            "qpos": self.data.qpos.numpy().copy(),
            "qvel": self.data.qvel.numpy().copy(),
            "ctrl": self.data.ctrl.numpy().copy(),
            "geom_xpos": self.data.geom_xpos.numpy().copy(),
        }

    def setState(
        self,
        time: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        indices: np.ndarray = None
    ):
        """
        Args:
            time:  [nworld]
            qpos:  [nworld, nq]
            qvel:  [nworld, nv]
            ctrl:  [nworld, nu]
        """
        if indices is None:
            if time.ndim == 1 and time.shape[0] != self.nworld:
                time = np.broadcast_to(time[:1], (self.nworld,))
            if qpos.ndim == 1:
                qpos = np.broadcast_to(qpos, (self.nworld, qpos.shape[0]))
            if qvel.ndim == 1:
                qvel = np.broadcast_to(qvel, (self.nworld, qvel.shape[0]))
            if ctrl.ndim == 1:
                ctrl = np.broadcast_to(ctrl, (self.nworld, ctrl.shape[0]))

            self.next_frame_time = 0.0
            self.data.time.assign(wp.array(time, dtype=wp.float32))
            self.data.qpos.assign(wp.array(qpos, dtype=wp.float32))
            self.data.qvel.assign(wp.array(qvel, dtype=wp.float32))
            self.data.ctrl.assign(wp.array(ctrl, dtype=wp.float32))

        else:
            if 0 in indices: self.next_frame_time = 0.0
            time_np = self.data.time.numpy()
            qpos_np = self.data.qpos.numpy()
            qvel_np = self.data.qvel.numpy()
            ctrl_np = self.data.ctrl.numpy()

            time_np[indices] = time
            qpos_np[indices] = qpos
            qvel_np[indices] = qvel
            ctrl_np[indices] = ctrl

            self.data.time.assign(wp.array(time_np, dtype=wp.float32))
            self.data.qpos.assign(wp.array(qpos_np, dtype=wp.float32))
            self.data.qvel.assign(wp.array(qvel_np, dtype=wp.float32))
            self.data.ctrl.assign(wp.array(ctrl_np, dtype=wp.float32))

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

    def step(self, tau_action: float, ctrl_target: np.ndarray, render: bool = False) -> list[np.ndarray]:
        """
        Args:
            tau_action:   duration to simulate
            ctrl_target:  [nworld, nu] target control at end of window

        Follows a quadratic ("spline") trajectory in ctrl-space, matching the
        threaded CPU implementation's r(dt) = prev_ctrl + v0*dt + accel_coef*dt^2,
        but with prev_ctrl/v0/accel_coef computed and evaluated entirely on the
        GPU. The only host->device transfer per call is uploading ctrl_target
        itself; everything else (reading current ctrl/qvel, computing the
        coefficients, evaluating the polynomial every substep, writing back
        into data.ctrl) happens in warp kernels with no intermediate numpy
        round-trip.
        """
        steps = math.ceil(tau_action / self.tau_sim)
        lmbda = 2.0 * tau_action
        wp_device = self.data.ctrl.device

        if ctrl_target.ndim == 1:
            ctrl_target = np.broadcast_to(ctrl_target, (self.nworld, ctrl_target.shape[0]))

        ctrl_target_wp = wp.array(np.ascontiguousarray(ctrl_target), dtype=wp.float32, device=wp_device)

        # One-shot: derive prev_ctrl / v0 / accel_coef for this window directly
        # from the live GPU state (data.ctrl, data.qvel).
        wp.launch(
            compute_spline_coeffs_kernel,
            dim=(self.nworld, self.nu),
            inputs=[
                self.data.ctrl,
                self.data.qvel,
                ctrl_target_wp,
                self.joint_vel_ids[0],
                lmbda,
            ],
            outputs=[self.prev_ctrl, self.v0, self.accel_coef],
            device=wp_device,
        )

        frames = []

        for k in range(steps):
            dt = (k + 1) * self.tau_sim

            # Evaluate the polynomial for this substep and write straight into
            # data.ctrl -- no CPU involvement at all.
            wp.launch(
                eval_spline_kernel,
                dim=(self.nworld, self.nu),
                inputs=[self.prev_ctrl, self.v0, self.accel_coef, dt],
                outputs=[self.data.ctrl],
                device=wp_device,
            )

            mjw.step(self.model, self.data)

            if render and self.data.time.numpy()[0] >= self.next_frame_time:

                mjw.get_data_into(self.mj_data, self.mj_model, self.data)
                self.renderer.update_scene(self.mj_data, self.camera)
                frames.append(self.renderer.render())

                self.next_frame_time += self.frame_dt

        return frames
    