import math
import mujoco
import numpy as np
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

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

        ### PARALLEL SIMS ###
        self.nworld = cfg.parallel_sims
        self.sim_count = cfg.get("sim_count", 10)
        self.max_workers = cfg.get("max_workers", 5)
        
        if self.nworld == 1:
            self.sim_count = 1
            self.max_workers = 1

        if self.nworld % self.sim_count != 0:
            raise ValueError(
                f"parallel_sims ({self.nworld}) must be divisible by sim_count ({self.sim_count})"
            )
        # Number of worlds each worker thread is responsible for stepping sequentially.
        self.worlds_per_sim = self.nworld // self.sim_count

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.data = []
        for _ in range(self.sim_count):
            self.data.append(mujoco.MjData(self.mj_model))

        self.data_time = np.zeros((self.nworld,))
        self.data_qpos = np.zeros((self.nworld, self.mj_data.qpos.shape[0]))
        self.data_qvel = np.zeros((self.nworld, self.mj_data.qvel.shape[0]))
        self.data_ctrl = np.zeros((self.nworld, self.mj_data.ctrl.shape[0]))
        self.data_geom_xpos = np.zeros((self.nworld, self.mj_data.geom_xpos.shape[0], 3))

        ### COST COMPUTATION ###
        self.q_mask = np.array(cfg.get("q_mask", []))
        self.dist_weight = cfg.get("dist_weight", 0.1)
        self.dist_max = cfg.get("dist_max", 0.2)
        self.vel_weight = cfg.get("velocity_weight", 0.0)

        ### RENDERING ###
        self.frame_dt = 1.0 / cfg.get("fps", 24.0)
        self.next_frame_time = 0.0
        # NOTE: mujoco.Renderer owns a GL (GLX/EGL) context that is tied to the
        # thread that created it (here, the main thread). GL contexts can only be
        # current on one thread at a time, so render()/update_scene() must ONLY
        # ever be called from the main thread. step_seq() runs on ThreadPoolExecutor
        # worker threads, so it must not touch self.renderer directly -- see step().

        render_w = cfg.get("render_w", 640)
        render_h = cfg.get("render_h", 480)
        self.camera = cfg.get("camera", "fixed_cam")

        self.renderer = mujoco.Renderer(self.mj_model, render_h, render_w)

    def gen_numpy_dict(self):
        self.numpy_dict = {
            "time": self.data_time.copy(),
            "qpos": self.data_qpos.copy(),
            "qvel": self.data_qvel.copy(),
            "ctrl": self.data_ctrl.copy(),
            "geom_xpos": self.data_geom_xpos.copy(),
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
            # Broadcast single-world inputs to all worlds. np.broadcast_to returns a
            # read-only view, so take a copy to make the arrays writable.
            if time.ndim == 1 and time.shape[0] != self.nworld:
                time = np.broadcast_to(time[:1], (self.nworld,)).copy()
            else:
                time = np.array(time, copy=True)

            if qpos.ndim == 1:
                qpos = np.broadcast_to(qpos, (self.nworld, qpos.shape[0])).copy()
            else:
                qpos = np.array(qpos, copy=True)

            if qvel.ndim == 1:
                qvel = np.broadcast_to(qvel, (self.nworld, qvel.shape[0])).copy()
            else:
                qvel = np.array(qvel, copy=True)

            if ctrl.ndim == 1:
                ctrl = np.broadcast_to(ctrl, (self.nworld, ctrl.shape[0])).copy()
            else:
                ctrl = np.array(ctrl, copy=True)

            # Copy into the simulator state.
            self.data_time = time
            self.data_qpos = qpos
            self.data_qvel = qvel
            self.data_ctrl = ctrl

            self.data_geom_xpos = np.zeros(
                (self.nworld, self.mj_data.geom_xpos.shape[0], 3)
            )
            self.next_frame_time = 0.0

        else:
            self.data_time[indices] = time
            self.data_qpos[indices] = qpos
            self.data_qvel[indices] = qvel
            self.data_ctrl[indices] = ctrl

            self.data_geom_xpos[indices] = 0.0
            if 0 in indices: self.next_frame_time = 0.0

    def getState(self):
        """
        Returns:
            time:  [nworld]  float32 numpy array
            qpos:  [nworld, nq]
            qvel:  [nworld, nv]
            ctrl:  [nworld, nu]
        """
        return (
            self.data_time.copy(),
            self.data_qpos.copy(),
            self.data_qvel.copy(),
            self.data_ctrl.copy(),
        )

    def render_state(self, qpos: np.ndarray) -> np.ndarray:
        self.mj_data.qpos[:] = qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data, self.camera)
        return self.renderer.render()

    def step_seq(self, tau_action: float, ctrl_targets: np.ndarray, sim_idx: int, render: bool = False) -> list[np.ndarray]:
        """
        Args:
            tau_action:   duration to simulate
            ctrl_targets: [worlds_per_sim, nu] target control at end of window
                          for the worlds owned by this thread
            sim_idx:      index of the worker / MjData replica handling this chunk
        """

        # NOTE: this runs on a worker thread. We must NOT call self.renderer here
        # (see note in __init__) -- instead we record qpos snapshots for any frame
        # that's due, and the caller (step(), on the main thread) renders them
        # afterwards. Only sim_idx == 0's chunk ever appends anything, since that's
        # the only chunk that can contain nworld_idx == 0.
        qpos_snapshots = []
        steps = math.ceil(tau_action / self.tau_sim)
        for ctrl_i, ctrl_target in enumerate(ctrl_targets):

            # Global index into the nworld-sized state arrays for this world.
            nworld_idx = sim_idx * self.worlds_per_sim + ctrl_i

            self.data[sim_idx].time = self.data_time[nworld_idx]
            self.data[sim_idx].qpos[:] = self.data_qpos[nworld_idx]
            self.data[sim_idx].qvel[:] = self.data_qvel[nworld_idx]
            self.data[sim_idx].ctrl[:] = self.data_ctrl[nworld_idx]

            mujoco.mj_forward(self.mj_model, self.data[sim_idx])
            prev_ctrl = self.data_ctrl[nworld_idx]

            for k in range(steps):
                perc = (k + 1) / steps
                interpolated_ctrl = prev_ctrl * (1 - perc) + ctrl_target * perc
                self.data[sim_idx].ctrl[:] = interpolated_ctrl
                mujoco.mj_step(self.mj_model, self.data[sim_idx])

                if render and nworld_idx == 0 and self.data[sim_idx].time >= self.next_frame_time:
                    qpos_snapshots.append(self.data[sim_idx].qpos.copy())
                    self.next_frame_time += self.frame_dt
                    # print(self.data[sim_idx].time)

            self.data_time[nworld_idx] = self.data[sim_idx].time
            self.data_qpos[nworld_idx] = self.data[sim_idx].qpos[:]
            self.data_qvel[nworld_idx] = self.data[sim_idx].qvel[:]
            self.data_ctrl[nworld_idx] = self.data[sim_idx].ctrl[:]
            self.data_geom_xpos[nworld_idx] = self.data[sim_idx].geom_xpos

        return qpos_snapshots

    def step(self, tau_action: float, ctrl_target: np.ndarray, render: bool = False):
        
        if ctrl_target.ndim == 1: ctrl_target = ctrl_target.reshape(1, -1)
        
        futures = [
            self.executor.submit(
                self.step_seq,
                tau_action,
                ctrl_target[
                    sim_idx * self.worlds_per_sim
                    :
                    (sim_idx + 1) * self.worlds_per_sim
                ],
                sim_idx,
                render
            )
            for sim_idx in range(self.sim_count)
        ]
        qpos_snapshots = []
        for future in as_completed(futures):
            qpos_snapshots.extend(future.result())

        # Actual GL rendering happens here, on the main thread that owns the
        # renderer's GL context -- never inside step_seq() / worker threads.
        frames = [self.render_state(qpos) for qpos in qpos_snapshots]

        return frames
