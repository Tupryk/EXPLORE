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
        # mj_model / mj_data are kept only for CPU-only MuJoCo calls
        # (mj_name2id, mj_geomDistance, mj_objectVelocity).
        # All simulation state lives in self.model / self.data (warp).
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.mj_model.opt.timestep = self.tau_sim
        self.mj_data = mujoco.MjData(self.mj_model)

        if self.verbose:
            print(f"Loaded config '{cfg.xml_path}' with position values:")
            print(self.mj_data.qpos)
            explain_qpos(self.mj_model)

        ### WARP MODEL AND DATA ###
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.put_data(self.mj_model, self.mj_data, nworld=cfg.parallel_sims)
        self.nworld = cfg.parallel_sims

        ### COST COMPUTATION ###
        self.q_mask = np.array(cfg.get("q_mask", []))
        self.dist_weight = cfg.get("dist_weight", 0.1)
        self.dist_max = cfg.get("dist_max", 0.2)
        self.vel_weight = cfg.get("velocity_weight", 0.0)

        # Basic
        self.custom_state_sequence = []
        self.custom_state_sequence_scaled = []
        if len(self.q_mask):
            def q_state() -> np.ndarray:
                # [nworld, nq]
                return self.data.qpos.numpy().copy()
            self.custom_state_sequence.append(q_state)
            self.custom_state_sequence_scaled.append(lambda: q_state() * self.q_mask)

        if self.vel_weight:
            def qvel_state() -> np.ndarray:
                # [nworld, nv]
                return self.data.qvel.numpy().copy()
            self.custom_state_sequence.append(qvel_state)
            self.custom_state_sequence_scaled.append(lambda: qvel_state() * self.vel_weight)

        # Contacts — geom IDs resolved on mj_model (indices identical in warp)
        obj_names = cfg.get("objs", [])
        self.objs = []
        for geom_name in obj_names:
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.objs.append(geom_id)

        contact_names = cfg.get("contacts", [])
        self.contacts = []
        for geom_name in contact_names:
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.contacts.append(geom_id)

        if self.objs:
            def dists_state() -> np.ndarray:
                # mj_geomDistance is CPU-only; loop over worlds by copying each row
                # into mj_data, running a forward kinematics update, then querying.
                # Returns [nworld, n_objs * n_contacts].
                n_pairs = len(self.objs) * len(self.contacts)
                all_dists = np.zeros((self.nworld, n_pairs))
                qpos_all = self.data.qpos.numpy()   # [nworld, nq]
                qvel_all = self.data.qvel.numpy()   # [nworld, nv]
                fromto = np.zeros(6)
                for w in range(self.nworld):
                    self.mj_data.qpos[:] = qpos_all[w]
                    self.mj_data.qvel[:] = qvel_all[w]
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    i = 0
                    for oi in self.objs:
                        for ci in self.contacts:
                            dist = mujoco.mj_geomDistance(
                                self.mj_model, self.mj_data, ci, oi, self.dist_max, fromto
                            )
                            all_dists[w, i] = 1 - np.clip(dist / self.dist_max, 0.0, 1.0)
                            i += 1
                return all_dists
            self.custom_state_sequence.append(dists_state)
            self.custom_state_sequence_scaled.append(lambda: dists_state() * self.dist_weight)

        # Geometries
        self.geoms_in_cost = []
        geoms_in_cost_names = cfg.get("geoms_in_cost", [])
        self.geoms_in_cost_weights = np.array(cfg.get("geoms_in_cost_weights", []))
        self.vels_geoms_in_cost_weights = np.array(cfg.get("vels_geoms_in_cost_weights", []))
        for geom_name in geoms_in_cost_names:
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.geoms_in_cost.append(geom_id)

        if self.geoms_in_cost:
            def geoms_state() -> np.ndarray:
                # geom_xpos is [nworld, ngeom, 3]; select desired geoms → [nworld, n_geoms, 3]
                return self.data.geom_xpos.numpy()[:, self.geoms_in_cost, :].reshape(self.nworld, -1)
            self.custom_state_sequence.append(geoms_state)
            self.custom_state_sequence_scaled.append(lambda: geoms_state() * self.geoms_in_cost_weights)

        if len(self.vels_geoms_in_cost_weights):
            def geoms_vels_state() -> np.ndarray:
                # mj_objectVelocity is CPU-only; loop over worlds as above.
                # Returns [nworld, n_geoms * 3].
                n_geoms = len(self.geoms_in_cost)
                all_vels = np.empty((self.nworld, n_geoms * 3))
                qpos_all = self.data.qpos.numpy()
                qvel_all = self.data.qvel.numpy()
                for w in range(self.nworld):
                    self.mj_data.qpos[:] = qpos_all[w]
                    self.mj_data.qvel[:] = qvel_all[w]
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    for i, geom_id in enumerate(self.geoms_in_cost):
                        vel6 = np.zeros(6)
                        mujoco.mj_objectVelocity(
                            self.mj_model,
                            self.mj_data,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            geom_id,
                            vel6,
                            0,  # world frame
                        )
                        all_vels[w, i * 3:(i + 1) * 3] = vel6[3:]
                return all_vels
            self.custom_state_sequence.append(geoms_vels_state)
            self.custom_state_sequence_scaled.append(
                lambda: geoms_vels_state() * self.vels_geoms_in_cost_weights
            )

    def getCustomState(self) -> np.ndarray:
        return np.concatenate([cs() for cs in self.custom_state_sequence], axis=-1)

    def getCustomStateScaled(self) -> np.ndarray:
        return np.concatenate([cs() for cs in self.custom_state_sequence_scaled], axis=-1)

    def pushConfig(self, joint_state: np.ndarray, ctrl_state: np.ndarray, indices: np.ndarray = None):
        """
        Reset worlds to the given state and run a forward pass.
        Args:
            joint_state: [n, nq] where n == nworld if indices is None, else n == len(indices)
            ctrl_state:  [n, nu]
            indices:     optional 1-D array of world indices to reset; if None, resets all
        """
        if indices is None:
            self.data.time.assign(wp.zeros(self.nworld, dtype=wp.float32))
            self.data.qpos.assign(wp.array(joint_state, dtype=wp.float32))
            self.data.qvel.assign(wp.zeros_like(self.data.qvel))
            self.data.ctrl.assign(wp.array(ctrl_state, dtype=wp.float32))
        else:
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

    def step(self, tau_action: float, ctrl_target: np.ndarray):
        """
        Args:
            tau_action:   duration to simulate
            ctrl_target:  [nworld, nu] target control at end of window
        """
        steps = math.ceil(tau_action / self.tau_sim)
        prev_ctrl = self.data.ctrl.numpy().copy()   # [nworld, nu]

        for k in range(steps):
            perc = (k + 1) / steps
            interpolated_ctrl = prev_ctrl * (1 - perc) + ctrl_target * perc
            self.data.ctrl.assign(wp.array(interpolated_ctrl, dtype=wp.float32))
            mjw.step(self.model, self.data)
            