from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class StaGE_task(Task):

    def __init__(self,
                 mujoco_xml: str,
                 tau_sim: float,
                 q_mask: np.ndarray,
                 cost_max_method: bool=False) -> None:

        mj_model = mujoco.MjModel.from_xml_path(mujoco_xml)
        mj_model.opt.timestep = tau_sim

        super().__init__(
            mj_model
        )

        self.q_mask = jnp.array(q_mask)
        self.cost_max_method = cost_max_method

        self.target_state = jnp.zeros_like(q_mask)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        e = (state.qpos - self.target_state) * self.q_mask
        if self.cost_max_method:
            cost = jnp.abs(e).max()
        else:
            cost = e.T @ e
        return cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        e = (state.qpos - self.target_state) * self.q_mask
        if self.cost_max_method:
            cost = jnp.abs(e).max()
        else:
            cost = e.T @ e
        return cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}
