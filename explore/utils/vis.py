import time
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from explore.env.mujoco_sim import MjSim


def AdjMap(costs: np.ndarray, min_value: float=0.0, max_value: float=0.1, save_as: str=""):

    fig, ax = plt.subplots()

    im = ax.imshow(costs, cmap="Blues", interpolation="nearest", vmin=min_value, vmax=max_value)

    cmap = ListedColormap(["red"])
    overlay = ax.imshow(np.full_like(costs, np.nan), cmap=cmap, interpolation="nearest", alpha=0.6)
    im.set_data(costs)
    mask = costs < min_value
    overlay.set_data(np.where(mask, 1, np.nan))

    plt.colorbar(im, ax=ax, label="Cost")
    ax.set_title("Cost Between Configs")
    ax.set_xlabel("End Config")
    ax.set_ylabel("Start Config")
    
    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()


def play_path(path: list[dict], sim: MjSim,
              start_state: np.ndarray, target_state: np.ndarray,
              playback_time: float=1., tau_action: float=.1, save_intro_as: str="",
              camera: str="", save_as: str="path.gif", reset_state: bool=False) -> list[np.ndarray]:
    
    play_path = path.copy()
    
    print(f"Playing path with length {len(play_path)}")
    sim.setupRenderer(camera=camera)

    
    if len(start_state) and len(target_state):
        sim.pushConfig(start_state, ignore_warn=True)
        im_start = sim.renderImg()
        sim.pushConfig(target_state, ignore_warn=True)
        im_end = sim.renderImg()
        sim.pushConfig(play_path[-1]["state"][1], ignore_warn=True)
        im_reached = sim.renderImg()
        
        fig, axes = plt.subplots(1, 3, figsize=(30, 20))
        axes[0].set_title("Start Config", fontsize=24, fontweight="bold")
        axes[0].imshow(im_start)
        axes[0].axis("off")
        axes[1].set_title("Target Config", fontsize=24, fontweight="bold")
        axes[1].imshow(im_end)
        axes[1].axis("off")
        axes[2].set_title("Reached Config", fontsize=24, fontweight="bold")
        axes[2].imshow(im_reached)
        axes[2].axis("off")
        plt.tight_layout()
        
        if not save_intro_as:
            plt.show()
        else:
            plt.savefig(save_intro_as)

    frames = []
    states = []

    prev_node = play_path[0]
    sim.setState(*prev_node["state"])

    for node in play_path[1:]:
        
        if reset_state:
            sim.setState(*prev_node["state"])
        
        # TODO: Make this nicer
        # if "q_sequence" not in node.keys():
        q_target = node["state"][4] if sim.use_spline_ref else node["state"][3]
        f, s, c = sim.step(tau_action, q_target, view=camera)
        frames.extend(f)
        states.extend(s)
        # else:
        #     for q in node["q_sequence"]:
        #         f, s, c = sim.step(sim.tau_sim, q, view=camera)
        #         frames.extend(f)
        #         states.extend(s)
        
        prev_node = node

    # frames.extend([np.zeros_like(f) for _ in range(24)])

    if sim.viewer != None:
        time.sleep(3)
    if save_as:
        imageio.mimsave(save_as, frames, fps=24 * playback_time, loop=0)
    
    return frames, states

def play_path_mppi(controls: np.ndarray, sim: MjSim,
                   start_state: np.ndarray, target_state: np.ndarray,
                   playback_time: float = 1., tau_action: float = 0.1,
                   save_intro_as: str = "", camera: str = "",
                   save_as: str = "path.gif") -> tuple[list, list]:
    """
    Play a path defined by an (N, dof) array of joint position targets.
    Each row is a q_target passed to sim.step for tau_action seconds.
    """
    print(f"Playing MPPI path with {len(controls)} steps")

    if not hasattr(sim, 'renderer') or sim.renderer is None:
            sim.setupRenderer(camera=camera)

    frames = []
    states = []

    # Run the full playback first so we can grab the last state for the intro figure
    sim.setState(0.0, start_state, np.zeros(sim.model.nv), start_state[:sim.model.nu])
    for q_target in controls:
        f, s, c = sim.step(tau_action, q_target, view=camera)
        frames.extend(f)
        states.extend(s)

    # --- Intro figure: start / target / reached ---
    if len(start_state) and len(target_state):
        sim.pushConfig(start_state, ignore_warn=True)
        im_start = sim.renderImg()

        sim.pushConfig(target_state, ignore_warn=True)
        im_end = sim.renderImg()

        # Last state from the rollout above
        last_qpos = states[-1][1] if isinstance(states[-1], tuple) else states[-1]
        sim.pushConfig(last_qpos, ignore_warn=True)
        im_reached = sim.renderImg()

        fig, axes = plt.subplots(1, 3, figsize=(30, 20))
        axes[0].set_title("Start Config", fontsize=24, fontweight="bold")
        axes[0].imshow(im_start)
        axes[0].axis("off")
        axes[1].set_title("Target Config", fontsize=24, fontweight="bold")
        axes[1].imshow(im_end)
        axes[1].axis("off")
        axes[2].set_title("Reached Config", fontsize=24, fontweight="bold")
        axes[2].imshow(im_reached)
        axes[2].axis("off")
        plt.tight_layout()

        if not save_intro_as:
            plt.show()
        else:
            plt.savefig(save_intro_as)
        
        plt.close(fig)

    if sim.viewer is not None:
        time.sleep(3)

    if save_as:
        imageio.mimsave(save_as, frames, fps=24 * playback_time, loop=0)

    return frames, states

import mujoco
def visualize_path_mppi(controls: np.ndarray, 
                         sim: MjSim,
                         start_qpos: np.ndarray,
                         start_ctrl: np.ndarray,
                         playback_speed: float = 1.0) -> None:
    """
    Replays a stored MPPI trajectory using the MjSim class.
    controls: shape (N, nu) - raw actuator values stored during generation
    """
    print(f"Visualizing MPPI path: {len(controls)} steps")

    if sim.viewer is None:
        sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    sim.pushConfig(start_qpos, start_ctrl)

    for i, ctrl in enumerate(controls):
        sim.step(
            tau_action=sim.tau_sim,   # one sim step per control
            ctrl_target=ctrl,
            view="viewer"             # triggers viewer.sync()
        )
        time.sleep(sim.tau_sim / playback_speed)

    print("Rollout complete.")