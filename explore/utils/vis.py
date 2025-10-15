import time
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from explore.env.mujoco_sim import MjSim


def AdjMap(costs: np.ndarray, min_value: float=0.0, max_value: float=0.1, save_as: str=""):

    fig, ax = plt.subplots()

    im = ax.imshow(costs, cmap="Blues", interpolation="nearest", vmin=min_value, vmax=max_value)

    green_cmap = ListedColormap(["red"])
    overlay = ax.imshow(np.full_like(costs, np.nan), cmap=green_cmap, interpolation="nearest", alpha=0.6)
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


def play_path(start_state: np.ndarray, target_state: np.ndarray,
              path: list[dict], sim: MjSim, playback_time: float=1.,
              tau_action: float=.1, play_intro: bool=True, camera: str="",
              save_as: str="path.gif"):
    
    print(f"Playing path with length {len(path)}")
    sim.setupRenderer(camera=camera)

    if play_intro:
        sim.pushConfig(start_state)
        im_start = sim.renderImg()
        if sim.viewer != None:
            time.sleep(3)
        sim.pushConfig(target_state)
        im_end = sim.renderImg()
        if sim.viewer != None:
            time.sleep(3)
        sim.pushConfig(path[-1]["state"][1])
        im_reached = sim.renderImg()
        if sim.viewer != None:
            time.sleep(3)
    
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
    plt.show()

    frames = []

    sim.setState(*path[0]["state"])
    path.pop(0)

    for node in path:
        q_target = node["state"][3]
        f = sim.step(tau_action, q_target, view=tau_action*playback_time)
        if save_as:
            frames.extend(f)

    if sim.viewer != None:
        time.sleep(3)
    if save_as:
        imageio.mimsave(save_as, frames, fps=24, loop=0)
