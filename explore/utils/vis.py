import time
import numpy as np
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
              path: list[dict], sim: MjSim, playback_time: float=1., play_intro: bool=True):
    
    print(f"Playing path with length {len(path)}")

    if play_intro:
        sim.pushConfig(start_state)
        time.sleep(4)
        sim.pushConfig(target_state)
        time.sleep(4)

    tau_action = .1
    sim.setState(*path[0]["state"])
    path.pop(0)
    for node in path:
        q_target = node["state"][3]
        sim.step(tau_action, q_target, view=tau_action*playback_time)
