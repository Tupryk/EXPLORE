import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
        