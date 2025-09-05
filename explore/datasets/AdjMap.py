import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from explore.datasets.rnd_configs import RndConfigs


class AdjMap:

    def __init__(self, epsilon: float=5e-2, configs_dir: str="",
                 output_dir: str="", notebook: bool=False):

        self.notebook = notebook
        stable_config_count = 100
        self.output_dir = output_dir
        self.costs = np.zeros((stable_config_count, stable_config_count))

        g_path = "configs/twoFingers.g"
        h5_path = "configs/rnd_twoFingers.h5"
        if configs_dir:
            g_path = os.path.join(configs_dir, g_path)
            h5_path = os.path.join(configs_dir, h5_path)
        D = RndConfigs(g_path, h5_path)

        self.epsilon = epsilon

        if not self.notebook:
            plt.ion()
        fig, ax = plt.subplots()

        self.im = ax.imshow(self.costs, cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.0)

        green_cmap = ListedColormap(["red"])
        self.overlay = ax.imshow(np.full_like(self.costs, np.nan), cmap=green_cmap, interpolation="nearest", alpha=0.6)

        plt.colorbar(self.im, ax=ax, label="Cost")
        ax.set_title("Cost Between Configs")
        ax.set_xlabel("End Config")
        ax.set_ylabel("Start Config")

        for start_idx in range(stable_config_count):
            for end_idx in range(stable_config_count):
                if end_idx == start_idx:
                    continue

                start_state = D.positions[start_idx]
                end_state = D.positions[end_idx]

                cost = np.linalg.norm(start_state - end_state)
                self.set_value(start_idx, end_idx, cost)

        self.update_data()
        if not self.notebook:
            plt.ioff()
            plt.pause(0.1)

    def set_value(self, start_idx: int, end_idx: int, cost: float):
        self.costs[start_idx, end_idx] = cost

    def update_data(self):
        self.im.set_data(self.costs)
        mask = self.costs < self.epsilon
        self.overlay.set_data(np.where(mask, 1, np.nan))
        if not self.notebook:
            plt.pause(0.01)

    def save(self, prefix: str=""):
        im_path = f"{prefix}AdjMap.png"
        if self.output_dir:
            im_path = os.path.join(self.output_dir, im_path)
        plt.savefig(im_path)

    def show(self):
        plt.show()


if __name__ == "__main__":
    import time
    adj_map = AdjMap()
    adj_map.set_value(10, 80, 0.)
    adj_map.save()
    time.sleep(1.)
    adj_map.update_data()
    time.sleep(1.)
    