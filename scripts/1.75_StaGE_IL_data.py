import os
import h5py
import hydra
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from explore.datasets.StaGE import StaGE
from explore.datasets.stage_dataset import StaGEDataset
from explore.utils.learned_stage import *


@hydra.main(
    version_base="1.3",
    config_path="../configs/yaml/StaGE",
    config_name="fingerRamp")
def main(cfg: DictConfig):

    # Generate initial tree
    file = h5py.File(cfg.configs_path, 'r')
    qpos = file["qpos"] if "qpos" in file.keys() else file["q"]
    ctrl = file["ctrl"]

    S = StaGE(qpos, ctrl, cfg.RRT)
    tree = S.init_tree(0)

    dataset_size = int(1e6)
    dataset = StaGEDataset()

    # Main loop
    pbar = tqdm(total=dataset_size, desc="Filling dataset")
    total_trees = 0
    while True:
        # Generate a new tree
        S.start_ids = [np.random.randint(0, S.manifold_size)]
        tree = S.run()

        # Load tree into buffer
        end_nodes, reached_targets = get_tree_successful_nodes(tree, S.all_G_star, S.min_cost)
        connection_ratio = len(reached_targets) / len(S.all_G_star) * 100.

        states, actions, next_states, rewards, dones = tree_to_buffer(
            tree, end_nodes, reached_targets, S, 0.0, min_traj_len=1.0
        )

        if len(states) != 0:
            dataset.add_multiple(
                states,
                actions,
                next_states,
                rewards.reshape(-1, 1),
                dones.reshape(-1, 1)
            )
        else:
            tqdm.write(f"WARNING: No connections found in loop {total_trees + 1}!")

        pbar.set_postfix({"conn%": f"{connection_ratio:.2f}", "loop": total_trees + 1})
        pbar.n = min(len(dataset), dataset_size)
        pbar.refresh()

        total_trees += 1
        if len(dataset) >= dataset_size:
            break

    stage_dataset_path = os.path.join(cfg.RRT.output_dir, "stage_dataset.pt")
    torch.save(dataset.state_dict(), stage_dataset_path)


if __name__ == "__main__":
    main()
    