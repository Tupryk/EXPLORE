import os
from omegaconf import OmegaConf

from explore.datasets.dataset import ExploreDataset


cfg = OmegaConf.load("./configs/yaml/IL_flow.yaml")
dataset = ExploreDataset(cfg.data_dir, cfg.policy.horizon, cfg.policy.history)

dataset_len = len(dataset)
print("Dataset size: ", dataset_len)

for i in range(dataset_len):
    action, state, goal_cond = dataset[i]

    print(f"---------------- {i} ----------------")
    print("State:\n", state)
    print("Action:\n", action)
