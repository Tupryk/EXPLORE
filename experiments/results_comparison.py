import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from explore.utils.vis import play_path
from explore.env.mujoco_sim import MjSim


explorer_paths = {
    "rs": "data/Pandas/rs",
    "cma": "data/Pandas/cma"
}

start_idx = 6
end_idx = 1

trees = {}
for k, v in explorer_paths.items():
    
    tree = []
    data_path = os.path.join(v, f"trees/tree_{start_idx}.pkl")
    with open(data_path, "rb") as f:
        tree: list[dict] = pickle.load(f)
    
    data_path = os.path.join(v, f"trees/tree_{end_idx}.pkl")
    with open(data_path, "rb") as f:
        target_state: np.ndarray = pickle.load(f)[0]["state"][1]
    
    data_path = os.path.join(v, "time_taken.txt")
    with open(data_path, "r") as f:
        s = f.read().strip()
    time_taken = float(s)
    
    best_node_idx = -1
    best_cost = np.inf
    for i, node in enumerate(tree):
        e = target_state - node["state"][1]
        cost = e.T @ e
        if best_cost > cost:
            best_node_idx = i
            best_cost = cost
            
    node = tree[best_node_idx]
    path = []
    while True:
        path.append(node)
        if node["parent"] == -1: break
        node = tree[node["parent"]]
    path.reverse()
    assert path[0] == tree[0]

    trees[k] = {
        "tree": tree,
        "time": time_taken,
        "cost": best_cost,
        "path": path
    }

# start_state = trees["rs"]["path"][0]["state"][1]
# sim = MjSim("configs/franka_emika_panda/scene.xml", view=True)
# play_path(start_state, target_state, trees["rs"]["path"], sim, playback_time=10)
# play_path(start_state, target_state, trees["cma"]["path"], sim, playback_time=10, play_intro=False)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
methods = trees.keys()
costs = [float(trees[k]["cost"]) for k in methods]
times = [trees[k]["time"] for k in methods]

ax[0].bar(methods, costs)
ax[1].bar(methods, times)
ax[0].set_ylabel("Cost")
ax[1].set_ylabel("Time")
plt.show()
