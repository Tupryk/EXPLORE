import os
import pickle
import numpy as np
from tqdm import tqdm

from explore.env.mujoco_sim import MjSim
from explore.utils.vis import AdjMap, play_path


ERROR_THRESH = 1
mujoco_xml = "configs/twoFingers.xml"
dataset = "data/Pandas/17-34-44/trees"
mujoco_xml = "configs/franka_emika_panda/scene.xml"


tree_count = len(os.listdir(dataset))
print(f"Tree Count: {tree_count}")
trees: list[list[dict]] = []

for i in range(tree_count):
    data_path = os.path.join(dataset, f"tree_{i}.pkl")
    with open(data_path, "rb") as f:
        tree: list[dict] = pickle.load(f)
        trees.append(tree)

top_nodes = []
min_costs = []
for i in tqdm(range(tree_count)):

    tree_min_costs = [float("inf") for _ in range(tree_count)]
    tree_top_nodes = [-1 for _ in range(tree_count)]
    
    for n, node in enumerate(trees[i]):
        for j in range(tree_count):
            e = trees[j][0]["state"][1] - node["state"][1]
            node_cost = e.T @ e
            if node_cost < tree_min_costs[j]:
                tree_min_costs[j] = node_cost
                tree_top_nodes[j] = n
    
    top_nodes.append(tree_top_nodes)
    min_costs.append(tree_min_costs)

min_costs = np.array(min_costs)

AdjMap(min_costs, ERROR_THRESH, min_costs.max())

start_idx = 1
end_idx = 0
costs = [min_costs[start_idx][i] for i in range(tree_count)]
print(f"Mean costs for start config {start_idx}: {sum(costs)/tree_count}, (Min: {min(costs)}, Max {max(costs)})")
print(f"Cost for target {end_idx} with start {start_idx}: {costs[end_idx]}")

colors = [-1 for _ in range(tree_count)]

max_color_idx = 0
for i in range(tree_count):
        
    if colors[i] == -1:
        colors[i] = max_color_idx
        max_color_idx += 1
        
        for j in range(tree_count):
            if i != j and min_costs[i][j] <= ERROR_THRESH:
                if colors[j] != -1:
                    c = colors[j]
                    for k in range(tree_count):
                        if colors[k] == c:
                            colors[k] = colors[i]
                else:
                    colors[j] = colors[i]

groups = []
for c in colors:
    if not c in groups:
        groups.append(c)

group_sizes = [0 for _ in groups]
for c in colors:
    group_sizes[groups.index(c)] += 1

print(colors)
print(groups)
print("Group Count: ", len(groups))
print("Group Sizes: ", group_sizes)


look_at_specific_start_idx = 6
look_at_specific_end_idx = 1
top_paths_data = []
for i in range(tree_count):
    for j in range(tree_count):
        if look_at_specific_start_idx != -1 and i != look_at_specific_start_idx:
            continue
        if look_at_specific_end_idx != -1 and j != look_at_specific_end_idx:
            continue

        if (
            (look_at_specific_end_idx != -1 and look_at_specific_end_idx != -1) or
            (min_costs[i][j] < ERROR_THRESH and i != j)
            ):
            top_paths_data.append(
                ((i, j), top_nodes[i][j])
            )
top_paths = []
top_paths_start = []
top_paths_goal = []
for path_data in top_paths_data:

    start_idx = path_data[0][0]
    end_idx = path_data[0][1]
    if start_idx == end_idx: continue
    
    tree = trees[start_idx]
    
    node = tree[path_data[1]]
    path = []
    
    while True:
        path.append(node)
        if node["parent"] == -1: break
        node = tree[node["parent"]]
    
    path.reverse()
    assert path[0] == tree[0]

    top_paths.append(path)
    top_paths_start.append(start_idx)
    top_paths_goal.append(end_idx)

target_counts = []
for i, path in enumerate(top_paths):
    goal_idx = top_paths_goal[i]
    target_counts.append(0)
    for node in path:
        if goal_idx == node["target_config_idx"]:
            target_counts[-1] += 1

percs = [float(np.round(c/len(top_paths[i])*100)) for i, c in enumerate(target_counts)]
percs.sort()
percs.reverse()

possible_paths = tree_count**2 - tree_count
print("Found Trajectories Count: ", len(top_paths), " of ", possible_paths)
print("When considering full graph: ", sum([v**2 for v in group_sizes]) - tree_count, " of ", possible_paths)

if not len(top_paths):
    print("No trajectories found!")
    exit()

print("Percentage of reached config used as target: ", percs)
print("Avg. use of reached config as target: ", sum(percs)/len(percs))

path_idx = np.random.randint(0, len(top_paths))
path = top_paths[path_idx]

start_idx = top_paths_start[path_idx]
end_idx = top_paths_goal[path_idx]
print("Start idx: ", start_idx)
print("End idx: ", end_idx)
path_lens = [len(p) for p in top_paths]

# print("Path lens: [", end="")
# for i in range(len(top_paths)):
#     end = "]\n" if i == len(top_paths)-1 else ", "
#     print(f"{path_lens[i]} ({top_paths_start[i]}, {top_paths_goal[i]})", end=end)

path_lens.sort()
path_lens.reverse()
print("Avg. Path length: ", sum(path_lens)/len(path_lens))
print("Sampled Path Length: ", len(path))

start_state = trees[start_idx][0]["state"][1]
target_state = trees[end_idx][0]["state"][1]
sim = MjSim(mujoco_xml, tau_sim=0.01, view=True)
play_path(start_state, target_state, path, sim)
