import os
import datetime
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, ListConfig
import matplotlib.pyplot as plt
import json 
from pathlib import Path

from explore.datasets.utils import cost_computation, load_trees, compute_hausdorff, compute_coverage_number_paths, compute_path_entropy, compute_paths

COMPUTE_HAUSDORFF = False
COMPUTE_ENTROPY = False
COMPUTE_COVERAGE = True

root_folder = "outputs/PandasBoxResults"

for item in os.listdir(root_folder):
    full_path = os.path.join(root_folder, item)
    if os.path.isdir(full_path):
        dataset_paths = os.path.join(root_folder, item)
        
        # Detect if it's a multirun or a single run
        is_multi_run = not Path(dataset_paths + "/.hydra").is_dir()
        
        sub_items = os.listdir(dataset_paths)
        folders = [f for f in sub_items if f.isdigit()]
        ds_c = len(folders) if is_multi_run else 1

        # loop over folders (0, 1, 2, 3 ...) in multirun
        for folder_idx in range(ds_c):
            all_mean_costs = []
            all_found_paths = []
            all_hausdorffs = []
            all_hd_implicit = []
            all_entropies = []
            all_coverage = []
            all_n_paths = []
            
            best_found_paths_count = 0
            best_folder_name = ""
            final_cfg = None # To save metadata for the final JSON

            folder_name = str(folder_idx) if is_multi_run else "."
            dataset = f"{dataset_paths}/{folder_name}"
            
            if not os.path.isdir(dataset):
                continue

            print(f"Processing folder: {dataset}")
            config_path = os.path.join(dataset, ".hydra/config.yaml")
            cfg = OmegaConf.load(config_path)
            final_cfg = cfg # Keep the last one for shared params
            
            start_ids = cfg.RRT.start_idx
            if not isinstance(start_ids, (list, ListConfig)):
                start_ids = [start_ids]

            # Determine Environment and Ablation
            ENV = "unknown"
            if "fingerRamp" in cfg.RRT.sim.mujoco_xml: ENV = "fingerRamp"
            elif "twoFingers" in cfg.RRT.sim.mujoco_xml: ENV = "fingersBox"
            elif "panda_single" in cfg.RRT.sim.mujoco_xml: ENV = "pandaHook"
            elif "pandas_table" in cfg.RRT.sim.mujoco_xml: ENV = "pandasBox"

            ABLATION_TYPE = "unknown"
            if is_multi_run:
                config_path_multi = os.path.join(dataset, "../multirun.yaml")
                cfg_multi = OmegaConf.load(config_path_multi)            
                if "RRT.n_best_actions" in cfg_multi.hydra.sweeper.params:
                    ABLATION_TYPE = "n_best_actions"
                elif "RRT.knnK" in cfg_multi.hydra.sweeper.params:
                    ABLATION_TYPE = "knnK"
            else:
                if cfg.RRT.disable_node_max_strikes == 1 and cfg.RRT.knnK != 1 and cfg.RRT.sample_uniform_prob == 0 and cfg.RRT.n_best_actions != 1: ABLATION_TYPE = "baseline"
                elif cfg.RRT.n_best_actions == 1: ABLATION_TYPE = "nonbest"
                elif cfg.RRT.knnK == 1: ABLATION_TYPE = "noknnk"
                elif cfg.RRT.sample_uniform_prob != 0: ABLATION_TYPE = "uniform"
                elif cfg.RRT.disable_node_max_strikes == -1: ABLATION_TYPE = "disable_node"

            print(f"Environment: {ENV}, Ablation Type: {ABLATION_TYPE}, Start IDs: {start_ids}")

            # --- 3. LOOP OVER START_IDS ---
            for idx in start_ids[:1]:
                ERROR_THRESH = cfg.RRT.min_cost
                cost_max_method = False
                cutoff = -1
                q_mask = np.array(cfg.RRT.q_mask)

                tree_dataset = os.path.join(dataset, "trees")
                trees, tree_count, total_nodes_count = load_trees(tree_dataset, cutoff, verbose=0)

                if q_mask.size == 0:
                    q_mask = np.ones_like(trees[0][0]["state"][1])

                costs_over_time = []
                # Compute costs for each node added to the tree
                for i_node, node in tqdm(enumerate(trees[idx]), total=len(trees[idx]), desc=f"Tree {idx}"):
                    costs = []
                    for target_idx in range(tree_count):
                        cost = cost_computation(trees[target_idx][0], node, q_mask, cost_max_method)
                        # Keep minimum cost found so far for this target
                        if i_node == 0 or costs_over_time[i_node-1][target_idx] > cost:
                            costs.append(cost)
                        else:
                            costs.append(costs_over_time[i_node-1][target_idx])
                    costs_over_time.append(costs)

                # Summary for this tree
                mean_cost_ot = [sum(c)/(len(c)-1) for c in costs_over_time]
                found_paths_ot = [len([1 for val in c if val < ERROR_THRESH])-1 for c in costs_over_time]

                # Store in global list
                all_mean_costs.append(mean_cost_ot)
                all_found_paths.append(found_paths_ot)

                if found_paths_ot[-1] > best_found_paths_count:
                    best_found_paths_count = found_paths_ot[-1]
                    best_folder_name = dataset

                # Metrics

                if COMPUTE_HAUSDORFF or COMPUTE_COVERAGE or COMPUTE_ENTROPY:
                    path_counts, end_nodes = compute_paths(trees, tree_count, q_mask, cost_max_method, ERROR_THRESH)
                hd, hd_imp = (compute_hausdorff(path_counts) 
                              if COMPUTE_HAUSDORFF else (0, 0))
                cov, n_p = (compute_coverage_number_paths(path_counts, start_idx=idx) 
                            if COMPUTE_COVERAGE else (0, 0))
                ent = compute_path_entropy(path_counts, end_notes, ERROR_THRESH) if COMPUTE_ENTROPY else 0

                all_hausdorffs.append(hd)
                all_hd_implicit.append(hd_imp)
                all_coverage.append(cov)
                all_n_paths.append(n_p)
                all_entropies.append(ent)

            # --- CALCULATE MEANS FOR THIS SUBFOLDER ---
            if not all_mean_costs:
                continue

            min_len = min(len(x) for x in all_mean_costs)
            all_mean_costs_trimmed = np.array([x[:min_len] for x in all_mean_costs])
            all_found_paths_trimmed = np.array([x[:min_len] for x in all_found_paths])

            avg_cost = np.mean(all_mean_costs_trimmed, axis=0)
            std_cost = np.std(all_mean_costs_trimmed, axis=0)
            avg_paths = np.mean(all_found_paths_trimmed, axis=0)
            std_paths = np.std(all_found_paths_trimmed, axis=0)

            # --- YOUR ORIGINAL PLOTTING STYLE ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Average Cost
            axes[0].set_title(f"Global Average Cost Over Time\n {ENV} Cost method: {cfg.RRT.cost_method} knnK: {cfg.RRT.knnK}, n_best_actions: {cfg.RRT.n_best_actions}", fontweight='bold')
            axes[0].plot(avg_cost, label="Global Mean Cost", color='blue', linewidth=2)
            axes[0].fill_between(range(min_len), avg_cost - std_cost, avg_cost + std_cost, alpha=0.15, color='blue')
            axes[0].axhline(y=ERROR_THRESH, color="red", linestyle="--", label="Success Threshold")
            axes[0].set_xlabel("Iteration/Node")
            axes[0].set_ylabel("Cost")
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='y', which='both', right=True, labelright=True)
            axes[0].legend(loc='upper right')

            # Plot 2: Average Found Paths
            axes[1].set_title(f"Global Average Found Paths Over Time\n {ENV} Cost method:{cfg.RRT.cost_method} knnK: {cfg.RRT.knnK} n_best_actions: {cfg.RRT.n_best_actions}", fontweight='bold')
            axes[1].plot(avg_paths, color='green', linewidth=2)
            axes[1].fill_between(range(min_len), avg_paths - std_paths, avg_paths + std_paths, alpha=0.15, color='green')
            axes[1].set_xlabel("Iteration/Node")
            axes[1].set_ylabel("Count")
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='y', which='both', right=True, labelright=True)

            if ENV == "fingersBox":
                axes[0].set_ylim(0, 2); axes[1].set_ylim(0, 30)
            elif ENV == "fingerRamp":
                axes[0].set_ylim(0, .3); axes[1].set_ylim(0, 25)
            elif ENV == "pandasBox":
                pass
            elif ENV == "pandaHook":
                pass

            # --- SAVING PER FOLDER ---
            save_dir = f"/home/denis/Desktop/{ENV}/{ABLATION_TYPE}"
            os.makedirs(save_dir, exist_ok=True)
            # Use folder_idx to name the file (0.png, 1.png, etc.)
            save_path_base = os.path.join(save_dir, str(folder_idx))

            results_data = {
                "n_best_actions": cfg.RRT.n_best_actions,
                "knnK": cfg.RRT.knnK,
                "cost_method": cfg.RRT.cost_method,
                "avg_final_successes": float(avg_paths[-1]),
                "best_experiment_folder": best_folder_name,
                "best_found_paths_count": int(best_found_paths_count),
                "avg_hausdorff": float(np.mean(all_hausdorffs)),
                "avg_hausdorff_implicit": float(np.mean(all_hd_implicit)),
                "avg_coverage": float(np.mean(all_coverage)),
                "avg_entropy": float(np.mean(all_entropies)),
                "total_found_paths": int(sum(all_n_paths))
            }

            with open(f"{save_path_base}.json", "w") as f:
                json.dump(results_data, f, indent=4)

            plt.tight_layout()
            plt.savefig(f"{save_path_base}.png")
            plt.close(fig)

            print(f"Subfolder {folder_idx} processed and saved to {save_path_base}")