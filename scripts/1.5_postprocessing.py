import os
import hydra
import pickle
import imageio
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from explore.utils.vis import play_path
from explore.env.mujoco_sim import MjSim
from explore.datasets.post_processing import extract_all_paths
from explore.datasets.utils import load_trees, generate_adj_map


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="postprocessing")
def main(cfg: DictConfig):

    config_path = os.path.join(cfg.data_dir, ".hydra/config.yaml")
    trees_cfg = OmegaConf.load(config_path)

    tree_dir = os.path.join(cfg.data_dir, "trees")
    trees, _, _ = load_trees(tree_dir)

    q_mask = np.array(trees_cfg.RRT.q_mask)
    min_costs, top_nodes = generate_adj_map(trees, q_mask, check_cached=cfg.data_dir)

    # Find connections between trees
    traj_pairs, paths, connect_mat = extract_all_paths(trees, min_costs, top_nodes, trees_cfg.RRT.min_cost, cfg.horizon)
    print(f"Total paths found before glueing: {len(traj_pairs)}")

    path_idx = np.argmax([len(gn) for gn in paths])
    start_idx, end_idx = traj_pairs[path_idx]
    print(f"Connections from {start_idx} to {end_idx}: ", connect_mat[start_idx][end_idx])

    frames = []
    sim_cfg = trees_cfg.RRT.sim
    for i, sub_path in enumerate(paths[path_idx]):
        
        si = connect_mat[start_idx][end_idx][i-1] if i != 0 else start_idx
        ei = connect_mat[start_idx][end_idx][i]
        start_state = trees[si][0]["state"][1]
        target_state = trees[ei][-1]["state"][1]

        sim = MjSim(
            sim_cfg.mujoco_xml, tau_sim=sim_cfg.tau_sim, interpolate=sim_cfg.interpolate_actions,
            joints_are_same_as_ctrl=sim_cfg.joints_are_same_as_ctrl, view=False
        )
        fs = play_path(
            sub_path, sim, start_state, target_state, save_intro_as=os.path.join(cfg.output_dir, f"intro{i}.png"),
            tau_action=sim_cfg.tau_action, camera=sim_cfg.camera, save_as=os.path.join(cfg.output_dir, f"path{i}.gif"), reset_state=True
        )
        frames.extend(fs)
    
    print("Frame count: ", len(frames))
    sampled_traj_dir = os.path.join(cfg.output_dir, "path.gif")
    imageio.mimsave(sampled_traj_dir, frames, fps=24, loop=0)

    # TODO: Glue paths through BBO
    # glued_paths = []
    # for path in tqdm(paths):

    paths_data_dir = os.path.join(cfg.output_dir, "paths_data.pkl")
    with open(paths_data_dir, "wb") as f:
        pickle.dump((traj_pairs, paths), f)


if __name__ == "__main__":
    main()
