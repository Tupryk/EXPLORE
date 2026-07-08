import os
import h5py
import pickle
import mujoco
import imageio
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import ListConfig
from sklearn.neighbors import KDTree

from explore.utils.mj import geom_names2ids
from explore.env.mujoco_warp_sim import MjSim
from explore.datasets.utils import build_path


def main():
    out_path = "outputs/2026-07-06/11-05-28"
    
    config_path = os.path.join(out_path, ".hydra/config.yaml")
    gif_path = os.path.join(out_path, "path_gifs")
    os.makedirs(gif_path, exist_ok=True)
    
    cfg = OmegaConf.load(config_path)
    
    file = h5py.File(cfg.configs_path, 'r')
    manifold_qpos = file["qpos"] if "qpos" in file.keys() else file["q"]
    manifold_size = manifold_qpos.shape[0]
    
    cfg = cfg.RRT
    
    # Start states / tree roots
    start_ids = cfg.get("start_idx", -1)
    
    if not isinstance(start_ids, ListConfig) and not isinstance(start_ids, list):
        if start_ids == -1:
            start_ids = list(range(manifold_size))
        else:
            start_ids = [start_ids]
    
    for start_id in start_ids:

        print(f"Analizing tree {start_id}...")
        tree_path = os.path.join(out_path, f"trees/tree{start_id}.pkl")
        
        with open(tree_path, "rb") as f:
            tree: list[dict] = pickle.load(f)
        
        phis = [node["goal_phi"] for node in tree]
        sds_tree = KDTree([p for p in phis if not np.any(np.isnan(p))])

        cfg.sim_interface.parallel_sims = 1
        sim = MjSim(cfg.sim_interface)
        G_ids = geom_names2ids(cfg.G, sim.mj_model)

        all_G_star = []
        
        for i in range(manifold_size):
            sim.mj_data.qpos[:] = manifold_qpos[i]
            mujoco.mj_forward(sim.mj_model, sim.mj_data)

            G = sim.mj_data.geom_xpos[G_ids, :].reshape(-1)
            all_G_star.append(G)
        
        reached_count = 0
        for i, manifold_point in tqdm(enumerate(all_G_star), total=len(all_G_star)):
            
            dist, ind = sds_tree.query([manifold_point], k=1)
            dist = dist[0][0]
            ind = ind[0][0]
            
            if dist < cfg.min_cost:
                reached_count += 1
                
                if tree[ind]["t"] > 1.:
                    # Reconstruct path
                    path = build_path(tree, ind)

                    # Render gif
                    goal_frame = sim.render_state(manifold_qpos[i])
                    
                    node = path[0]
                    sim.setState(
                        np.array([node["t"]]),
                        node["qpos"],
                        node["qvel"],
                        node["ctrl"]
                    )
                    
                    frames = []
                    for node in path[1:]:
                        fs = sim.step(
                            cfg.tau_action,
                            node["ctrl"],
                            render=True
                        )
                        frames.extend(fs)
                    
                    frames = [(frame.astype(float)*0.8 + goal_frame.astype(float)*0.2).astype(frame.dtype) for frame in frames]
                    imageio.mimsave(os.path.join(gif_path, f"{start_id}_to_{i}.gif"), frames, fps=24, loop=0)
                    
        print(f"{((reached_count/manifold_size)*100):.2f}% Coverage.")


if __name__ == "__main__":
    main()
