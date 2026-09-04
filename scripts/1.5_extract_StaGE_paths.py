import os
import h5py
import pickle
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import imageio
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import ListConfig
from sklearn.neighbors import KDTree

from explore.utils.mj import geom_names2ids
from explore.datasets.utils import build_path
# from explore.env.mujoco_warp_sim import MjSim
from explore.env.mujoco_threaded_sim import MjSim


def main():

    out_path = "outputs/2026-09-04/10-42-50"
    min_traj_time = 1.0
    horizon_same = 15
    
    config_path = os.path.join(out_path, ".hydra/config.yaml")
    gif_path = os.path.join(out_path, "path_gifs")
    traj_path = os.path.join(out_path, "trajs")
    goals_path = os.path.join(out_path, "goals")
    os.makedirs(gif_path, exist_ok=True)
    os.makedirs(traj_path, exist_ok=True)
    os.makedirs(goals_path, exist_ok=True)
    
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
        # phis = [node["manifold_phi"] for node in tree]
        # sds_tree = KDTree([p for p in phis if not np.any(np.isnan(p))])  # MuJoCo-Warp makes things NaN?
        # for i, p in enumerate(phis):
        #     if np.any(np.isnan(p)):
        #         print(f"Tree contains nan! Truncating to length {i} of {len(phis)}...")
        #         phis = phis[:i]
        #         break

        sds_tree = KDTree(phis)

        cfg.sim_interface.parallel_sims = 1
        cfg.sim_interface.verbose = 0
        sim = MjSim(cfg.sim_interface)
        G_ids = geom_names2ids(cfg.G, sim.mj_model)
        q_ids = cfg.q
        q_weight = cfg.q_weight

        all_G_star = []
        phi_stable_configs = []
        for i in range(manifold_size):
            sim.mj_data.qpos[:] = manifold_qpos[i]
            mujoco.mj_forward(sim.mj_model, sim.mj_data)

            q = sim.mj_data.qpos[q_ids[0]:q_ids[1]]
            G = sim.mj_data.geom_xpos[G_ids, :].reshape(-1)
            phi = np.concatenate([q * q_weight, G])
            
            all_G_star.append(G)
            phi_stable_configs.append(phi)
        
        reached_count = 0
        added_nodes = []
        # for end_id, manifold_point in tqdm(enumerate(phi_stable_configs), total=len(phi_stable_configs)):
        for end_id, manifold_point in tqdm(enumerate(all_G_star), total=len(all_G_star)):

            # Get all neighbors within min_cost, sorted nearest-first
            ind_arr, dist_arr = sds_tree.query_radius(
                [manifold_point],
                r=cfg.min_cost,
                return_distance=True,
                sort_results=True,
            )
            candidates = ind_arr[0]
            dists = dist_arr[0]

            if len(candidates) == 0:
                continue

            reached_count += 1

            for ind, dist in zip(candidates, dists):
                ind = int(ind)

                if tree[ind]["t"] <= min_traj_time or ind in added_nodes:
                    continue

                # Reconstruct path
                path = build_path(tree, ind)

                new_ids = [ind]
                new_ids.extend([node["parent"] for node in path[-(horizon_same - 1):]])
                if set(added_nodes) & set(new_ids):
                    continue

                break  # stop at first valid candidate
            else:
                # no candidate within radius passed the checks
                continue

            added_nodes.extend(new_ids)

            # Render gif
            goal_frame = sim.render_state(manifold_qpos[end_id])
            
            node = path[0]
            sim.setState(
                np.array([node["t"]]),
                node["qpos"],
                node["qvel"],
                node["ctrl"]
            )
            
            frames = []
            prev_ctrl = node["ctrl"]
            for node in path[1:]:
                fs = sim.step(
                    cfg.tau_action,
                    prev_ctrl + node["action"] * cfg.stepsize,
                    render=True
                )
                sim.setState(
                    np.array([node["t"]]),
                    node["qpos"],
                    node["qvel"],
                    node["ctrl"],
                    reset_frame_time=False
                )
                frames.extend(fs)
                prev_ctrl = node["ctrl"]

            # Save gif
            ratio = 0.4
            frames = [(frame.astype(float)*(1.-ratio) + goal_frame.astype(float)*ratio).astype(frame.dtype) for frame in frames]
            name = f"{start_id}_to_{end_id}_len_{path[-1]["t"]:.2f}s({len(path)})"
            imageio.mimsave(os.path.join(gif_path, f"{name}.gif"), frames, fps=24, loop=0)

            # Save traj and goal
            data_path = os.path.join(traj_path, f"{name}.pkl")
            with open(data_path, "wb") as f:
                pickle.dump(path, f)

            data_path = os.path.join(goals_path, f"{name}.pkl")
            with open(data_path, "wb") as f:
                pickle.dump(manifold_point, f)
                
        print(f"{((reached_count/manifold_size)*100):.2f}% Coverage. ({reached_count} states reached)")


if __name__ == "__main__":
    main()
