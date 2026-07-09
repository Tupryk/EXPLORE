import os
import time
import mujoco
import psutil
import pickle
import hnswlib
import numpy as np
from tqdm import tqdm
from tqdm import trange
from sklearn.neighbors import KDTree
from omegaconf import DictConfig, ListConfig

from explore.utils.mj import geom_names2ids
from explore.env.mujoco_warp_sim import MjSim


class StaGE_Node:
    def __init__(self,
                 parent: int,
                 t: float,
                 qpos: np.ndarray,
                 qvel: np.ndarray,
                 ctrl: np.ndarray,
                 manifold_phi: np.ndarray,
                 goal_phi: np.ndarray,
                 target_config_idx: int=-1):
        
        self.parent = parent
        self.t = t
        self.qpos = qpos.copy()
        self.qvel = qvel.copy()
        self.ctrl = ctrl.copy()
        self.manifold_phi = manifold_phi
        self.goal_phi = goal_phi
        self.target_config_idx = target_config_idx
        self.failed_expansion_count = 0

class StaGE:

    def __init__(self, configs: np.ndarray, configs_ctrl: np.ndarray, cfg: DictConfig):
        
        assert len(configs) == len(configs_ctrl)
        self.verbose = cfg.get("verbose", 0)
        self.output_dir = cfg.output_dir
        
        # Slice config arrays if too long
        self.max_configs = cfg.get("max_configs", -1)
        if self.max_configs != -1 and len(configs) > self.max_configs:
            self.manifold_qpos = configs[:self.max_configs]
            self.manifold_ctrl = configs_ctrl[:self.max_configs]
        else:
            self.manifold_qpos = configs
            self.manifold_ctrl = configs_ctrl
        
        self.manifold_size = self.manifold_qpos.shape[0]
        
        # Sim
        self.sim = MjSim(cfg.sim_interface)
        
        self.ctrl_dim = self.sim.mj_data.ctrl.shape[0]
        self.ctrl_ranges = self.sim.mj_model.actuator_ctrlrange
        self.state_dim = self.sim.mj_data.qpos.shape[0]
        
        self.sample_count = cfg.sim_interface.parallel_sims
        
        # State info
        self.q = cfg.q
        self.q_dot = cfg.q_dot
        self.q_obj_dot = cfg.q_obj_dot
        self.P = geom_names2ids(cfg.P, self.sim.mj_model)
        self.G = geom_names2ids(cfg.G, self.sim.mj_model)

        self.obs_pos_scale = cfg.get("obs_pos_scale", 1.0)
        self.obs_vel_scale = cfg.get("obs_vel_scale", 0.1)
        self.obs_ref_err_scale = cfg.get("obs_ref_err_scale", 10.0)

        self.q_weight = cfg.q_weight
        
        # Manifold embedings
        self.all_G_star = []
        self.phi_stable_configs = []
        
        for i in range(self.manifold_size):
            self.sim.mj_data.qpos[:] = self.manifold_qpos[i]
            mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

            q = self.sim.mj_data.qpos[self.q[0]:self.q[1]]
            G = self.sim.mj_data.geom_xpos[self.G, :].reshape(-1)
            phi = np.concatenate([q * self.q_weight, G])
            
            self.all_G_star.append(G)
            self.phi_stable_configs.append(phi)

        self.all_G_star = np.array(self.all_G_star)
        self.phi_stable_configs = np.array(self.phi_stable_configs)
        self.sds_manifold = KDTree(self.phi_stable_configs)

        # Environment specific variables
        self.min_cost = cfg.min_cost
        self.tau_action = cfg.tau_action
        self.stepsize = cfg.stepsize
        
        if isinstance(self.stepsize, ListConfig):
            self.stepsize = np.array(self.stepsize)
            
        self.target_min_dist = cfg.get("target_min_dist", 1000.0)
        self.max_expansions_per_tree = int(cfg.get("max_expansions_per_tree", 2500))
        
        # MPC
        self.action_sampler = lambda o, t: self.random_sample_ctrls(o, t)
        
        # StaGE params
        self.remove_expanded = cfg.get("remove_expanded", True)
        
        # Start states / tree roots
        self.start_ids = cfg.get("start_idx", -1)
        
        if not isinstance(self.start_ids, ListConfig):
            if self.start_ids == -1:
                self.start_ids = list(range(self.manifold_size))
            else:
                self.start_ids = [self.start_ids]
        
        self.start_ids = np.array(self.start_ids)
        
        # SDS
        self.ef_construction = cfg.get("ANN_ef_construction", 200)
        self.M = cfg.get("ANN_M", 16)

        if self.verbose:
            print(f"Starting search across {self.manifold_size} configs!")

    def init_tree(self, start_idx) -> list[ StaGE_Node]:
        
        tree: list[ StaGE_Node] = []
    
        root =  StaGE_Node(
            -1,
            0.0,
            self.manifold_qpos[start_idx],
            np.zeros((self.sim.data.qvel.shape[1],)),
            self.manifold_ctrl[start_idx],
            self.phi_stable_configs[start_idx],
            self.all_G_star[start_idx]
        )
        tree.append(root)
            
        return tree
    
    def store_tree(self, tree: list[StaGE_Node], folder_path: str, filename: str):
        dict_tree = []
        for node in tree:
            new_node = {
                "parent": node.parent,
                "t": node.t,
                "qpos": node.qpos,
                "qvel": node.qvel,
                "ctrl": node.ctrl,
                "manifold_phi": node.manifold_phi,
                "goal_phi": node.goal_phi,
                "target_config_idx": node.target_config_idx,
                "failed_expansion_count": node.failed_expansion_count
            }
            dict_tree.append(new_node)
            
        data_path = os.path.join(folder_path, f"{filename}.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(dict_tree, f)
        
    def run(self):
        
        tree_folder_path = os.path.join(self.output_dir, "trees")
        os.makedirs(tree_folder_path, exist_ok=True)
        
        start_time = time.time()

        for i, start_idx in tqdm(enumerate(self.start_ids), total=len(self.start_ids)):
            
            tree = self.init_tree(start_idx)
            self.sds_tree = hnswlib.Index(space="l2", dim=self.phi_stable_configs.shape[1])
            self.sds_tree.init_index(
                max_elements=self.max_expansions_per_tree * self.sample_count + 1,
                ef_construction=self.ef_construction,
                M=self.M
            )
            self.sds_tree.add_items(self.phi_stable_configs[start_idx].reshape(1, -1), ids=[0])

            if self.verbose > 1:
                pbar = trange(self.max_expansions_per_tree, desc=f"Tree {i+1}/{len(self.start_ids)}", unit="nodes")
            else:
                pbar = range(self.max_expansions_per_tree)
            
            for expansion_step_id in pbar:

                # Sample from manifold
                target_id = np.random.randint(self.manifold_size)
                
                # Pick closest node from the k nearest nodes
                ids, dists = self.sds_tree.knn_query(self.phi_stable_configs[target_id], k=1)
                parent_id = np.random.choice(ids[0])
                parent = tree[parent_id]
                if self.remove_expanded:
                    self.sds_tree.mark_deleted(parent_id)

                # Simulate random actions
                self.sim.setState(
                    np.array([parent.t]),
                    parent.qpos,
                    parent.qvel,
                    parent.ctrl
                )

                action = np.random.uniform(-1, 1, size=(self.sample_count, self.ctrl_dim))
                ctrl_np = self.sim.data.ctrl.numpy()
                ctrl_target = action * self.stepsize + ctrl_np
                
                self.sim.step(
                    self.tau_action,
                    ctrl_target
                )
                self.sim.gen_numpy_dict()
                
                # Add resulting nodes to tree
                new_phis = []
                
                q = self.sim.numpy_dict["qpos"][:, self.q[0]:self.q[1]]
                G = self.sim.numpy_dict["geom_xpos"][:, self.G, :].reshape(self.sample_count, -1)
                phi = np.concatenate([q * self.q_weight, G], axis=1)
                
                start_id = len(tree)
                for sim_i in range(self.sample_count):
                    new_node = StaGE_Node(
                        parent_id,
                        self.sim.numpy_dict["time"][sim_i],
                        self.sim.numpy_dict["qpos"][sim_i],
                        self.sim.numpy_dict["qvel"][sim_i],
                        self.sim.numpy_dict["ctrl"][sim_i],
                        phi[sim_i],
                        G[sim_i],
                        target_config_idx=target_id
                    )
                    tree.append(new_node)
                    new_phis.append(phi[sim_i])

                self.sds_tree.add_items(new_phis, ids=list(range(start_id, len(tree))))
                
                if self.verbose > 3 and (expansion_step_id + 1) % 100 == 0:
                    process = psutil.Process(os.getpid())
                    print(f"RSS (resident memory): {process.memory_info().rss / 1024**2:.2f} MB")
                    print(f"VMS (virtual memory): {process.memory_info().vms / 1024**2:.2f} MB")
            
            # Store information
            if self.verbose > 3:
                print(f"Storing tree {start_idx}")
            
            self.store_tree(tree, tree_folder_path, f"tree{start_idx}")
    
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.verbose > 1:
            print(f"Total time taken: {total_time:.2f} seconds")

        time_data_path = os.path.join(self.output_dir, "time_taken.txt")
        with open(time_data_path, "w") as f:
            f.write(f"{total_time}\n")
