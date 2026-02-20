import h5py
import hydra
from omegaconf import DictConfig

# from explore.datasets.generator_closest_targets import Search
from explore.datasets.generator_closest_targets_gpu import Search
import numpy as np

@hydra.main(version_base="1.3",
            config_path="../configs/yaml",
            config_name="finger_ramp_gen.yaml")
def main(cfg: DictConfig):
    
    file = h5py.File(cfg.configs_path, 'r')

    S = Search(file["qpos"], file["ctrl"], cfg.RRT)
    S.jit_simulator()

    reached_trajs = {}
    for i in range(26):
        for j in range(5):
            if i == j:
                continue
            elif i==1:
                traj, reached_goal = S.run_mppi_baseline(1, j)
                reached_trajs[f"target_{i}_{j}"] = (traj, reached_goal, i, j)

    with h5py.File('results4.h5', 'w') as f:
        for key, (traj, reached, start_idx, end_idx) in reached_trajs.items():
            group = f.create_group(key)
            
            group.create_dataset('trajectory', data=traj, compression="gzip")
            group.create_dataset('reached_goal', data=np.int8(reached))
            group.create_dataset('start_idx', data=np.int8(start_idx))
            group.create_dataset('end_idx', data=np.int8(end_idx))
            
            # Optional: Keep the attribute too (best practice for metadata)
            group.attrs['description'] = "MPPI baseline result"
    #S.run()


if __name__ == "__main__":
    main()
