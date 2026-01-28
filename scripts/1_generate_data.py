import h5py
import hydra
from omegaconf import DictConfig

from explore.datasets.generator_closest_targets import Search


@hydra.main(version_base="1.3",
            config_path="../configs/yaml",
            config_name="trajectory_generation")
def main(cfg: DictConfig):
    
    file = h5py.File(cfg.configs_path, 'r')

    S = Search(file["qpos"], file["ctrl"], cfg.RRT)

    S.run()


if __name__ == "__main__":
    main()
