import h5py
import hydra
from omegaconf import DictConfig

from explore.datasets.generator import Search


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="trajectory_generation")
def main(cfg: DictConfig):
    
    file = h5py.File(cfg.configs_path, 'r')
    stable_configs = file["positions"]

    S = Search(cfg.mujoco_xml, stable_configs, cfg.RRT)

    S.run()


if __name__ == "__main__":
    main()
