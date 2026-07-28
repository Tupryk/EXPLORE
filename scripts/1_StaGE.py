import h5py
import hydra
from omegaconf import DictConfig

from explore.datasets.StaGE import StaGE


@hydra.main(version_base="1.3",
            config_path="../configs/yaml/StaGE",
            config_name="humanoid")
def main(cfg: DictConfig):
    
    file = h5py.File(cfg.configs_path, 'r')
    qpos = file["qpos"] if "qpos" in file.keys() else file["q"]
    ctrl = file["ctrl"]

    S = StaGE(qpos, ctrl, cfg.RRT)
    S.run()


if __name__ == "__main__":
    main()
