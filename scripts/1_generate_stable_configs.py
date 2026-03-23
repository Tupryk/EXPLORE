import h5py
import hydra
from omegaconf import DictConfig

from explore.datasets.gen_stable_configs import GenStableConfigs


@hydra.main(version_base="1.3",
            config_path="../configs/yaml",
            config_name="gen_stable_configs")
def main(cfg: DictConfig):
    
    S = GenStableConfigs(cfg)
    S.run()


if __name__ == "__main__":
    main()
