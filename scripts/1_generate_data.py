import hydra
from omegaconf import DictConfig

from explore.datasets.generator import Search
from explore.datasets.rnd_configs import RndConfigs


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="trajectory_generation")
def main(cfg: DictConfig):
    
    D = RndConfigs("configs/rnd_twoFingers.h5")

    S = Search(D.positions, cfg.RRT)

    S.run(display=0.)


if __name__ == "__main__":
    main()
