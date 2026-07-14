import hydra
from omegaconf import DictConfig

from explore.utils.logger import get_logger


@hydra.main(
    version_base="1.3",
    config_path="../configs/yaml/Learned_StaGE",
    config_name="humanoid_box"
)
def main(cfg: DictConfig):
    logger = get_logger(cfg)

if __name__ == "__main__":
    main()
