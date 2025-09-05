import hydra
from omegaconf import DictConfig

from explore.train.trainer import Trainer
from explore.utils.logger import get_logger


@hydra.main(config_path="../configs", config_name="guided_RL")
def main(cfg: DictConfig):
    logger = get_logger(cfg)
    trainer = Trainer(cfg, logger)
    trainer.train()

if __name__ == "__main__":
    main()
