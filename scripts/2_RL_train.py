import hydra
from omegaconf import DictConfig

from explore.train.rl_trainer import RL_Trainer
from explore.utils.logger import get_logger


@hydra.main(version_base="1.3", config_path="../configs", config_name="guided_RL")
def main(cfg: DictConfig):
    logger = get_logger(cfg)
    trainer = RL_Trainer(cfg, logger)
    trainer.train()

if __name__ == "__main__":
    main()
