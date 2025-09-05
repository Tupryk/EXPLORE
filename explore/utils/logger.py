import logging
from omegaconf import DictConfig


def get_logger(cfg: DictConfig) -> logging.Logger:
    logger = logging.getLogger(cfg.get("experiment_name", "default_experiment"))
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
