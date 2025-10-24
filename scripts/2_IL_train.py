import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from explore.models.flow import FlowNet
from explore.utils.logger import get_logger
from explore.train.il_trainer import IL_Trainer


@hydra.main(version_base="1.3", config_path="../configs", config_name="IL_flow")
def main(cfg: DictConfig):
    logger = get_logger(cfg)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    model = FlowNet(cfg.model)

    trainer = IL_Trainer(model, loader, cfg.trainer)
    trainer.train(epochs=cfg.epochs)

if __name__ == "__main__":
    main()
