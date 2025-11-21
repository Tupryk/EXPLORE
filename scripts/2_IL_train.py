import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from explore.utils.logger import get_logger
from explore.policies.flow import FlowPolicy
from explore.train.il_trainer import IL_Trainer
from explore.datasets.dataset import ExploreDataset
from explore.env.stable_configs_env import StableConfigsEnv


@hydra.main(version_base="1.3", config_path="../configs/yaml", config_name="IL_flow")
def main(cfg: DictConfig):
    logger = get_logger(cfg)

    dataset = ExploreDataset(cfg.data_dir, cfg.policy.horizon, cfg.policy.history)
    # TODO: Normalize data? -> normalizer = dataset.get_normalizer(); model.set_normalizer(normalizer); ema_model.set_normalizer(normalizer)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    actions, obs, cond = dataset[0]
    print("Action shape: ", actions.shape)
    print("Observation shape: ", obs.shape)
    print("Condition shape: ", cond.shape)
    action_dim = actions.shape[-1]
    obs_dim = obs.shape[-1]
    cond_dim = cond.shape[-1]
    
    policy = FlowPolicy(obs_dim, action_dim, cond_dim, cfg.policy)
    env = StableConfigsEnv(cfg.env)

    trainer = IL_Trainer(policy, loader, cfg.trainer, logger)
    trainer.train(epochs=cfg.epochs, env=env)

if __name__ == "__main__":
    main()
