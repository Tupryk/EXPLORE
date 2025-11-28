import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from explore.utils.logger import get_logger
from explore.env.utils import eval_il_policy
from explore.policies.flow import FlowPolicy
from explore.datasets.dataset import ExploreDataset
from explore.env.stable_configs_env import StableConfigsEnv


@hydra.main(version_base="1.3", config_path="../configs/yaml", config_name="eval_IL_policy")
def main(cfg: DictConfig):
    logger = get_logger(cfg)

    train_config_path = os.path.join(cfg.training_dir, ".hydra/config.yaml")
    train_cfg = OmegaConf.load(train_config_path)

    dataset = ExploreDataset(
        train_cfg.data_dir,
        horizon=train_cfg.policy.horizon,
        history=train_cfg.policy.history,
        min_path_len=train_cfg.policy.horizon,
        start_idx=train_cfg.env.start_config_idx,
        end_idx=train_cfg.env.target_config_idx,
        tau_action=train_cfg.data_tau_action
    )

    actions, obs, cond = dataset[0]
    print("Action shape: ", actions.shape)
    print("Observation shape: ", obs.shape)
    print("Condition shape: ", cond.shape)
    action_dim = actions.shape[-1]
    obs_dim = obs.shape[-1]
    cond_dim = cond.shape[-1]

    policy = FlowPolicy(
        obs_dim,
        action_dim,
        cond_dim,
        train_cfg.policy,
        device=cfg.device,
        action_normalizer=dataset.action_normalizer,
        state_normalizer=dataset.state_normalizer
    )
    state_dict = torch.load(os.path.join(cfg.training_dir, cfg.checkpoint), map_location="cpu")
    policy.load_state_dict(state_dict)

    env = StableConfigsEnv(train_cfg.env)

    eval_il_policy(
        policy, env,
        save_path=cfg.output_dir,
        eval_count=cfg.eval_count,
        history=policy.history,
        substates=dataset.sub_states
    )

if __name__ == "__main__":
    main()
