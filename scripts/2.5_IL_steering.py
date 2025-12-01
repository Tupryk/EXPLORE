import os
import hydra
import torch
from stable_baselines3 import SAC
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback

from explore.utils.logger import get_logger
from explore.policies.flow import FlowPolicy
from explore.datasets.dataset import ExploreDataset
from explore.env.stable_configs_env import StableConfigsEnv
from explore.env.flow_policy_wrapper import FlowPolicyEnvWrapper


@hydra.main(version_base="1.3", config_path="../configs/yaml", config_name="IL_steering")
def main(cfg: DictConfig):
    logger = get_logger(cfg)
    
    # Load flow policy
    train_config_path = os.path.join(cfg.training_dir, ".hydra/config.yaml")
    train_cfg = OmegaConf.load(train_config_path)

    dataset = ExploreDataset(
        train_cfg.data_dir,
        horizon=train_cfg.policy.horizon,
        history=train_cfg.policy.history
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
        device=cfg.flow_device,
        action_normalizer=dataset.action_normalizer,
        state_normalizer=dataset.state_normalizer
    )
    state_dict = torch.load(
        os.path.join(cfg.flow_policy_dir, cfg.flow_policy_checkpoint),
        map_location=cfg.flow_device
    )
    policy.load_state_dict(state_dict)
    
    # Load environment with flow policy
    env = StableConfigsEnv(cfg.env)
    env = FlowPolicyEnvWrapper(env, policy)
    
    # Setup and run DSRL
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]

    net_arch = []
    for _ in range(cfg.train.num_layers):
        net_arch.append(cfg.train.layer_size)
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.train.actor_lr,
        buffer_size=20000000,                                                           # Replay buffer size
        learning_starts=1,                                                              # How many steps before learning starts (total steps for all env combined)
        batch_size=cfg.train.batch_size,
        tau=cfg.train.tau,                                                              # Target network update rate
        gamma=cfg.train.discount,                                                       # Discount factor
        train_freq=cfg.train.train_freq,                                                # Update the model every train_freq steps
        gradient_steps=cfg.train.utd,                                                   # How many gradient steps to do at each update
        action_noise=None,                                                              # No additional action noise
        optimize_memory_usage=False,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,            # Automatic entropy tuning
        target_update_interval=1,                                                       # Update target network every interval
        target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,  # Automatic target entropy
        use_sde=False,
        sde_sample_freq=-1,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_freq,
        save_path=cfg.output_dir,
        name_prefix="DSRL"
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback = checkpoint_callback
    )
    save_path = os.path.join(cfg.output_dir, "final_rl_policy")
    model.save(save_path)
    logger.info(f"Model saved as {save_path}")
    
    # Eval


if __name__ == "__main__":
    main()
