import os
import hydra
import imageio
from stable_baselines3 import PPO
from omegaconf import DictConfig, OmegaConf

from explore.utils.logger import get_logger
from explore.env.stable_configs_env import StableConfigsEnv


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_policy")
def main(cfg: DictConfig):
    logger = get_logger(cfg)
    logger.info("Starting evaluation...")

    config_path = os.path.join(cfg.checkpoint_dir, ".hydra/config.yaml")
    train_cfg = OmegaConf.load(config_path)
    
    env = StableConfigsEnv(train_cfg.env)

    logger.info(f"Using device: {cfg.device}")
    
    model_path = os.path.join(cfg.checkpoint_dir, "trained_rl_policy.zip")
    model = PPO.load(model_path, env=env, device=cfg.device)

    obs, info = env.reset()
    logger.info(info)
    img = env.render()
    imgs = [img]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        img = env.render()
        imgs.append(img)

    result_path = os.path.join(cfg.output_dir, "result.gif")
    imageio.mimsave(result_path, imgs, fps=24)


if __name__ == "__main__":
    main()
