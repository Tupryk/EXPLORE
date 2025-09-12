import os
import hydra
import imageio
from omegaconf import DictConfig
from stable_baselines3 import PPO

from explore.utils.logger import get_logger
from explore.env.finger_balls_env import FingerBallsEnv


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_policy")
def main(cfg: DictConfig):
    logger = get_logger(cfg)
    logger.info("Starting evaluation...")

    env = FingerBallsEnv(cfg.env)
    model = PPO.load(cfg.checkpoint_dir, env=env)

    obs, _ = env.reset()
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
    imageio.mimsave(result_path, imgs, fps=10)


if __name__ == "__main__":
    main()
