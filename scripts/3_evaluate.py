import os
import hydra
import imageio
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from omegaconf import DictConfig, OmegaConf

from explore.utils.vis import play_path
from explore.utils.logger import get_logger
from explore.env.stable_configs_env import StableConfigsEnv


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_policy")
def main(cfg: DictConfig):
    logger = get_logger(cfg)
    logger.info("Starting evaluation...")

    config_path = os.path.join(cfg.policy_dir, ".hydra/config.yaml")
    train_cfg = OmegaConf.load(config_path)
    
    env = StableConfigsEnv(train_cfg.env)

    logger.info(f"Using device: {cfg.device}")
    
    model_path = os.path.join(cfg.policy_dir, f"{cfg.checkpoint}.zip")
    if train_cfg.rl_method == "PPO":
        model = PPO.load(model_path, env=env, device=cfg.device)

    elif train_cfg.rl_method == "SAC":
        model = SAC.load(model_path, env=env, device=cfg.device)
    
    else:
        raise Exception(f"RL method '{train_cfg.rl_method}' not available.")

    for i in range(cfg.eval_count):
        
        if cfg.start_idx != -1 and cfg.end_idx != -1:
            options = {"traj_pair": (cfg.start_idx, cfg.end_idx), "eval_view": cfg.eval_view, "no_exist_fine": True}
        else:
            options = {"eval_view": cfg.eval_view}

        obs, init_info = env.reset(options=options)
        logger.info(init_info)

        imgs = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames = info["frames"]
            imgs.extend(frames)
            done = terminated or truncated
            print(f"Iter: {env.iter}; Reward: {env.reward:.4f} (Goal Reward: {info['goal_reward']:.4f} Guiding Reward: {info['guiding_reward']:.4f})")

        # Save results
        result_path = os.path.join(cfg.output_dir, f"result{i}.gif")
        imageio.mimsave(result_path, imgs, fps=24, loop=0)

        im_start = env.render(cfg.eval_view, config_idx=init_info["start_config_idx"])
        im_end = env.render(cfg.eval_view, config_idx=init_info["end_config_idx"])
        im_reached = env.render(cfg.eval_view)

        fig, axes = plt.subplots(1, 3, figsize=(30, 20))
        axes[0].set_title("Start Config", fontsize=24, fontweight="bold")
        axes[0].imshow(im_start)
        axes[0].axis("off")
        axes[1].set_title("Target Config", fontsize=24, fontweight="bold")
        axes[1].imshow(im_end)
        axes[1].axis("off")
        axes[2].set_title("Reached Config", fontsize=24, fontweight="bold")
        axes[2].imshow(im_reached)
        axes[2].axis("off")
        plt.tight_layout()    
        plt.savefig(os.path.join(cfg.output_dir, f"intro{i}.png"))
        
        # TODO: Save full guiding path gif
        # if train_cfg.env.guiding:
        #     plt.imsave(os.path.join(cfg.output_dir, "end_config.png"), img)
        #     play_path(path, sim, save_intro_as=, tau_action=sim_cfg.tau_action, camera=cfg.RRT.sim.camera)

        if cfg.start_idx != -1 and cfg.end_idx != -1:
            break


if __name__ == "__main__":
    main()
