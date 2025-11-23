import os
import torch
import imageio
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt


def eval_il_policy(policy : nn.Module, env: gym.Env, history: int=1, horizon: int=1,
                   save_path: str="", eval_count: int=1) -> list[float]:

    policy.eval()
    all_rewards = []
    for i in range(eval_count):
        
        options = {"eval_view": "fixed_cam"}
        new_obs, init_info = env.reset(options=options)

        imgs = []
        done = False
        obs = torch.from_numpy(new_obs)
        obs = obs.expand(1, history, obs.shape[0])
        
        episode_rewards = []
        while not done:
            
            new_obs = torch.from_numpy(new_obs)
            history = obs.shape[1]
            shifted = obs.clone()
            shifted[:, :history-1, :] = obs[:, 1:, :]
            shifted[:, -1, :] = new_obs
            obs = shifted.float()

            obs_in = obs[:, :, :-policy.cond_dim].clone()
            goal_cond = obs[:, 0, -policy.cond_dim:].clone()

            actions = policy(obs_in.to(policy.device), goal_cond.to(policy.device))["pred"]
            actions = actions.detach().cpu().numpy()[0]
            
            for a in actions:
                new_obs, reward, terminated, truncated, info = env.step(a)
                
                frames = info["frames"]
                imgs.extend(frames)
                done = terminated or truncated
                episode_rewards.append(reward)
                
                print(f"Iter: {env.iter}; Reward: {env.reward:.4f} (Goal Reward: {info['goal_reward']:.4f} Guiding Reward: {info['guiding_reward']:.4f})")
        
        all_rewards.append(episode_rewards)

        if save_path:
            result_path = os.path.join(save_path, f"result{i}.gif")
            imageio.mimsave(result_path, imgs, fps=24, loop=0)

            im_start = env.render("fixed_cam", config_idx=init_info["start_config_idx"])
            im_end = env.render("fixed_cam", config_idx=init_info["end_config_idx"])
            im_reached = env.render("fixed_cam")

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
            plt.savefig(os.path.join(save_path, f"intro{i}.png"))
            plt.close('all')

    return all_rewards
    