import math
import hydra
from omegaconf import DictConfig

from explore.env.SGRL import SGRL
from explore.models.TD7.TD7 import TD7
from explore.datasets.workspace import Workspace


@hydra.main(version_base="1.3", config_path="../configs/yaml", config_name="guided_RL")
def main(cfg: DictConfig):
    #-- create env
    W = Workspace(cfg)
    env = W.create_Gym()

    gamma = math.pow(0.5, env.cfg.tau_step/W.cfg.SGRL.discount_half_life)
    print("-- tag:", cfg.tag)
    print("-- discount gamma:", gamma)

    #-- create RL method
    cfg.TD7.tag = cfg.tag
    cfg.TD7.discount = gamma
    rl_method = TD7(env, cfg.TD7)
    if cfg.SGRL.initialize_agent != "none":
        rl_method.agent.load(cfg.SGRL.initialize_agent)

    #-- setup SGRL
    sgrl = SGRL(env, W)

    #-- SGRL loop
    t = 0
    while(t<cfg.SGRL.T_end):
        sgrl.update_environment(env, t, rl_method, cfg.SGRL)
        rl_method.learn(total_timesteps=cfg.SGRL.T_block, reset_num_timesteps=False, log_interval=100, tb_log_name=cfg.tag)
        t += cfg.SGRL.T_block


if __name__ == "__main__":
    main()
