import hydra
from omegaconf import DictConfig

from explore.datasets.optim_trajs import OptimPair
from explore.datasets.utils import find_config_pairs


@hydra.main(version_base="1.3",
            config_path="../configs/yaml",
            config_name="short_trajs_gen")
def main(cfg: DictConfig):
    # pairs = find_config_pairs()
    op = OptimPair()
    ctrls = op.run()

if __name__ == "__main__":
    main()
