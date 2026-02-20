import h5py as h5
import numpy as np  
from explore.utils.vis import play_path_mppi

def load_h5_to_dict(h5obj):
    result = {}
    for key, item in h5obj.items():
        if isinstance(item, h5.Dataset):
            result[key] = item[()]  # load into memory
        elif isinstance(item, h5.Group):
            result[key] = load_h5_to_dict(item)
    return result

# usage
with h5.File("results.h5", "r") as f:
    data = load_h5_to_dict(f)

for key, value in data.items():
    # You can also access the trajectory and reached_goal like this:
    traj = value['trajectory']
    reached_goal = value['reached_goal']
    play_path_mppi(traj, sim, start_state, target_state, tau_action=sim_cfg.tau_action, camera=cfg.RRT.sim.camera, reset_state=True)