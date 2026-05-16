import h5py
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from explore.env.mujoco_sim import MjSim
from explore.utils.mj import get_model_quaternions
from explore.datasets.utils import cost_computation


@hydra.main(version_base="1.3",
            config_path="../configs/yaml",
            config_name="short_trajs_gen")
def main(cfg: DictConfig):

    h5_file = "configs/stable/humanoid_box_grasps.h5"
    mujoco_xml = "configs/mujoco_/unitree_g1/table_box_scene.xml"

    file = h5py.File(h5_file, 'r')
    stable_configs = file["qpos"]
    stable_configs_ctrl = file["ctrl"]
    
    sim = MjSim(mujoco_xml, verbose=1, tau_sim=1e-3)
    scene_quat_indices = get_model_quaternions(sim.model)

    state_vectors = []
    for i, sc in enumerate(stable_configs):
        q = stable_configs_ctrl[i].copy()
        sim.pushConfig(sc, q)
        sv = sim.getStateVector(
            cfg.q_mask,
            cfg.velocity_weight,
            cfg.objs, cfg.contacts,
            cfg.dist_weight, cfg.dist_max,
            cfg.geoms_in_cost, cfg.geoms_in_cost_weights
        )
        state_vectors.append(sv)
    
    pairs = []
    for i, sv_a in tqdm(enumerate(state_vectors), total=len(state_vectors)):
        for j, sv_b in enumerate(state_vectors):
            cost = cost_computation(sv_a, sv_b, scene_quat_indices)
            if cost < cfg.min_cost:
                pairs.append((i, j))
                
    print(pairs)
    print(f"Total pairs: {len(pairs)} (of {len(state_vectors)**2} possible ({((len(pairs)/len(state_vectors)**2)*100):.2f}))")
                
    # for pair in pairs:
    #     start_id, end_id = pair
    #     sc = stable_configs[start_id].copy
    #     q = stable_configs_ctrl[start_id].copy()
    #     target = state_vectors[end_id]
        
    #     sim.pushConfig(sc, q)
    #     optimize_traj(sim, target)


if __name__ == "__main__":
    main()
