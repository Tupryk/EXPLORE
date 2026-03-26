import cma
import h5py
import mujoco
import numpy as np
from tqdm import trange

from explore.env.mujoco_sim import MjSim
from explore.utils.mj import explain_qpos


mujoco_xml = "configs/mujoco_/unitree_g1/table_box_scene.xml"

sim = MjSim(mujoco_xml, view=False, verbose=1)
explain_qpos(sim.model)


def random_quaternion():
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()

    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])

    return q


def sample_uniform_qpos():
    qpos = np.zeros_like(sim.data.qpos)
    
    qpos[0:2] = np.random.uniform(-1.5, 1.5, (2,))
    qpos[2] = np.random.uniform(1., 1.5)
    qpos[3:7] = [1, 0, 0, 0]

    ranges = sim.model.actuator_ctrlrange[:, 1] - sim.model.actuator_ctrlrange[:, 0]
    qpos[7:-7] = ranges * np.random.uniform(size=ranges.shape) + sim.model.actuator_ctrlrange[:, 0]

    if np.random.uniform() < 0.75:
        qpos[-7:-4] = qpos[0:3] + np.random.randn(3) * 0.2
    else:
        qpos[-7:-5] = np.random.uniform(-1.5, 1.5, (2,))
        qpos[-5] = np.random.uniform(1., 1.5)
    qpos[-4:] = random_quaternion()

    ctrl = qpos[7:-7]
    
    return qpos, ctrl

def eval_state(qpos: np.ndarray, target_qpos: np.ndarray) -> tuple[float, float, float]:
    
    sim.pushConfig(qpos, qpos[7:-7])
    sim.step(0.5, view=0.)

    vel_cost = sim.data.qvel.T @ sim.data.qvel
    qpos_e = sim.data.qpos - target_qpos
    qpos_cost = qpos_e[7:-7].T @ qpos_e[7:-7]
    qpos_cost *= 0.1
    qpos_cost += 20 * qpos_e[2]**2
    qpos_cost += 100 * (qpos_e[3:7].T @ qpos_e[3:7])
    qpos_cost += 20 * qpos_e[-5]**2
    dist_cost = max(min_distance_to_geom(sim.model, sim.data, robot_geom_ids, box_geom_id), 0)
    dist_cost += max(min_distance_to_geom(sim.model, sim.data, robot_geom_ids, floor_geom_id), 0)

    return vel_cost, qpos_cost, dist_cost

def eval_states(candidates: np.ndarray, target_qpos: np.ndarray) -> list[float]:
    results = []
    for c in candidates:
        v, p, d = eval_state(c, target_qpos)
        results.append(v * 50 + p + d * 100)
    
    return results

def get_all_descendant_body_ids(model, root_body_id):
    """Get all body IDs in the subtree rooted at root_body_id."""
    body_ids = [root_body_id]
    # model.body_parentid lets us walk the tree
    for body_id in range(model.nbody):
        if body_id == root_body_id:
            continue
        # Walk up the parent chain to check if root is an ancestor
        current = body_id
        while current != 0:  # 0 is the world body (root)
            if current == root_body_id:
                body_ids.append(body_id)
                break
            current = model.body_parentid[current]
    return body_ids

def get_robot_geom_ids(model, root_body_name):
    """Get all geom IDs for a body and all its descendants."""
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    all_body_ids = get_all_descendant_body_ids(model, root_body_id)
    
    geom_ids = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] in all_body_ids:
            geom_ids.append(geom_id)
    return geom_ids

robot_geom_ids = get_robot_geom_ids(sim.model, "pelvis")

def min_distance_to_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
    target_geom_id: int,
    dist_upper_bound: float = 1e9,
) -> float:
    fromto = np.zeros(6)
    return min(
        mujoco.mj_geomDistance(model, data, gid, target_geom_id, dist_upper_bound, fromto)
        for gid in geom_ids
        if gid != target_geom_id
    )

box_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_col")
floor_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

stable_configs = []
stable_configs_ctrl = []

for i in trange(10000):
    
    while True:
        target_qpos, target_ctrl = sample_uniform_qpos()
        sim.pushConfig(target_qpos, target_ctrl)
        if sim.data.ncon == 0: break
    
    # sim.step(10, view=0.)
    # sim.step(1, view=1.)

    # Optimize
    while True:
        initial_guess_qpos, initial_guess_ctrl = sample_uniform_qpos()
        sim.pushConfig(initial_guess_qpos, initial_guess_ctrl)
        if sim.data.ncon == 0: break
    
    es = cma.CMAEvolutionStrategy(target_qpos, .5, {
        "popsize": 64,
        "maxfevals": 6400,
        "verbose": -1
    })

    while not es.stop():
        candidates = es.ask()

        results = eval_states(candidates, target_qpos)
        print("Min cost: ", min(results))

        es.tell(candidates, results)
        es.disp()

    final_qpos = es.result.xbest
    best_vel_cost, best_qpos_cost, best_dist_cost = eval_state(final_qpos, target_qpos)

    print(f"Done! with cost {best_vel_cost}, {best_qpos_cost}, {best_dist_cost}")
    sim.pushConfig(final_qpos, final_qpos[7:-7])
    # sim.step(10, view=1.)

    stable_configs.append(final_qpos)
    stable_configs_ctrl.append(final_qpos[7:-7])

    qpos_array = np.array(stable_configs)
    ctrl_array = np.array(stable_configs_ctrl)

    with h5py.File("./experiments/stable_configs.h5", "w") as f:
        f.create_dataset("qpos", data=qpos_array)
        f.create_dataset("ctrl", data=ctrl_array)
