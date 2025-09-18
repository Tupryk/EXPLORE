import os
import pickle
import numpy as np

from explore.env.mujoco_sim import MjSim


def get_state_vectors(dataset: str) -> np.ndarray:
    sim = MjSim("configs/twoFingers.xml")

    states = []
    tree_count = len(os.listdir(dataset))
    for i in range(tree_count):
        
        data_path = os.path.join(dataset, f"tree_{i}.pkl")
        with open(data_path, "rb") as f:
            tree: list[dict] = pickle.load(f)

            for node in tree:
                node_state = node["state"]
                sim.setState(*node_state)
                
                pos = node_state[1]
                vel = node_state[2]
                contacts = sim.getContacts()
                
                state = np.hstack((pos, vel, contacts))
                state = np.hstack((pos, contacts))
                states.append(state)
        
    states = np.array(states).astype(np.float32)
    return states

def eval_state_decoding(original_vec: np.ndarray, recon_vec: np.ndarray):
    total_err = 0
    total_err_colls = 0
    print(f"    Original | Reconstructed")
    for j in range(original_vec.shape[0]):
        if j < 13:
            err = np.abs(original_vec[j] - recon_vec[j])
            total_err += err
        else:
            err = np.abs(original_vec[j] - np.round(recon_vec[j]))
            total_err_colls += err
        print(f"{j:2d}: {original_vec[j]:8.4f} | {recon_vec[j]:8.4f} -> err {err:8.4f}")
    print(f"------------------------")
    print(f"Avg. error: {total_err/25:.4f}, Avg. accuracy for collisions: {100*(1-total_err_colls/15):.2f}%\n")
