import torch
import random
import numpy as np
import robotic as ry
import torch.nn as nn

from explore.env.MujocoSim import MjSim
from explore.datasets.rnd_configs import RndConfigs


def compute_cost(state0: tuple,
                 state1: tuple,
                 relevant_frames_idxs: list[int],
                 relevant_frames_weight: list[float]=[]) -> float:

    cost = 0.

    for i, frame_idx in enumerate(relevant_frames_idxs):
        
        pos0 = state0[0][frame_idx][:3]
        pos1 = state1[0][frame_idx][:3]
        
        err = pos0 - pos1
        err = np.linalg.norm(err)
        if len(relevant_frames_weight):
            err *= relevant_frames_weight[i]
        
        cost += err

    return cost


def compute_contacts(C0: ry.Config, contactees: list[str], contacted: list[str]) -> np.ndarray:
    contacts = np.zeros( len(contactees) * len(contacted) - len(contactees) )
    idx = 0
    for ees in contactees:
        for ed in contacted:
            if ees != ed:
                if 0. >= C0.eval(ry.FS.negDistance, [ees, ed])[0][0]:
                    contacts[idx] = 1
                idx += 1
    return contacts*.1


def randint_excluding(low: int, high: int, exclude: int):
    x = np.random.randint(low, high - 1)
    return x if x < exclude else x + 1


def k_means(vectors: list[list[float]], k: int, cost_func, max_iter: int=100) -> list[int]:
    
    centroid_idxs = np.random.choice(len(vectors), k, replace=False)
    centroids = vectors[centroid_idxs]

    for _ in range(max_iter):
        # Assign points to the nearest centroid
        labels = []
        for vec in vectors:
            distances = [cost_func([vec], [c]) for c in centroids]
            labels.append(np.argmin(distances))
        labels = np.array(labels)
        
        # Update centroids
        new_centroids = np.array([
            vectors[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        
        # Convergence check
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    
    return labels


def sample_cluster_balanced(node_idx: list[int], labels: list[int]) -> tuple[int, int]:

    cluster_idx = random.randint(0, max(labels))

    idxs = [i for i, l in enumerate(labels) if cluster_idx == l]
    sampled_idx = random.choice(idxs)

    return node_idx[sampled_idx], cluster_idx


def play_model_mlp(model: nn.Module,
                   start_config_idx: int,
                   target_config_idx: int=-1,
                   steps: int=20) -> float:
    
    Ds = RndConfigs("data/twoFingers.g", "data/rnd_twoFingers.h5")
    Ds.set_config(start_config_idx)

    sim = MjSim(open('data/twoFingers.xml', 'r').read(), Ds.C, view=False)
    tau = .1
    time_offset = .0
    horizon = 4
    history = 1
    relevant_frames = ["obj", "l_fing", "r_fing"]
    relevant_frames_idxs = [Ds.C.getFrameNames().index(rf) for rf in relevant_frames]

    if target_config_idx != -1:
        De = RndConfigs("data/twoFingers.g", "data/rnd_twoFingers.h5")
        De.set_config(target_config_idx)
        sim_ = MjSim(open('data/twoFingers.xml', 'r').read(), De.C, view=False)
        goal_state = sim_.getState()[0][relevant_frames_idxs, :3].flatten()
        goal_state = torch.tensor(goal_state).float().unsqueeze(0).to("cuda")

    state = sim.getState()[0][relevant_frames_idxs, :3].flatten()
    model_in = [state for _ in range(history+1)]
    model_in = torch.tensor(model_in).reshape(1, -1).float().to("cuda")

    for i in range(0, steps, horizon):

        if target_config_idx != -1:
            model_out = model(torch.cat((goal_state, model_in.reshape(history+1, -1)), dim=0).reshape(1, -1))
        else:
            model_out = model(model_in)

        q_targets = model_out.cpu().detach().numpy().reshape(horizon, -1)

        for j, q in enumerate(q_targets):
            sim.resetSplineRef(time_offset)
            sim.setSplineRef(q.reshape(1,-1), [.1], append=False)
        
            sim.step([], tau, ry.ControlMode.spline, .5)
            time_offset += tau
            Ds.C.view(True)

            print(f"Finished step number {i+j}")

        state = sim.getState()[0][relevant_frames_idxs, :3].flatten()
        model_in = model_in[:, :len(state)*(history)]
        state = torch.tensor(state).float().to("cuda")
        model_in = torch.cat((model_in.squeeze(0), state)).unsqueeze(0)

    error = 0.
    if target_config_idx != -1:
        # TODO: Calculate error
        pass

    return error
