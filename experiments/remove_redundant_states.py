import h5py
import mujoco
import numpy as np

from explore.utils.mj import geom_names2ids


MIN_DIST = 0.01

old_h5 = "configs/stable/double_sphere.h5"
new_h5 = "configs/stable/double_sphere_clean.h5"
xml_path = "configs/mujoco_/doubleSphere.xml"

G = ["obj"]
q_ids = [0, 3]
q_weight = 0.1

file = h5py.File(old_h5, 'r')
manifold_qpos = file["qpos"] if "qpos" in file.keys() else file["q"]
manifold_ctrl = file["ctrl"]
manifold_size = manifold_qpos.shape[0]

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

G_ids = geom_names2ids(G, mj_model)

phi_stable_configs = []
for i in range(manifold_size):
    mj_data.qpos[:] = manifold_qpos[i]
    mujoco.mj_forward(mj_model, mj_data)

    q_ = mj_data.qpos[q_ids[0]:q_ids[1]]
    G_ = mj_data.geom_xpos[G_ids, :].reshape(-1)
    phi = np.concatenate([q_ * q_weight, G_])
    phi_stable_configs.append(phi)

phi_stable_configs = np.array(phi_stable_configs)

# --- Greedy redundancy reduction ---
kept_indices = []
kept_phis = []  # list of np.ndarray, stacked lazily for distance checks

for i in range(manifold_size):
    phi_i = phi_stable_configs[i]

    if kept_phis:
        kept_arr = np.stack(kept_phis, axis=0)  # (K, D)
        dists = np.linalg.norm(kept_arr - phi_i[None, :], axis=1)
        min_dist = dists.min()
    else:
        min_dist = np.inf

    if min_dist >= MIN_DIST:
        kept_indices.append(i)
        kept_phis.append(phi_i)

kept_indices = np.array(kept_indices)

new_manifold_qpos = np.array(manifold_qpos)[kept_indices]
new_manifold_ctrl = np.array(manifold_ctrl)[kept_indices]

file.close()

with h5py.File(new_h5, "w") as f:
    f.create_dataset("qpos", data=new_manifold_qpos)
    f.create_dataset("ctrl", data=new_manifold_ctrl)

print(f"Success! Data saved to {new_h5}. (Total points: {new_manifold_qpos.shape[0]} / {manifold_size})")
