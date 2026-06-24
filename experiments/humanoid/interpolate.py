import time
import h5py
import mujoco
import numpy as np
import mujoco.viewer
from sklearn.neighbors import KDTree
from explore.utils.mj import geom_names2ids

def view_state(model, data, qpos, viewer, seconds=1.):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(seconds)


xml_path = "configs/mujoco_/unitree_g1/box_scene.xml"
stable_path = "configs/stable/humanoid_box_grasps.h5"
q_weight = 0.1
q_id = [7, 36]
G_id = ["obj_col", "box_marker_0", "box_marker_1", "box_marker_2"]

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)

G_id = geom_names2ids(G_id, model)

stable_configs = h5py.File(stable_path, 'r')
stable_configs_qpos = stable_configs["qpos"]
config_count = stable_configs_qpos.shape[0]

phi_stable_configs = []
for i in range(config_count):
    data.qpos[:] = stable_configs_qpos[i]
    mujoco.mj_forward(model, data)

    q = data.qpos[q_id[0]:q_id[1]]
    G = data.geom_xpos[G_id, :].reshape(-1)
    phi = np.concatenate([q * q_weight, G])
    
    phi_stable_configs.append(phi)

phi_stable_configs = np.array(phi_stable_configs)
sds = KDTree(phi_stable_configs)

s_cfg_idx = np.random.randint(0, config_count)
e_cfg_idx = np.random.randint(0, config_count)

print("End ", e_cfg_idx)
view_state(model, data, stable_configs_qpos[e_cfg_idx], viewer)
print("Start ", s_cfg_idx)
view_state(model, data, stable_configs_qpos[s_cfg_idx], viewer)

for i, t in enumerate(np.linspace(0., 1., 100)):

    query = (
        phi_stable_configs[s_cfg_idx] * t +
        phi_stable_configs[e_cfg_idx] * (1. - t)
    )
    query = query.reshape(1, -1)
    
    _, ind = sds.query(query, k=1)
    projection_id = int(ind[0][0])
    
    print("Config ", i+1)
    view_state(model, data, stable_configs_qpos[projection_id], viewer, seconds=.1)

# Humanoid locomanipulation through object diffusion
# Problem: obj diffussion does not respect collisions
stds = np.array([2., 2., .3, 1., 1., 1., 1.])
offsets = np.array([0., 0., 1., 0., 0., 0., 0.])
noised_cube = np.random.randn(7) * stds + offsets

for i, t in enumerate(np.linspace(0., 1., 100)):

    cube_pos = stable_configs_qpos[s_cfg_idx][-7:] * (1. - t) + noised_cube * t
    
    print("Config ", i+1)
    noised_qpos = stable_configs_qpos[s_cfg_idx].copy()
    noised_qpos[-7:] = cube_pos
    
    view_state(model, data, noised_qpos, viewer, seconds=.1)
