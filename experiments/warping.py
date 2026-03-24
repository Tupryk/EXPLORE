import mujoco
import mujoco_warp as mjw


mj_model = mujoco.MjModel.from_xml_path("configs/mujoco_/unitree_g1/scene.xml")
mj_data = mujoco.MjData(mj_model)

model = mjw.put_model(mj_model)
data = mjw.put_data(mj_model, mj_data, nworld=4096)

mjw.reset_data(model, data)
for t in range(10):
    print(f"Step {t}")
    mjw.step(model, data)
print(data.qpos.shape)
