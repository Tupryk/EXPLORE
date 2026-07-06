import time
import mujoco
import mujoco.viewer

from explore.utils.mj import explain_qpos

xml_path = "configs/mujoco_/unitree_go2/parkour.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

explain_qpos(model)
mujoco.viewer.launch_passive(model, data)

time.sleep(5.)
