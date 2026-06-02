import mujoco
import numpy as np


def explain_qpos(model: mujoco.MjModel):
    print("nq =", model.nq)
    print("njnt =", model.njnt)

    for j in range(model.njnt):
        jname = model.joint(j).name
        jtype = model.jnt_type[j]
        addr  = model.jnt_qposadr[j]
        dof = {0:"free(7)", 1:"ball(4)", 2:"slide(1)", 3:"hinge(1)"}[jtype]
        print(f"Joint {j}: {jname}, type={jtype}, qpos address={addr}, dof={dof}")
        
    print("Ctrl ranges: ", model.actuator_ctrlrange)
    print("Ctrl ranges magnitude: ", model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0])

def get_model_quaternions(model: mujoco.MjModel):

    scene_quat_indices = []
    
    for j in range(model.njnt):
        joint_type = model.jnt_type[j]
        qpos_adr = model.jnt_qposadr[j]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            scene_quat_indices.append(int(qpos_adr+3))

    return scene_quat_indices

def getPossibleContacts(model):
    frames = []
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        frames.append(body_id)

    possible_contacts = []
    while len(frames):
        for i in range(1, len(frames)):
            possible_contacts.append((frames[0], frames[i]))
        del frames[0]
    
    return possible_contacts

def getContacts(data, model, possible_contacts) -> np.ndarray:
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1 = contact.geom1
        geom2 = contact.geom2

        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        contacts.append((body1, body2))
    
    contacts_vec = np.zeros(len(possible_contacts))
    for c in contacts:
        if c in possible_contacts:
            contacts_vec[possible_contacts.index(c)] = 1
        else:
            contacts_vec[possible_contacts.index((c[1], c[0]))] = 1
    
    return contacts_vec
