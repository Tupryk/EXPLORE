import mujoco


def explain_qpos(model: mujoco.MjModel):
    print("nq =", model.nq)
    print("njnt =", model.njnt)

    for j in range(model.njnt):
        jname = model.joint(j).name
        jtype = model.jnt_type[j]
        addr  = model.jnt_qposadr[j]
        dof = {0:"free(7)", 1:"ball(4)", 2:"slide(1)", 3:"hinge(1)"}[jtype]
        print(f"Joint {j}: {jname}, type={jtype}, qpos address={addr}, dof={dof}")
        