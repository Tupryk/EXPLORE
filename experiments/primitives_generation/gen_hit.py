import numpy as np
import robotic as ry

from utils import gauss_sample_goal

# seed = np.random.randint(0, 1e6)
# print(seed)
seed = 397041
np.random.seed(seed)


txt_file = "data/joint_states.txt"

C = ry.Config()
C.addFile("/home/eckart/git/robotic/rai-robotModels/scenarios/pandasTable.g")
frame = C.addFrame("obj", "world")
frame.setPosition([0., 0., .7])
frame.setColor([0., .4, .8])
frame.setShape(ry.ST.sphere, [.03])
frame.setMass(.08)
frame.setContact(1)
frame.setJoint(ry.JT.trans3, [
    -1.0, -1.0, 0.5,
     1.0,  1.0, 2.0
])

# data = np.loadtxt(txt_file, dtype=np.float64)
# idx = np.random.randint(0, data.shape[0])
# idx = 0
# q = data[idx]

# q_min, q_max = C.getJointLimits()
# q = np.array([np.random.uniform(low, high) for low, high in zip(q_min, q_max)])
q = gauss_sample_goal(C, perc=.1)

C.setJointState(q)

CG = ry.Config()
CG.addConfigurationCopy(C)

goal_q = gauss_sample_goal(C, perc=.1)

CG.setJointState(goal_q)

C.view(False, "Start State")
CG.view(True, "Goal State")

"""
TODO: Incorporate the states current velocities into the KOMO problem
Possibilities for the motion primitive:
    - Motions:
        - Hit
        - Move
    - POAs (determine the amount of phases):
        - Keep subset of initial POAs
        - Switch POAs
        - Up to three phases

There should be some herugistic to determine which start and end configs should use which motions/POAs
"""

q_goal = np.zeros(21)
q_goal[:8] = goal_q[:8]
q_goal[8] = goal_q[7]
q_goal[9:17] = goal_q[8:16]
q_goal[17] = goal_q[15]
q_goal[18:] = goal_q[16:]

contact_frames = [
    "l_panda_link0",  "r_panda_link0",
    "l_panda_joint1", "l_panda_joint2", "l_panda_joint3", "l_panda_joint4", "l_panda_joint5", "l_panda_joint6", "l_panda_joint7",
    "r_panda_joint1", "r_panda_joint2", "r_panda_joint3", "r_panda_joint4", "r_panda_joint5", "r_panda_joint6", "r_panda_joint7",
    "l_panda_finger_joint1",  "l_panda_rightfinger_0", "r_panda_finger_joint1",  "r_panda_rightfinger_0"
]

for i, contact_frame in enumerate(contact_frames):

    # contact_frame = np.random.choice(contact_frames)
    print(f"Attempt number {i} manipulation with frame '{contact_frame}'. ", end="")

    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(2, 32, .2, 2)

    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)
    komo.addControlObjective([], 2, 1e-1)

    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)

    # Contact with obj
    # Not really realistic as the surface normals in the contact are not accounted for.
    # It basically works as if the two objects where glued together in the interval [1., 1.1].
    # Maybe we should use friction cone constraints or so?
    komo.addObjective([1., 1.1], ry.FS.negDistance, [contact_frame, "obj"], ry.OT.eq, [1e1], [0.])
    komo.addObjective([1., 1.1], ry.FS.positionDiff, [contact_frame, "obj"], ry.OT.eq, [1e1], [0.], 1)

    komo.addObjective([0., 1.], ry.FS.position, ["obj"], ry.OT.eq, [1e1], [0., 0., -9.8], 2)
    komo.addObjective([1.1, 2.], ry.FS.position, ["obj"], ry.OT.eq, [1e1], [0., 0., -9.8], 2)
    komo.addObjective([2.], ry.FS.qItself, [], ry.OT.sos, [1e2], q_goal)

    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()

    if not ret.feasible:
        print("Failed")
        continue
        print("No solution found!")
        komo.view(True)
        komo.view_play(True)
        exit()

    print("Success")
    komo.view(True)
    komo.view_play(True)
