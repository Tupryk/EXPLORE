import robotic as ry


C = ry.Config()
C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))
# ball_pos = [.3, .3, 1.2]
ball_pos = C.getFrame("l_gripper").getPosition()
C.addFrame("ball", "world") \
    .setPosition(ball_pos) \
    .setShape(ry.ST.sphere, [.03]) \
    .setColor([.1, .2, .7]) \
    .setJoint(ry.JT.trans3) \
    .setContact(1)
C.getFrame("table").setShape(ry.ST.ssBox, [4., 4., .05, .01])
C.view(True)

komo = ry.KOMO()
komo.setConfig(C, True)
komo.setTiming(4, 32, .8, 2)

komo.addControlObjective([], 0, 1e-1)
komo.addControlObjective([], 1, 1e-1)
komo.addControlObjective([], 2, 1e-1)

komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)

komo.addObjective([0., 3.], ry.FS.positionDiff, ["ball", "l_gripper"], ry.OT.eq, [1e1])
komo.addObjective([3.], ry.FS.position, ["l_gripper"], ry.OT.eq, [0, 1e1, 0], [0., 1, 0], 1)
komo.addObjective([3.], ry.FS.position, ["l_gripper"], ry.OT.eq, [1e1, 0, 1e1], [0., 0., -9.8], 2)
komo.addObjective([3., 4.], ry.FS.position, ["ball"], ry.OT.eq, [0, 1e1, 0], [0., 1, 0], 1)
komo.addObjective([3., 4.], ry.FS.position, ["ball"], ry.OT.eq, [1e1, 0, 1e1], [0., 0., -9.8], 2)

komo.addObjective([4.], ry.FS.negDistance, ["ball", "table"], ry.OT.eq, [1e1], [0.0])
komo.addObjective([3., 4.], ry.FS.positionDiff, ["l_gripper", "ball"], ry.OT.sos, [1e-1])

ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()

if not ret.feasible:
    print("No solution found!")
    komo.view(True)
    komo.view_play(True)
    exit()

komo.view(True)
komo.view_play(True)
