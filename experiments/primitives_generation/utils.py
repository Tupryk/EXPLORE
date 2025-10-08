import numpy as np
import robotic as ry


def gauss_sample_goal(C: ry.Config, perc: float=.1) -> np.ndarray:
    
    q = C.getJointState()
    q_min, q_max = C.getJointLimits()
    C_copy = ry.Config()
    C_copy.addConfigurationCopy(C)
    
    while True:
        goal_q = q + (np.random.randn(q.shape[0]) * (q_max - q_min) * perc)

        for i, _ in enumerate(goal_q):

            if goal_q[i] < q_min[i]:
                goal_q[i] = q_min[i]
            
            elif goal_q[i] > q_max[i]:
                goal_q[i] = q_max[i]
        
        C_copy.setJointState(goal_q)
        C_copy.computeCollisions()
        collisions = C_copy.getCollisions()
        if not len(collisions):
            break

    return goal_q
