import h5py
import numpy as np


class RndConfigs:

    def __init__(self, datafile: str, verbose: int=0):
        file = h5py.File(datafile, 'r')
        self.positions = file["positions"][()]
        
        self.joint_states = []
        self.frame_states = []
        for i in range(self.positions.shape[0]):
            # TODO: generalize this to any mujoco config
            joint_state = self.positions[i,3:]
            frame_state = np.zeros((1, 7))
            frame_state[0, :3] = self.positions[i,:3]
            
            self.joint_states.append(joint_state)
            self.frame_states.append(frame_state)
        
        self.joint_states = np.array(self.joint_states)
        self.frame_states = np.array(self.frame_states)

        if verbose:
            print(type(self.positions), self.positions.shape)
            