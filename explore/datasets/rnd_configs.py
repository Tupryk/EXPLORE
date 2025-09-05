import time
import h5py
import numpy as np
import robotic as ry


class RndConfigs:

    def __init__(self, configfile: str, datafile: str, verbose: int=0):
        self.C = ry.Config()
        self.C.addFile(configfile)
        self.obj = self.C.getFrame("obj")

        file = h5py.File(datafile, 'r')
        self.positions = file["positions"][()]
        if verbose:
            print(type(self.positions), self.positions.shape)

    def set_config(self, i):
        self.obj.setPosition(self.positions[i,:3])
        self.C.setJointState(self.positions[i,3:])

    def simulate(self, duration: float=1., move: bool=False, tau: float=.01):
        t = 0
        sim = ry.Simulation(self.C, engine=ry.SimulationEngine.physx, verbose=1)
        if move:
            q = self.C.getJointState()
            q1 = q + .1 * np.random.randn(q.size)
            sim.setSplineRef(q1, [.1])

        while t < duration:
            sim.step([], tau, ry.ControlMode.none)
            self.C.view(False, f"simulating t={t:.2f}")
            time.sleep(tau)
            t += tau

    def simulate_all(self):
        for i in range(self.positions.shape[0]):
            self.set_config(i)
            self.simulate(1)
    
    def display_all(self):
        for i in range(self.positions.shape[0]):
            self.set_config(i)
            self.C.view()
            time.sleep(1.1)
