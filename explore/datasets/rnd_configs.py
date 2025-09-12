import h5py


class RndConfigs:

    def __init__(self, datafile: str, verbose: int=0):
        file = h5py.File(datafile, 'r')
        self.positions = file["positions"][()]
        if verbose:
            print(type(self.positions), self.positions.shape)
            