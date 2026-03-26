from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, tag):
        self.data = {}
        self.tag = tag
        self.writer = SummaryWriter(log_dir=f"tensorboard/{self.tag}")

    def add(self, key, value):
        if not key in self.data:
            self.data[key] = [0.0, 0.0]
        self.data[key][0] += float(value)
        self.data[key][1] += 1.0

    def get(self):
        dat = {}
        for key, value in self.data.items():
            if value[1] > 0.0:
                dat[key] = value[0] / value[1]
                self.data[key] = [0.0, 0.0]
        return dat

    def write(self, t, verbose=0):
        D = self.get()
        for key, value in D.items():
            self.writer.add_scalar(f"RL/{key}", value, t)
        self.writer.flush()
