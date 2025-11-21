import torch
import matplotlib.pyplot as plt

from explore.models.utils import SinusoidalPosEmb


if __name__ == "__main__":
    layer = SinusoidalPosEmb(256)
    
    ts = torch.linspace(0, 1, 256)
    im = layer(ts).T
    print(im.shape)
    plt.imshow(im)
    plt.show()
