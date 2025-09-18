import time
import torch

from explore.models.vaes import VAE
from explore.env.mujoco_sim import MjSim
from experiments.state_encoding.utils import get_state_vectors, eval_state_decoding


sim = MjSim("configs/twoFingers.xml", view=True)

states = get_state_vectors("data/15-04-21/trees")
states = torch.from_numpy(states)

model = VAE()
model.load_state_dict(torch.load("experiments/state_encoding/model.pth"))

model.eval()
with torch.no_grad():
    
    start_idx = 1
    end_idx = 0
    
    original = states[start_idx].reshape(1, -1)
    recon, mu, logvar = model(original)
    
    recon_vec = recon[0].cpu().numpy()
    original_vec = original[0].cpu().numpy()
    eval_state_decoding(original_vec, recon_vec)
    
    print("Original State:")
    joint_state = original_vec[:13]
    sim.pushConfig(joint_state)
    time.sleep(4)
    
    print("Reconstructed State:")
    joint_state = recon_vec[:13]
    sim.pushConfig(joint_state)
    time.sleep(4)
