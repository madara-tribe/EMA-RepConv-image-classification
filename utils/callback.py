import os
from pathlib import Path
import torch

class CallBackModelCheckpoint(object):
    def __init__(self, config):
        Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = config.ckpt_dir

    def __call__(self, global_step, loss, model: torch.nn.Module):
        torch.save(model.state_dict(), os.path.join(self.ckpt_dir, str(global_step)+'_'+str(loss)+'_'+"model.pth"))



