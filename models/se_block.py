import torch
import torch.nn as nn
import torch.nn.functional as F

class Squeeze_excitation_layer(nn.Module):
    def __init__(self, insize):
        super(Squeeze_excitation_layer, self).__init__()
        self.ratio = 4
        self.insize = insize
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(insize, int(insize/self.ratio))
        self.fc2 = nn.Linear(int(insize/self.ratio), insize)

    def forward(self, x):
        size = int(x.size(3))
        inputs = x
        x = F.avg_pool2d(x, kernel_size=size)
        x = torch.flatten(x, 1)
        x = self.fc2(self.relu(self.fc1(x)))
        x = F.sigmoid(x)
        excitation = torch.reshape(x, shape=(-1, self.insize, 1, 1))
        return torch.multiply(inputs, excitation)

        
