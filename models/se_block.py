import torch
import torch.nn as nn
import torch.nn.functional as F

#   https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html

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
        scale = torch.multiply(inputs, excitation)
        return scale

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        size = int(inputs.size(3))
        x = F.avg_pool2d(inputs, kernel_size=size)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class Squeeze_excitation_layer__(nn.Module):
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
        excitation = x.view(-1, self.insize, 1, 1)
        return torch.multiply(inputs, excitation)
        
