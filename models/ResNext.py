import torch
import torch.nn as nn
import torch.nn.functional as F
from .RepConv import _RepConv
from .se_block import Squeeze_excitation_layer

        
        
class DownConv(nn.Module):
    def __init__(self, in_planes, outsize, cardinality=4, bottleneck_width=1.25):
        super(DownConv, self).__init__()
        group_width = int(in_planes * bottleneck_width)
        conv_group = cardinality if (group_width%cardinality)==0 else 1
                
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_planes, outsize, kernel_size=1, bias=False),
            nn.BatchNorm2d(outsize),
            nn.Conv2d(outsize, group_width, kernel_size=3, stride=1, padding=1, groups=conv_group, bias=False),
            nn.BatchNorm2d(group_width),
            nn.Conv2d(group_width, outsize, kernel_size=1, bias=False),
            nn.BatchNorm2d(outsize),
            nn.MaxPool2d(2, 2))


    def forward(self, x):
        x = self.dw_conv(x) 
        return x
    
class Grouped_Block(nn.Module):
    '''Grouped convolution block.'''
    def __init__(self, in_planes, bottleneck_width=1.25, cardinality=4, stride=1, act=True):
        super(Grouped_Block, self).__init__()
        self.act = nn.SiLU() if act else nn.ReLU()
        group_width = int(in_planes * bottleneck_width)
        conv_group = cardinality if (group_width%cardinality)==0 else 1
        
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, group_width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, in_planes, kernel_size=1, groups=conv_group, bias=False)
        self.bn3 = nn.BatchNorm2d(in_planes)
        self.scse = Squeeze_excitation_layer(in_planes)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.act(self.bn3(self.conv3(out)))
        out = self.scse(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out
        
class ResNeXt(nn.Module):
    def __init__(self, deploy, bottleneck_width, cardinality=4):
        super(ResNeXt, self).__init__()
        num_blocks = [3, 32, 64, 128, 256, 512]
        self.deploy = deploy
        
        i = 1
        # 0
        self.stem = nn.Sequential(
            nn.Conv2d(num_blocks[i-1], num_blocks[i], kernel_size=7, padding=2),
            nn.BatchNorm2d(num_blocks[i]),
            nn.LeakyReLU(),
            nn.Conv2d(num_blocks[i], num_blocks[i], kernel_size=7, padding=2),
            nn.BatchNorm2d(num_blocks[i]),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))
            
        # 1 
        i += 1
        self.conv1 = DownConv(num_blocks[i-1], num_blocks[i], cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.res_block1 = Grouped_Block(in_planes=num_blocks[i], bottleneck_width=bottleneck_width, cardinality=cardinality, stride=1)
        
        # 2
        i += 1
        self.conv2 = DownConv(num_blocks[i-1], num_blocks[i], cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.res_block2 = Grouped_Block(in_planes=num_blocks[i], bottleneck_width=bottleneck_width, cardinality=cardinality, stride=1)

        # 3
        i += 1
        self.conv3 = DownConv(num_blocks[i-1], num_blocks[i], cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.res_block3 = Grouped_Block(in_planes=num_blocks[i], bottleneck_width=bottleneck_width, cardinality=cardinality, stride=1)
        
        
        # 4
        i += 1
        self.conv4 = DownConv(num_blocks[i-1], num_blocks[i], cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.res_block4 = Grouped_Block(in_planes=num_blocks[i], bottleneck_width=bottleneck_width, cardinality=cardinality, stride=1)
        
        self.repconv = _RepConv(num_blocks[i], num_blocks[i], num_blocks=0, deploy=self.deploy)

    def forward(self, x):
        # 0
        x0 = self.stem(x)

        # 1
        x = self.conv1(x0)
        x = self.res_block1(x)
        x1 = self.res_block1(x)

        # 2
        x = self.conv2(x1)
        x = self.res_block2(x)
        x2 = self.res_block2(x)

        # 3
        x = self.conv3(x2)
        x = self.res_block3(x)
        x3 = self.res_block3(x)

        # 3
        x = self.conv4(x3)
        x = self.res_block4(x)
        x4 = self.res_block4(x)

        # repconv
        out = self.repconv(x4)
        return out
    

#if __name__=='__main__':
#    from torchsummary import summary
#    model = ResNeXt(deploy=False, bottleneck_width=2, cardinality=4)
#    summary(model, (3, 260, 260))
#    inp = torch.randn(4, 3, 260, 260)
#    out = model(inp)
#    print(out.shape)


