import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import StemBlock, DownConv, ResBlock

class ResNeXt(nn.Module):
    def __init__(self, num_cls, bottleneck_width=2, canadility=32):
        super(ResNeXt, self).__init__()
        num_blocks = [3, 32, 64, 128, 256, 512]
        self.bottleneck_width = bottleneck_width
        
        # 0
        self.stem = StemBlock(3, 32)
        assert 32 == canadility
            
        # 1
        assert 64 == canadility * self.bottleneck_width
        self.conv1 = DownConv(32, 64)
        self.res_block1 = ResBlock(64, bottle_width=self.bottleneck_width, canadility=canadility)
        self.res_block2 = ResBlock(64, bottle_width=self.bottleneck_width, canadility=canadility)
        
        # 2
        self.bottleneck_width = self.bottleneck_width * 2
        assert 128 == canadility * self.bottleneck_width
        self.conv2 = DownConv(64, 128)
        self.res_block3 = ResBlock(128, bottle_width=self.bottleneck_width, canadility=canadility)
        self.res_block4 = ResBlock(128, bottle_width=self.bottleneck_width, canadility=canadility)
        
        # 3
        self.bottleneck_width = self.bottleneck_width * 2
        assert 256 == canadility * self.bottleneck_width
        self.conv3 = DownConv(128, 256)
        self.res_block5 = ResBlock(256, bottle_width=self.bottleneck_width, canadility=canadility)
        self.res_block6 = ResBlock(256, bottle_width=self.bottleneck_width, canadility=canadility)
        
        
        # 4
        self.bottleneck_width = self.bottleneck_width * 2
        assert 512 == canadility * self.bottleneck_width
        self.conv4 = DownConv(256, 512)
        self.res_block7 = ResBlock(512, bottle_width=self.bottleneck_width, canadility=canadility)
        self.res_block8 = ResBlock(512, bottle_width=self.bottleneck_width, canadility=canadility)
        
        ## fc
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_cls)
                            

    def forward(self, x):
        # 0
        x0 = self.stem(x)

        # 1
        x = self.conv1(x0)
        x = self.res_block1(x)
        x1 = self.res_block2(x)

        # 2
        x = self.conv2(x1)
        x = self.res_block3(x)
        x2 = self.res_block4(x)

        # 3
        x = self.conv3(x2)
        x = self.res_block5(x)
        x3 = self.res_block6(x)

        # 3
        x = self.conv4(x3)
        x = self.res_block7(x)
        out = self.res_block8(x)

        # repconv
        #out = self.repconv(x4)
        
        # fc
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

class LLMFc_ResNeXt(nn.Module):
    def __init__(self, num_cls, bottleneck_width=2, canadility=32):
        super(LLMFc_ResNeXt, self).__init__()
        num_blocks = [3, 32, 64, 128, 256, 512]
        self.canadility = canadility
        self.bottleneck_width = bottleneck_width
        fc_dim = 512 * 16
        
        # 0
        self.stem = StemBlock(3, 32)
        assert 32 == canadility
        
        # 1
        assert 64 == canadility * self.bottleneck_width
        self.conv1 = DownConv(32, 64)
        self.res_block1 = ResBlock(64, bottle_width=self.bottleneck_width, canadility=canadility)
        self.res_block2 = ResBlock(64, bottle_width=self.bottleneck_width, canadility=canadility)
        
        # 2
        self.bottleneck_width = self.bottleneck_width * 2
        assert 128 == canadility * self.bottleneck_width
        self.conv2 = DownConv(64, 128)
        self.res_block3 = ResBlock(128, bottle_width=self.bottleneck_width, canadility=canadility)
        self.res_block4 = ResBlock(128, bottle_width=self.bottleneck_width, canadility=canadility)
        
        # 3
        self.bottleneck_width = self.bottleneck_width * 2
        assert 256 == canadility * self.bottleneck_width
        self.conv3 = DownConv(128, 256)
        self.res_block5 = ResBlock(256, bottle_width=self.bottleneck_width, canadility=canadility)
        self.res_block6 = ResBlock(256, bottle_width=self.bottleneck_width, canadility=canadility)
        
        
        # 4
        self.bottleneck_width = self.bottleneck_width * 2
        assert 512 == canadility * self.bottleneck_width
        self.conv4 = DownConv(256, 512)
        self.res_block7 = ResBlock(512, bottle_width=16, canadility=canadility)
        self.res_block8 = ResBlock(512, bottle_width=16, canadility=canadility)
        
        ## fc
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.LLMfc = nn.Sequential(nn.Linear(512, fc_dim),
                                    nn.BatchNorm1d(fc_dim),
                                    nn.ReLU(),
                                    nn.Linear(fc_dim, num_cls))
                            

    def forward(self, x):
        # 0
        x0 = self.stem(x)

        # 1
        x = self.conv1(x0)
        x = self.res_block1(x)
        x1 = self.res_block2(x)

        # 2
        x = self.conv2(x1)
        x = self.res_block3(x)
        x2 = self.res_block4(x)

        # 3
        x = self.conv3(x2)
        x = self.res_block5(x)
        x3 = self.res_block6(x)

        # 3
        x = self.conv4(x3)
        x = self.res_block7(x)
        out = self.res_block8(x)

        # fc
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.LLMfc(out)
        return out




    
#if __name__=='__main__':
#    from torchsummary import summary
#    model = LLMFc_ResNeXt(16, bottleneck_width=2, canadility=32)
#    summary(model, (3, 260, 260))
#    inp = torch.randn(4, 3, 260, 260)
#    out = model(inp)
#    print(out.shape)


