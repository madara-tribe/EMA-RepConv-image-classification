import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .pool_layers import _create_fc, _create_pool
from .RepConv import _RepConv
from .ResNext import ResNeXt

class _FinalLayers(nn.Module):
    def __init__(self, num_classes, num_features, pool_type="avg"):
        super(_FinalLayers, self).__init__()
        self.global_pool, num_pooled_features = _create_pool(num_features, pool_type=pool_type)
        self.fc = _create_fc(num_pooled_features, num_classes)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        #out = self.drop(x)
        out = self.global_pool(x)
        out = self.fc(out)
        return out    

class RecognitionModel(nn.Module):
    def __init__(self, embedding_size, deploy=False, bottleneck_width=2, cardinality=4):
        super(RecognitionModel, self).__init__()
        num_feature = 512
        self.deploy = deploy
        self.residual_model = ResNeXt(deploy=deploy, bottleneck_width=bottleneck_width, cardinality=cardinality)
        self.head = _FinalLayers(embedding_size, num_feature)
         

    def forward(self, x):
        out = self.residual_model(x)
        out = self.head(out)
        return out

#if __name__=='__main__':
 #   model = RecognitionModel(embedding_size=69, num_blocks=0, deploy=False)
  #  inp = torch.randn(4, 3, 260, 260, requires_grad=True)
   # m, out = model(inp)
    #print(m.shape, out.shape)
