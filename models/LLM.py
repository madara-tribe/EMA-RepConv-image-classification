import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .pool_layers import _create_fc, _create_pool

model_name = "resnext50d_32x4d"

class _FinalLayers(nn.Module):
    def __init__(self, num_classes, num_features, pool_type="avg"):
        super(_FinalLayers, self).__init__()
        self.global_pool, num_pooled_features = _create_pool(num_features, pool_type=pool_type)
        self.fc = _create_fc(num_pooled_features, num_classes)

    def forward(self, x):
        out = self.global_pool(x)
        out = self.fc(out)
        return out

class LLMModel(nn.Module):
    def __init__(self, embedding_size):
        super(LLMModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True) 
        self.freeze_layers()
        self.replace_layer(embedding_size)

    def freeze_layers(self):
        for j, param in enumerate(self.model.parameters()):
            param.requires_grad = False
        
    def replace_layer(self, embedding_size):
        num_feature = 2048
        self.llm_layer4 = self.model.layer4
        self.model.global_pool = self.model.fc = self.model.layer4 = nn.Identity()
        self.head = _FinalLayers(embedding_size, num_feature)
        

    def forward(self, x):
        middle = self.model(x)
        out = self.llm_layer4(middle)
        out = self.head(out)
        return middle, out


