import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .sese import SCSEModule
model_name = "inception_v3"

class LLMModel(nn.Module):
    def __init__(self, opt):
        super(LLMModel, self).__init__()
        self.item_cls = opt.num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        self.freeze()
        self.replace()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for i, child in enumerate(self.model.children()):
            if i > len(list(self.model.children()))-3:
                print("{}st child layer {} is grad True".format(i, child))
                for param in child.parameters():
                    param.requires_grad = True

    def replace(self):
        num_feature=2048
        self.model.fc = self.model.global_pool = self.model.head_drop = nn.Identity()
        self.model.global_pool = nn.Sequential(
                    SCSEModule(num_feature),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1)
                    )
        self.model.fc = nn.Linear(num_feature, self.item_cls)

    def forward(self, x):
        out = self.model(x)
        return out

