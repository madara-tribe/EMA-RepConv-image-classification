import torch
import torch.nn as nn
import torch.nn.functional as F

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1

def _create_fc(num_features, num_classes):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc

def _create_pool(
        num_features: int,
        pool_type: str = 'avg',
        flatten: bool = False
        ):
    global_pool = SelectAdaptivePool2d(
        output_size = 1,
        pool_type=pool_type,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size,
            pool_type: str = 'fast',
    ):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        if pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.flatten = nn.Flatten(1) 

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x
    
    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)

def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


