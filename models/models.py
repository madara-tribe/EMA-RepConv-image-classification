
def call_ResNeXt(cfg, device):
    from .ResNext import ResNeXt
    model = ResNeXt(num_cls=cfg.emmbed_size, bottleneck_width=cfg.bottleneck_width, canadility=cfg.cardinality).to(device)
    return model
    
def call_LLM(cfg, device):
    from .ResNext import LLMFc_ResNeXt
    model = LLMFc_ResNeXt(num_cls=cfg.emmbed_size, bottleneck_width=cfg.bottleneck_width, canadility=cfg.cardinality).to(device)
    return model

def call_RepConvResNeXt(cfg, device, deploy=False):
    from .RepConvResNeXt import RepConvResNeXt_
    model = RepConvResNeXt_(num_cls=cfg.emmbed_size, bottleneck_width=cfg.bottleneck_width, canadility=cfg.cardinality, deploy=deploy).to(device)
    return model

