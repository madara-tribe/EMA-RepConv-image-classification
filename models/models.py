
def call_RepConvResNeXt(cfg, device, deploy=False):
    from .RepConvResNeXt import RepConvResNeXt_
    model = RepConvResNeXt_(num_cls=cfg.num_class, bottleneck_width=cfg.bottleneck_width, canadility=cfg.cardinality, deploy=deploy).to(device)
    return model
