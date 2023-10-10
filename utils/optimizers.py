import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def create_optimizers(model, cfg):
    if cfg.TRAIN_OPTIMIZER == 'adam':
        optimizer = torch.optim.AdamW(params=[
        {'params': model.parameters(), 'lr': 0.1*cfg.lr},
        ], lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08)
    elif cfg.TRAIN_OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': 0.1*cfg.lr},
        ], lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
    return optimizer, scheduler
