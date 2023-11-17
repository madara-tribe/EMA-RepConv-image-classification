import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.pin_memory=True
Cfg.num_worker = 8
Cfg.train_batch = 4
Cfg.val_batch = 1
Cfg.epochs = 50
Cfg.eval_step = 100
Cfg.input_size=260
Cfg.label_smoothing = 0

# model
Cfg.bottleneck_width = 2
Cfg.cardinality=32

## hyperparameter
Cfg.half = 0
Cfg.scale_size = 1
Cfg.cutmix = 1
Cfg.cutmixup_on = 0
Cfg.SCALE_SIZE = [0, 30]
#Cfg.TRAIN_OPTIMIZER="sgd"
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.lr = 0.01

Cfg.lr_step_size = 30
Cfg.lr_gamma = 0.1
Cfg.T_0 = 50
Cfg.T_mult = 2
Cfg.eta_min =0.001
Cfg.tta = True
## dataset
ROOT = "datasets/dataset"
Cfg.num_class = 69
Cfg.root_train = os.path.join(ROOT, 'train/*.jpg')
Cfg.root_valid = os.path.join(ROOT, 'valid/*.jpg')
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')


