import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.pin_memory=True
Cfg.num_worker = 4
Cfg.train_batch = 4
Cfg.val_batch = 1
Cfg.epochs = 50
Cfg.eval_step = 100
Cfg.val_interval = 2000
Cfg.gpu_id = '3'
Cfg.input_size=256

# EMA
Cfg.model_ema=True
Cfg.world_size = 1
Cfg.model_ema_steps=32
Cfg.model_ema_decay=0.99998
Cfg.lr_warmup_epochs=0 # "the number of epochs to warmup (default: 0)"


#Cfg.TRAIN_OPTIMIZER="sgd"
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.lr = 0.001
Cfg.lr_step_size = 30
Cfg.lr_gamma = 0.1

## dataset
Cfg.emmbed_size = 69
Cfg.X_train = "datasets/npy/X_train.npy"
Cfg.y_train = "datasets/npy/y_train.npy"
Cfg.X_test = "datasets/npy/X_test.npy"
Cfg.y_test = "datasets/npy/y_test.npy"
Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')
