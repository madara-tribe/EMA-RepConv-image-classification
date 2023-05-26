import argparse
import sys
import os
import torch 
from cfg import Cfg
from trainer import Trainer
from test import Predictor
from utils.data_prepare import prepare_data

def main(config, opt):
    weight_path = opt.weight_path #'tools/best_140800_2.1696_ema.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.mode=='data':
        prepare_data(cfg)
    elif opt.mode=='train':
        Trainer(config, device, num_workers=cfg.num_worker, pin_memory=True, weight_path=weight_path)
    elif opt.mode=='eval':
        Predictor(cfg, device, weight_path, tta=None)
    else:
        print('--mode sholud be [train]/[eval]/[data]')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_path', type=str, default=None, help='ttrained weights path')
    parser.add_argument('--mode', type=str, default='train', help='train / eval/ data')
    opt = parser.parse_args()
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    try:
        main(cfg, opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise


