import sys, os
import torch 
from cfg import Cfg
from trainer import LLMTrainer, FeatureLearningTrainer


def main(config):
    model_type = "student"
    LLM_WEIGHT_PATH = 'tools/best_140800_2.1696_ema.pth'
    weight_path = None #'tools/best_140800_2.1696_ema.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'llm' in model_type:
        LLMTrainer(config, device, num_workers=cfg.num_worker, pin_memory=True, weight_path=weight_path, model_type=model_type)
    else:
        FeatureLearningTrainer(config, device, num_workers=cfg.num_worker, pin_memory=True, llm_weight_path=LLM_WEIGHT_PATH, weight_path=weight_path)


if __name__ == '__main__':
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    weight_path = None
    try:
        main(cfg)
    except KeyboardInterrupt:
        sys.exit(1)
        raise

