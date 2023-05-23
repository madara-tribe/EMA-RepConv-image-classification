import argparse
import sys, os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataloader import DataLoader
from models.BaseModel import RecognitionModel
from cfg import Cfg    
from models.LLM import LLMModel


root="datasets/npy"

def TTA(imgs:torch.Tensor):
    imgs2 = imgs.flip(2,3)
    imgs3 = torch.rot90(imgs,1,[2,3])
    imgs4 = torch.rot90(imgs,-1,[2,3])
    return imgs, imgs2, imgs3, imgs4

class BasePredictor:
    def __init__(self, cfg, device, weight_path, model_type):
        #LLMModel = self.load_trained_model(cfg, device, weight_path, model_type)
        #self.predict(cfg, tta=False)
        print("start predict")
    def create_data_loader(self, config):
        val_transform = A.Compose([
                   A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                   ToTensorV2()])
        val_dst = DataLoader(config.X_test, config.y_test, transform=val_transform)
        val_loader = data.DataLoader(
                    val_dst, batch_size=1, shuffle=None, num_workers=0, pin_memory=None)
        print(" Query set: %d" %(len(val_dst)))
        return val_loader, val_dst

    def load_trained_model(self, cfg, device, weight_path, model_type):
        if 'llm' in model_type:
            model = LLMModel(embedding_size=cfg.emmbed_size)
        else:
            model = RecognitionModel(embedding_size=cfg.emmbed_size, num_blocks=0, deploy=True)
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(model)
        from torchsummary import summary
        summary(model, (3, cfg.input_size, cfg.input_size))
        return model

class LLMPredictor(BasePredictor):
    def __init__(self, cfg, device, weight_path, model_type):
        LLMModel = self.load_trained_model(cfg, device, weight_path, model_type)
        self.predict(LLMModel, cfg, tta=False)
    
    def predict(self, model, cfg, tta=None):
        val_loader, val_dst = self.create_data_loader(cfg)
        model.eval()
        num_query = len(val_dst)
        acc = 0
        for i, (x_val, y_val) in tqdm(enumerate(val_loader)):
            x_val = x_val.to(device=device)
            label = int(y_val.to('cpu').detach().numpy().copy())
            if tta:
                img1, img2, img3, img4 = TTA(x_val)
                _, pred1 = model(img1)
                _, pred2 = model(img2)
                _, pred3 = model(img3)
                _, pred4 = model(img4)
                _, pred = (pred1+pred2+pred3+pred4)/4
            else:
                _, pred = model(x_val)
            pred = torch.argmax(pred)
            pred = pred.to('cpu').detach().numpy().copy()
            #print(pred.shape, type(pred), label, type(label))
            acc += 1 if int(pred)==label else 0
        print("total accuracy is ", acc/num_query)

class FeatureLearningPredictor(BasePredictor):
    def __init__(self, cfg, device, weight_path, model_type):
        model = self.load_trained_model(cfg, device, weight_path, model_type)
        self.predict(model, cfg, tta=False)
    
    def predict(self, model, cfg, tta=None):
        val_loader, val_dst = self.create_data_loader(cfg)
        model.eval()
        num_query = len(val_dst)
        acc = 0
        for i, (x_val, y_val) in tqdm(enumerate(val_loader)):
            x_val = x_val.to(device=device)
            label = int(y_val.to('cpu').detach().numpy().copy())
            if tta:
                img1, img2, img3, img4 = TTA(x_val)
                pred1 = model(img1)
                pred2 = model(img2)
                pred3 = model(img3)
                pred4 = model(img4)
                pred = (pred1+pred2+pred3+pred4)/4
            else:
                pred = model(x_val)
            pred = torch.argmax(pred)
            pred = pred.to('cpu').detach().numpy().copy()
            #print(pred.shape, type(pred), label, type(label))
            acc += 1 if int(pred)==label else 0
        print("total accuracy is ", acc/num_query)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, default=None, help='weight path')
    parser.add_argument('-m', '--model_type', type=str, default='llm', help='weight path')
    opt = parser.parse_args()
    if opt.weight:
        weight_path = str(opt.weight)
    else:
        print("weight_path should be difined")
        sys.exit(1)
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'llm' in opt.model_type:
        LLMPredictor(cfg, device, weight_path, opt.model_type)
    else:
        FeatureLearningPredictor(cfg, device, weight_path, opt.model_type)






