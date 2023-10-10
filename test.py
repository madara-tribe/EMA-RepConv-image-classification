import os
import torch
import time
from tqdm import tqdm
from torch.utils import data
from models.models import call_RepConvResNeXt
from utils.augmentations import transforms_
from utils.dataloader import EvalDataLoader
from tools import create_cofusion_matrix, save_pred_img

def TTA(imgs:torch.Tensor):
    imgs2 = imgs.flip(2,3)
    imgs3 = torch.rot90(imgs,1,[2,3])
    imgs4 = torch.rot90(imgs,-1,[2,3])
    return imgs, imgs2, imgs3, imgs4

class BasePredictor:
    def __init__(self, cfg, device, weight_path):
        print("start predict")

    def create_data_loader(self, config):
        train_transform, val_transform = transforms_(config)
        val_dst = EvalDataLoader(root=config.root_valid, transform=val_transform)
        val_loader = data.DataLoader(
                    val_dst, batch_size=1, shuffle=None, num_workers=0, pin_memory=None)
        print(" Query set: %d" %(len(val_dst)))
        return val_loader, val_dst

    def load_trained_model(self, cfg, device, weight_path):
        model = call_RepConvResNeXt(cfg, device, deploy=False)
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        if cfg.half==1:
            model.half().float()
        return model

class Predictor(BasePredictor):
    def __init__(self, cfg, device, weight_path, tta=None):
        model = self.load_trained_model(cfg, device, weight_path)
        self.predict(model, cfg, device, tta=tta)
    
    def predict(self, model, cfg, device, tta=None):
        filename='results'
        sub_dir = 'miss'
        class_dir = os.path.join(filename, 'class')
        val_loader, _ = self.create_data_loader(cfg)
        model.eval()
        y_pred, y_test = [], []
        start = time.time()
        for i, (imgs, label, simg) in tqdm(enumerate(val_loader)):
            imgs = imgs.to(device=device)
            label = int(label.to('cpu').detach().numpy().copy())
            if tta:
                img1, img2, img3, img4 = TTA(imgs)
                pred1 = model(img1)
                pred2 = model(img2)
                pred3 = model(img3)
                pred4 = model(img4)
                pred = (pred1+pred2+pred3+pred4)/4
            else:
                pred = model(imgs)  
            pred = model(imgs)
            # brand accuracy
            pred = int(torch.argmax(pred).to('cpu').detach().numpy().copy())
            if label!=pred:
                save_pred_img(i, simg, label, pred, path=os.path.join(sub_dir, 'brand'))
            y_pred.append(pred)
            y_test.append(label)
        latency = time.time() - start
        print(f'Prediction Latency: {latency}') 
        #print(f"each accuracy [Brand]:{brand_acc/nums}")
        # cm
        target_name = [str(i)+'class' for i in range(69)]
        create_cofusion_matrix(y_test, y_pred, target_names=target_name, filename=label_dir)

