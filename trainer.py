import logging
import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchsummary import summary
from tqdm import tqdm
import tensorboardX as tbx
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2

from cfg import Cfg
from utils.dataloader import DataLoader
from utils.optimizers import create_optimizers
from utils.callback import CallBackModelCheckpoint
from utils import utils
from utils.loss import distilladtion_loss
from models.BaseModel import RecognitionModel
from models.LLM import LLMModel

class BasicTrainer:
    def create_data_loader(self, config, use_imagefolder=None):
        """ Dataset And Augmentation
        if 0 ~ 1 normalize, just use:
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
        """
        train_transform = A.Compose([
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            #Random Erasing
            A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])

        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
        if use_imagefolder:
            self.train_dst = ImageFolder("datasets/dataset", transform = train_transform)
            self.val_dst = ImageFolder("datasets/dataset", transform=val_transform)
        else:
            self.train_dst = DataLoader(config.X_train, config.y_train, transform=train_transform)
            self.val_dst = DataLoader(config.X_test, config.y_test, transform=val_transform)

        self.train_loader = data.DataLoader(
                self.train_dst, batch_size=config.train_batch, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = data.DataLoader(
                    self.val_dst, batch_size=config.val_batch, shuffle=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))

    def train(self, config, device, weight_path=None, model_type='llm'):
        # model
        if 'llm' in model_type:
            model = LLMModel(embedding_size=config.emmbed_size)
        else:
            model = RecognitionModel(embedding_size=config.emmbed_size, deploy=False, bottleneck_width=2, cardinality=4)
        model = model.to(device)
        model_ema = None
        if config.model_ema:
            adjust = config.world_size * config.train_batch * config.model_ema_steps / config.epochs
            alpha = 1.0 - config.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
        print(model)
         
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        #if torch.cuda.device_count() > 1:
           # backbone = nn.DataParallel(backbone)
        summary(model, (3, config.input_size, config.input_size))
                
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Learning rate:   {config.lr}
            Training size:   {len(self.train_dst)}
            Validation size: {len(self.val_dst)}
        ''')

        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer, lr_scheduler = create_optimizers(model, config)

        # 5. Begin training
        self.global_step = 0
        iters = len(self.train_loader)
        interval_loss = 0
        center, side1, side2 = 0, 0, 0
        print('Start training')
        for epoch in range(1, config.epochs+1):
            model.train()
            if 'llm' in model_type:
                self.train_one_epoch_loop(config, epoch, device, model, self.train_dst, self.train_loader, optimizer, self.criterion, self.tfwriter, model_ema=model_ema)
            else:
                self.train_one_epoch_loop(config, epoch, device, model, self.train_dst, self.train_loader, optimizer, self.tfwriter, model_ema=model_ema)


            #if global_step % config.val_interval == 0:
            if 'llm' in model_type:
                self.validate(model, self.val_loader, self.global_step, self.criterion, epoch, device, self.tfwriter, self.callback_checkpoint, ema=False)
                if config.model_ema:
                    self.validate(model_ema.module, self.val_loader, self.global_step, self.criterion, epoch, device, self.tfwriter, self.callback_checkpoint, ema=True)
            else:
                self.validate(model, self.val_loader, self.global_step, epoch, device, self.tfwriter, self.callback_checkpoint, ema=False)
                if config.model_ema:
                    self.validate(model_ema.module, self.val_loader, self.global_step, epoch, device, self.tfwriter, self.callback_checkpoint, ema=True)
            lr_scheduler.step()
      

class LLMTrainer(BasicTrainer):
    def __init__(self, config, device, num_workers, pin_memory, weight_path, model_type):
        global logging
        self.val_loss = 1e+5
        self.val_ema_loss = 1e+5
        self.setup_resouces(config, device, num_workers, pin_memory)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f'Using device {device}')
        self.train(config, device, weight_path=weight_path, model_type=model_type)
        
    def setup_resouces(self, config, device, num_workers, pin_memory):
        self.tfwriter = tbx.SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0)
        self.create_data_loader(config, use_imagefolder=None)
        self.callback_checkpoint = CallBackModelCheckpoint(config)
        
    def train_one_epoch_loop(self, cfg, epoch, device, model, train_dst, train_loader, optimizer, criterion, tfwriter, model_ema):
        interval_loss = 0
        with tqdm(total=int(len(train_dst)/cfg.train_batch), desc=f'Epoch {epoch}/{cfg.epochs}') as pbar:
            for i, (x_img, label) in enumerate(train_loader):
                x_img = x_img.to(device=device)
                label = label.to(device=device).long()
                _, pred = model(x_img)
                #print(x_img.shape, x_img.min(), x_img.max(), label, pred.shape)
                loss = criterion(pred, label)
                
                interval_loss += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            
                if i==10:
                    break
                if model_ema and i % cfg.model_ema_steps == 0:
                    model_ema.update_parameters(model)
                    if epoch < cfg.lr_warmup_epochs:
                        # Reset ema buffer to keep copying weights during warmup period
                        model_ema.n_averaged.fill_(0)

                pbar.update()
                self.global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                # Evaluation round
                if self.global_step % cfg.eval_step == 0:
                    nums = cfg.eval_step
                    tfwriter.add_scalar('train/loss', interval_loss/nums, self.global_step)

                    print("Epoch %d, Itrs %d, TrainLoss=%f" % (epoch, self.global_step, interval_loss/nums))
                    interval_loss = 0
                
    def validate(self, model, val_loader, global_step, criterion,
             epoch, device, tfwriter, callback_checkpoint, ema=False):
        interval_valloss = 0
        nums = 0
        acc = 0
        model.eval()
        print("validating .....")
        with torch.no_grad():
            for i, (x_val, y_val) in tqdm(enumerate(val_loader)):
                x_val = x_val.to(device=device)
                label = y_val.to(device=device)
                _, pred = model(x_val)
                
                valid_loss = criterion(pred, label)
                interval_valloss += valid_loss.item()
                nums += 1
                
            if ema:
                tfwriter.add_scalar('valid/ema_loss', interval_valloss/nums, global_step)
                if interval_valloss/nums < self.val_ema_loss:
                    self.val_ema_loss = interval_valloss/nums
                    callback_checkpoint(global_step, np.round(self.val_ema_loss, decimals=4), model, ema=True)
                    logging.info(f'EMA Model Checkpoint {epoch} saved! loss is {self.val_ema_loss}')

                
            else:
                tfwriter.add_scalar('valid/val_loss', interval_valloss/nums, global_step)
                if interval_valloss/nums < self.val_loss:
                    self.val_loss = interval_valloss/nums
                    callback_checkpoint(global_step, np.round(self.val_loss, decimals=4), model, ema=False)
                    logging.info(f'Model Checkpoint {epoch} saved! loss is {self.val_loss}')
            print("Epoch %d, Itrs %d, valid_Loss=%f" % (epoch, global_step, interval_valloss/nums))




class FeatureLearningTrainer(BasicTrainer):
    def __init__(self, config, device, num_workers, pin_memory, llm_weight_path, weight_path=None):
        global logging
        self.val_loss = 1e+5
        self.val_ema_loss = 1e+5
        self.setup_resouces(config, device, num_workers, pin_memory)
        
        self.llm_model = LLMModel(embedding_size=config.emmbed_size)
        self.llm_model.load_state_dict(torch.load(llm_weight_path, map_location=device), strict=False)
        self.llm_model.eval()
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f'Using device {device}')
        self.train(config, device, weight_path=weight_path, model_type='student')

    def setup_resouces(self, config, device, num_workers, pin_memory):
        self.tfwriter = tbx.SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.create_data_loader(config, use_imagefolder=None)
        self.callback_checkpoint = CallBackModelCheckpoint(config)
        
    def train_one_epoch_loop(self, cfg, epoch, device, model, train_dst, train_loader, optimizer, tfwriter, model_ema):
        interval_loss = 0
        middles, heads = 0, 0
        with tqdm(total=int(len(train_dst)/cfg.train_batch), desc=f'Epoch {epoch}/{cfg.epochs}') as pbar:
            for i, (x_img, label) in enumerate(train_loader):
                x_img = x_img.to(device=device)
                label = label.to(device=device).long()
                
                pred = model(x_img)
                #_, llm_pred = self.llm_model(x_img)
                #middle = F.adaptive_avg_pool2d(middle, output_size=17)
                #print(x_img.shape, x_img.min(), x_img.max(), label, pred.shape)
                #print(middle.shape, pred.shape, llm_middle.shape, llm_pred.shape)
                loss = distilladtion_loss(i, pred, label)
                #loss = dist_loss + pseudo_loss
                #middles += dist_loss.item()
                #heads += pseudo_loss.item()
                interval_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if model_ema and i % cfg.model_ema_steps == 0:
                    model_ema.update_parameters(model)
                    if epoch < cfg.lr_warmup_epochs:
                        # Reset ema buffer to keep copying weights during warmup period
                        model_ema.n_averaged.fill_(0)
                pbar.update()
                self.global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                 
                # Evaluation round
                if self.global_step % cfg.eval_step == 0:
                    nums = cfg.eval_step
                    tfwriter.add_scalar('train/loss', interval_loss/nums, self.global_step)
                    #tfwriter.add_scalar('train/middle', middles/nums, self.global_step)
                    #tfwriter.add_scalar('train/head', heads/nums, self.global_step)

                    #print("Epoch %d, Itrs %d, trainLoss=%fi, dist_loss=%f, pesudo_loss=%f" % (epoch, self.global_step, interval_loss/nums, middles/nums, heads/nums))
                    print("Epoch %d, Itrs %d, trainLoss=%f" % (epoch, self.global_step, interval_loss/nums))
                    interval_loss = 0
                    middles, heads = 0, 0
                
    def validate(self, model, val_loader, global_step,
             epoch, device, tfwriter, callback_checkpoint, ema=False):
        interval_valloss = 0
        middles, heads = 0, 0
        nums = 0
        acc = 0
        model.eval()
        print("validating .....")
        with torch.no_grad():
            for i, (x_val, y_val) in tqdm(enumerate(val_loader)):
                x_val = x_val.to(device=device)
                label = y_val.to(device=device)
                
                pred_ = model(x_val)
                #_, llm_pred_ = self.llm_model(x_val)
                #print(x_img.shape, x_img.min(), x_img.max(), label, pred.shape)
                #print(llm_middle_.shape, pred_.shape, llm_middle_.shape, llm_pred_.shape)
                loss = distilladtion_loss(i, pred_, label)
                #loss = dist_loss + pseudo_loss
                #middles += dist_loss.item()
                #heads += pseudo_loss.item()
                interval_valloss += loss.item()
                nums += 1
            if ema:
                tfwriter.add_scalar('valid/ema_loss', interval_valloss/nums, global_step)
                if interval_valloss/nums < self.val_ema_loss:
                    self.val_ema_loss = interval_valloss/nums
                    callback_checkpoint(global_step, np.round(self.val_ema_loss, decimals=4), model, ema=True)
                    logging.info(f'EMA Model Checkpoint {epoch} saved! loss is {self.val_ema_loss}')

                
            else:
                tfwriter.add_scalar('valid/val_loss', interval_valloss/nums, global_step)
                #tfwriter.add_scalar('valid/middle', middles/nums, global_step)
                #tfwriter.add_scalar('valid/head', heads/nums, global_step)
                
                if interval_valloss/nums < self.val_loss:
                    self.val_loss = interval_valloss/nums
                    callback_checkpoint(global_step, np.round(self.val_loss, decimals=4), model, ema=False)
                    logging.info(f'Model Checkpoint {epoch} saved! loss is {self.val_loss}')
            #print("Epoch %d, Itrs %d, valid_Loss=%f, dist_loss=%f, pesudo_loss=%f" % (epoch, global_step, interval_valloss/nums, middles/nums, heads/nums))
            print("Epoch %d, Itrs %d, valid_Loss=%f" % (epoch, global_step, interval_valloss/nums))
   

