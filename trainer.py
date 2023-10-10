import logging
import math
import numpy as np
import torch
import torch.nn as nn

from torch.utils import data
from torchsummary import summary
from tqdm import tqdm
import tensorboardX as tbx
import torch.nn.functional as F
from utils.dataloader import DataLoader
from utils.optimizers import create_optimizers
from utils.callback import CallBackModelCheckpoint
from models.ema import EMA
from utils.augmentations import transforms_, mixup_, mixup_cutmix
from models.models import call_RepConvResNeXt


class BasicTrainer:
    def create_data_loader(self, config, use_imagefolder=None):
        """ Dataset And Augmentation
        if 0 ~ 1 normalize, just use:
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
        """
        train_transform, val_transform = transforms_(config)
        if use_imagefolder:
            from torchvision.datasets import ImageFolder
            self.train_dst = ImageFolder("datasets/dataset", transform = train_transform)
            self.val_dst = ImageFolder("datasets/dataset", transform=val_transform)
        else:
            self.train_dst = DataLoader(root=config.root_train, transform=train_transform)
            self.val_dst = DataLoader(root=config.root_valid, transform=val_transform)

        self.train_loader = data.DataLoader(
                self.train_dst, batch_size=config.train_batch, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = data.DataLoader(
                    self.val_dst, batch_size=config.val_batch, shuffle=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))

    def train(self, config, device, weight_path=None):
        model = call_RepConvResNeXt(config, device, deploy=False)
        if config.half==1:
            model.half().float()
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
        optimizer, self.lr_scheduler = create_optimizers(model, config)
        ema = EMA(model.parameters(), decay_rate=0.995, num_updates=0)

        # 5. Begin training
        self.global_step = 0
        print('Start training')
        for epoch in range(1, config.epochs+1):
            model.train()
            self.train_one_epoch_loop(config, epoch, device, model, self.train_dst, self.train_loader, optimizer, self.tfwriter, ema=ema)
            self.validate(model, self.val_loader, self.global_step, epoch, device, self.tfwriter, self.callback_checkpoint, ema=ema)
      
class Trainer(BasicTrainer):
    def __init__(self, config, device, num_workers, pin_memory, weight_path=None):
        global logging
        self.val_loss = 1e+5
        self.val_ema_loss = 1e+5
        self.setup_resouces(config, device, num_workers, pin_memory)
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f'Using device {device}')
        self.train(config, device, weight_path=weight_path)

    def setup_resouces(self, config, device, num_workers, pin_memory):
        self.tfwriter = tbx.SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.create_data_loader(config, use_imagefolder=None)
        self.callback_checkpoint = CallBackModelCheckpoint(config)
        
    def train_one_epoch_loop(self, cfg, epoch, device, model, train_dst, train_loader, optimizer, tfwriter, ema):
        interval_loss = 0
        iters = len(train_loader)
        with tqdm(total=int(len(train_dst)/cfg.train_batch), desc=f'Epoch {epoch}/{cfg.epochs}') as pbar:
            for i, (img, label) in enumerate(train_loader):
                img = img.to(device=device)
                label = label.to(device=device).long()
                if cfg.scale_size == 1:
                    gs = cfg.SCALE_SIZE[int(np.random.randint(0, 2, size=1))]
                    ns = [math.ceil(x + gs) for x in img.shape[2:]]
                    img = F.interpolate(img, size=ns, mode='bilinear', align_corners=False)
                    # print('pass', ns, imgs_.shape)
                pred = model(img)
                loss = self.criterion(pred, label)
                interval_loss += loss.item()
                loss.backward()
                optimizer.step()
                self.lr_scheduler.step(epoch + i / iters)
                optimizer.zero_grad()
                ema.update(model.parameters())

                pbar.update()
                self.global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # Evaluation round
                if self.global_step % cfg.eval_step == 0:
                    nums = cfg.eval_step
                    tfwriter.add_scalar('train/loss', interval_loss/nums, self.global_step)
                    print("Epoch %d, Itrs %d, trainLoss=%f" % (epoch, self.global_step, interval_loss/nums))
                    interval_loss = 0
                    
    def validate(self, model, val_loader, global_step,
             epoch, device, tfwriter, callback_checkpoint, ema):
        interval_valloss = 0
        nums = 0
        model.eval()
        print("validating .....")
        with torch.no_grad():
            for i, (img, y) in tqdm(enumerate(val_loader)):
                img = img.to(device=device)
                y = y.to(device=device).long()
                ema.store(model.parameters())
                ema.copy(model.parameters())
                pred = model(img)
                loss = self.criterion(pred, y)
               
                interval_valloss += loss.item()
                nums += 1
            tfwriter.add_scalar('valid/val_loss', interval_valloss/nums, global_step)
                
            if interval_valloss/nums < self.val_loss:
                self.val_loss = interval_valloss/nums
                callback_checkpoint(global_step, np.round(self.val_loss, decimals=4), model)
                logging.info(f'Model Checkpoint {epoch} saved! loss is {self.val_loss}')
            print("Epoch %d, Itrs %d, valid_Loss=%f" % (epoch, global_step, interval_valloss/nums))
   


