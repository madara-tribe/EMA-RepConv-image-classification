import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.mixup import Mixup

def mixup_(cutmix=True):
    # https://timm.fast.ai/mixup_cutmix#Both-Mixup-and-Cutmix
    mixup_alpha = 0. if cutmix else 1.
    cutmix_alpha = 1.0 if cutmix else 0.
    mixup_args = {
    'mixup_alpha': mixup_alpha,
    'cutmix_alpha': cutmix_alpha,
    'cutmix_minmax': None,
    'prob': 1.0,
    'switch_prob': 0.,
    'mode': 'batch',
    'label_smoothing': 0,
    'num_classes': 10}
    mixup_fn = Mixup(**mixup_args)
    return mixup_fn

def mixup_cutmix():
    mixup_args = {
    'mixup_alpha': 0.3,
    'cutmix_alpha': 0.3,
    'cutmix_minmax': None,
    'prob': 1.0,
    'switch_prob': 0.5,
    'mode': 'elem',
    'label_smoothing': 0,
    'num_classes': 10}
    mixup_fn = Mixup(**mixup_args)
    return mixup_fn


def transforms_(opt):
    train_transform = A.Compose([
            A.Resize(opt.input_size, opt.input_size),
            A.HorizontalFlip(),
            A.Rotate((0, 45)),
            #A.RandomSizedCrop((10, 10)),
            # Random Erasing
            A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
    val_transform = A.Compose([
                A.Resize(opt.input_size, opt.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])
    return train_transform, val_transform



class TrainTransforms:
    def __init__(self, opt):
        self.transforms = A.Compose([
            A.Resize(opt.input_size, opt.input_size),
            A.HorizontalFlip(),
            A.RandomRotation((0, 45)),
            # Random Erasing
            A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

class ValidTransforms:
    def __init__(self, opt):
        self.transforms = A.Compose([
                A.Resize(opt.input_size, opt.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))



