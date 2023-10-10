import glob
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class DataLoader(data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.full_path = glob.glob(root)

    def __len__(self):
        return len(self.full_path)

    def __getitem__(self, index): 
        p = self.full_path[index]
        img = pil_loader(p)
        label = int(self.full_path[index].split('/')[-1].split('_')[0])
        if self.transform is not None:
            image = self.transform(image=np.array(img))['image']
        return image, label


class EvalDataLoader(data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.full_path = glob.glob(root)

    def __len__(self):
        return len(self.full_path)

    def __getitem__(self, index):
        p = self.full_path[index]
        img = pil_loader(p)
        label = int(self.full_path[index].split('/')[-1].split('_')[0])
        if self.transform is not None:
            image = self.transform(image=np.array(img))['image']
        save_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return image, label, save_img


