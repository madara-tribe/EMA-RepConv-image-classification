import os
import torch
import torch.utils.data as data
import numpy as np
import cv2

class DataLoader(data.Dataset):
    def __init__(self, x_img, y_label, transform=None):
        self.transform = transform
        self.X = np.load(x_img)
        self.y = np.load(y_label)
        assert (len(self.X) == len(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        image:
            must be numpy array type
        """
        
        image = self.X[index]
        y_label = self.y[index]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, y_label
