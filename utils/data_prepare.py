import cv2
import os
import json
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import img_padding
from cfg import Cfg

root="datasets/dataset"
nppath="datasets/npy"
meta_label = ['retriever', 'terrier', 'poodle', 'mastiff','collie', 'sheepdog', 'spaniel', 'setter', 'schnauzer', 'hound']

os.makedirs(nppath, exist_ok=True)

def preprocess(p, size=224):
    img = cv2.imread(p)
    #img = cv2.resize(img, (size, size))
    img = img_padding(img, desired_size=size)
    return img

def ToNumpy(X, y, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    np.save(os.path.join(nppath, "X_train"), X_train)
    np.save(os.path.join(nppath, "y_train"), y_train)
    np.save(os.path.join(nppath, "X_test"), X_test)
    np.save(os.path.join(nppath, "y_test"), y_test)

def main():
    imgs, labels = [], []
    label_names = os.listdir(root)
    label_names = sorted(label_names)
    for cls, label in enumerate(tqdm(label_names)):
        path = os.path.join(root, label, '*.jpg')
        print(cls, label, path)
        for p in glob.glob(path):
            img = preprocess(p, size=260)
            imgs.append(img)
            labels.append(int(cls))
    
    X = np.array(imgs)
    y = np.array(labels)
    ToNumpy(X, y, test_size=0.05)

if __name__=='__main__':
    main()

