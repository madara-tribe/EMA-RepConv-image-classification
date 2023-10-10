import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from utils import img_padding

root="datasets/dataset"
nppath="datasets/npy"

os.makedirs(nppath, exist_ok=True)

def preprocess(p, size=224):
    img = cv2.imread(p)
    #img = cv2.resize(img, (size, size))
    img = img_padding(img, desired_size=size)
    return img

def ToNumpy(X_train, y_train, X_test, y_test):
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    np.save(os.path.join(nppath, "X_train"), X_train)
    np.save(os.path.join(nppath, "y_train"), y_train)
    np.save(os.path.join(nppath, "X_test"), X_test)
    np.save(os.path.join(nppath, "y_test"), y_test)

def balanced_prepare_data(cfg):
    X_train, y_train, X_test, y_test = [], [], [], []
    #path = os.path.join(root, '*.jpg')
    label_folders = os.listdir(root)
    label_folders = sorted(label_folders)
    for i, label in enumerate(label_folders):
        labels = int(label)
        num_files = len(glob.glob(os.path.join(root, str(labels), '*.jpg')))
        print(f"num_file at {labels} is {num_files}")    
        num_test = int(num_files*0.1)
        pathes = os.path.join(root, str(labels), '*.jpg')
        print(f"num_file at {labels} is {num_files}", num_test, pathes)
        for i, p in enumerate(glob.glob(pathes)):
            img = preprocess(p, size=cfg.input_size)
            if i < num_test:
                print('test', i, labels, p)
                X_test.append(img)
                y_test.append(labels)
            else:
                print('train', i, labels, p)
                X_train.append(img)
                y_train.append(labels)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    ToNumpy(X_train, y_train, X_test, y_test)


    
def prepare_data(cfg):
    X_train, X_test, y_train, y_test, stack = [], [], [], [], []
    label_names = os.listdir(root)
    label_names = sorted(label_names)
    for cls, label in enumerate(tqdm(label_names)):
        path = os.path.join(root, label, '*.jpg')
        print(cls, label, path)
        for p in glob.glob(path):
            img = preprocess(p, size=int(cfg.input_size))
            if cls not in stack:
                print('test 1', cls, label, path)
                stack.append(cls)
                X_test.append(img)
                y_test.append(cls)
            elif stack.count(cls)<10:
                print('test 2', cls, label, path)
                stack.append(cls)
                X_test.append(img)
                y_test.append(cls)
            else:
                print('train', cls, label, path)
                stack.append(cls)
                X_train.append(img)
                y_train.append(cls)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    ToNumpy(X_train, y_train, X_test, y_test)
