import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
    
def create_cofusion_matrix(y_test, y_pred, target_names, filename='results'):
    os.makedirs(filename, exist_ok=True)
    matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrix, cmap='Blues', annot=True, fmt='d')
    plt.savefig(os.path.join(filename, 'confusion_matrix.png'))
    plt.clf()
    # classification_report
    report = classification_report(y_pred=y_pred, y_true=y_test, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(filename, 'accuracy.csv'))


def save_pred_img(i, tensor_img, label, pred, path=None):
    os.makedirs(os.path.join(path, str(label)), exist_ok=True)
    img = tensor_img.to('cpu').detach().numpy().copy()
    img = cv2.resize(np.squeeze(img, axis=0), (260, 260))
    #print(img.shape, img.max(), img.min())
    cv2.imwrite(os.path.join(path, str(label), '{}pred_{}.png'.format(i, str(pred))), img.astype(np.float32))


