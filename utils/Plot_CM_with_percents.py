""" 
Date: April 17, 2021
Author: YIN Chao
Functionality: 
- read PSNet-5 dataset processed files (data/xxx/processed)
- plot the confusion matrix in normalized form
"""
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from util import write_ply, read_ply, plot_confusion_matrix_seaborn
from visualize import Plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


VISUALIZE_IN_OPEN3D =True
SAVE_PLY = False
DATA_DIR='log_eval/psnet/respointnet2_dp_fi_df_fc1_1617375451--yc/results'
point_clouds = ['Area_3']
# point_clouds = ['Area_1','Area_2','Area_3','Area_4']
# point_clouds = ['Area_1','Area_4']

image_path='utils/'

# read ply data to obtain y_true and y_pred
targets_list=[]
preds_list=[]
for cloud_name in point_clouds:
    gt_filename = os.path.join(DATA_DIR,f"{cloud_name}_gt.ply")
    pred_filename = os.path.join(DATA_DIR,f"{cloud_name}_pred.ply")

    data_gt = read_ply(gt_filename)
    y_true = data_gt['gt']
    targets_list.append(y_true)

    data_pred = read_ply(pred_filename)
    y_pred = data_pred['preds']
    preds_list.append(y_pred)

y_true=np.vstack(targets_list).squeeze()
y_pred=np.vstack(preds_list).squeeze() 

label_to_names = {0: 'ibeam',
                  1: 'pipe',
                  2: 'pump',
                  3: 'rectangularbeam',
                  4: 'tank'}
num_classes= len(label_to_names)
label_values = np.sort([k for k, v in label_to_names.items()])
target_names = list(label_to_names.values())
# cm=confusion_matrix(y_true,y_pred, np.arange(num_classes)) # confusion_matrix is just a np array

print(f"save confusion matrix at root/ {image_path}")
# plot w. seaborn style
plot_confusion_matrix_seaborn(
    y_true,
    y_pred,
    "CM_percent",
    label_values,
    label_to_names,
    image_path, 
    figsize=(10,10), normalized=True)    
