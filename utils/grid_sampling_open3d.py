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
    cloud_points = data_gt['gt']
    cloud_color = data_gt['gt']

    data_pred = read_ply(pred_filename)
    y_pred = data_pred['preds']

    HACKING_RENDER_STYLE =True
    if VISUALIZE_IN_OPEN3D:
        xyzrgb = np.concatenate([cloud_points, cloud_colors], axis=-1)
        Plot.draw_pc(xyzrgb)  # visualize raw point clouds
        if HACKING_RENDER_STYLE:
            cloud_points = np.vstack((cloud_points,cloud_points[0]))
            cloud_classes = np.vstack((cloud_classes,np.array(4))) # 4 is tank class's id
        Plot.draw_pc_sem_ins(cloud_points, cloud_classes)  # visualize ground-truth
        # Plot.draw_pc_sem_ins(points, preds)  # visualize prediction