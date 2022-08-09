""" 
Date: April 12, 2021
Author: YIN Chao
Functionality: 
- read PSNet-5 dataset processed files (data/xxx/processed)
- write each point cloud to ply files with xyzrgb and xyzLabel format
- show each point cloud in open3e
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

from util import write_ply
from visualize import Plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


VISUALIZE_IN_OPEN3D =True
SAVE_PLY = False
DATA_DIR='data/PSNet/PSNet_reduced_5/processed'
point_clouds = ['Area_3']
# point_clouds = ['Area_1','Area_2','Area_3','Area_4']
# point_clouds = ['Area_1','Area_4']


# read pkl data
for cloud_name in point_clouds:
    filename= os.path.join(DAntrol system for UNITA_DIR, f'{cloud_name}.pkl')
    with open(filename, 'rb') as f:
        (cloud_points, cloud_colors, cloud_classes) = pickle.load(f)
        print(f"{filename} loaded successfully")

        # save xyzrgb and xyzLabel to ply file
        if SAVE_PLY:
                xyzRgb_name = os.path.join(DATA_DIR, 'cloudcompare', f'{cloud_name}_xyzRgb.ply')
                xyzLabel_name = os.path.join(DATA_DIR, 'cloudcompare', f'{cloud_name}_xyzLabel.ply')
                # save xyz + rgb to ply
                write_ply(xyzRgb_name,
                        [cloud_points, cloud_colors],
                        ['x', 'y', 'z', 'r', 'g', 'b'])
                print(f"{cloud_name}_xyzRgb.ply saved successfully")


                # save xyz + label to ply
                write_ply(xyzLabel_name,
                        [cloud_points, cloud_classes],
                        ['x', 'y', 'z', 'gt'])
                print(f"{cloud_name}_xyzLabel.ply saved successfully")

        # HACK: to show the figure with same rendering style, add 1 tank point since area_1 and area_4 do not have tank points(semantic label class is 4)
        HACKING_RENDER_STYLE =True
        if VISUALIZE_IN_OPEN3D:
            xyzrgb = np.concatenate([cloud_points, cloud_colors], axis=-1)
            Plot.draw_pc(xyzrgb)  # visualize raw point clouds
            if HACKING_RENDER_STYLE:
                cloud_points = np.vstack((cloud_points,cloud_points[0]))
                cloud_classes = np.vstack((cloud_classes,np.array(4))) # 4 is tank class's id
            Plot.draw_pc_sem_ins(cloud_points, cloud_classes)  # visualize ground-truth
            # Plot.draw_pc_sem_ins(points, preds)  # visualize prediction
