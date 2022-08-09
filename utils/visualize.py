from open3d import linux as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = open3d.PointCloud()
        pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins
