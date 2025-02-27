import datetime
import random
import numpy as np
import open3d as o3d
import time
import sys
from sklearn.metrics import confusion_matrix
import warnings


def Visualization_pcd(pts, label, window_name='unspecified', scale=6):
    points = pts.copy()
    n = points.shape[0]
    pcd = o3d.geometry.PointCloud()
    points[:, 2] *= scale
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((n, 3), np.uint8)
    colors[label == 0] = np.array([[0, 0, 0]])
    colors[label == 1] = np.array([[0, 0, 255]])
    colors[label == 2] = np.array([[255, 0, 0]])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name, point_show_normal=False)

# This is a demo for single experiment
def POINTCLOUD_VIEWER():
    
    pcd =  o3d.io.read_point_cloud("./Flat_Defect/flatDefect_0.pcd")
    label = np.load("./Flat_Defect/label_flatDefect_0.npy")

    Visualization_pcd(np.asarray(pcd.points[:]).astype(np.float64), label, scale=1)
    

if __name__ == '__main__':
    # random seed
    np.random.seed(1)
    random.seed(1)

    # single test experiment
    POINTCLOUD_VIEWER()
