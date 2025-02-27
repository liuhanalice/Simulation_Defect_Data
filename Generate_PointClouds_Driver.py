import datetime
import random
import numpy as np
import open3d as o3d
import time
from Generate_PointClouds import generate_normal, generate_defects, generate_flat, generate_flat_defect
# import metrics
# from my_funcs import *
# from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
from sklearn.metrics import confusion_matrix
import warnings


def normalize_and_downSampling(pcd, label, npoints=2048):
    # normalize the point cloud
    point_set = np.asarray(pcd.points, dtype=np.float32)
    point_set = point_set - np.mean(point_set, axis=0)
    max_dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)))
    point_set = point_set / max_dist

    # down sampling
    choice = np.random.choice(len(point_set), npoints, replace=len(point_set) < npoints)
    point_set = point_set[choice, :]
    label = label[choice]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)
    return pcd, label


if __name__ == '__main__':
    # random seed
    np.random.seed(1)
    random.seed(1)
    
    generate_shape_type = "flat_defect" # "normal", "defect", "flat", "flat_defect"
    num_experiments = 10

    bg_ks = 0.0004 + np.random.uniform(-0.0002, 0.0002, num_experiments)
    bg_size = 20

    defect_poss = np.random.uniform(-1, 1, (2, num_experiments))
    defect_depths = 3. + np.random.uniform(0, 4., num_experiments)
    defect_radiuses = defect_depths * (0.8 + np.random.uniform(-0.1, 0.1, num_experiments)) 
    defect_transes = 1.4 + np.random.uniform(-0.1, 0.1, num_experiments)
    
    for i in range(num_experiments):
        # variation on bg_k and bg_size
        if generate_shape_type == "normal":
            print("Generating Normal Point Clouds ", i)

            pcd, label, _, _ = generate_normal(bg_k=bg_ks[i], bg_size=bg_size, bg_std_depth=0.1, bg_std_xy=0.02, spline_flag=False, spline_paras=None, spline_knot=None, num_p=150)

            o3d.io.write_point_cloud('./Normal/normal_' + str(i) + '.pcd', pcd)
            np.save('./Normal/label_normal_' + str(i), label)
        elif generate_shape_type == "defect": # generate surface with defect
            print("Generating Defect Point Clouds ", i)

            pcd, label, _, _ = generate_defects(bg_k=bg_ks[i], bg_size=bg_size, bg_std_depth=0.1, bg_std_xy=0.02, defect_pos=defect_poss[:, i:i+1], defect_depth=defect_depths[i], defect_radius=defect_radiuses[i], defect_trans=defect_transes[i], step=-0.35, spline_flag=False, spline_paras=None, spline_knot=None, num_p=150)
            o3d.io.write_point_cloud('./Defect/defect_' + str(i) + '.pcd', pcd)
            np.save('./Defect/label_defect_' + str(i), label)
            
            # Optional: also generate the normal point clouds (without defect) paired with the defect point clouds
            pcd_normal, label_normal, _, _ = generate_normal(bg_k=bg_ks[i], bg_size=bg_size, bg_std_depth=0.1, bg_std_xy=0.02, spline_flag=False, spline_paras=None, spline_knot=None, num_p=150)
            o3d.io.write_point_cloud('./Defect/normal_' + str(i) + '.pcd', pcd_normal)
            np.save('./Defect/label_normal_' + str(i), label_normal)
        elif generate_shape_type == "flat":
            print("Generating Flat Point Clouds ", i)

            pcd, label, _, _ = generate_flat(bg_size=bg_size, bg_std_depth=0.1, bg_std_xy=0.02, num_p=150)

            o3d.io.write_point_cloud('./Flat/flat_' + str(i) + '.pcd', pcd)
            np.save('./Flat/label_flat_' + str(i), label)
        elif generate_shape_type == "flat_defect":
            print("Generating Flat Defect Point Clouds ", i)

            pcd, label, _, _ = generate_flat_defect(bg_size=bg_size, bg_std_depth=0.1, bg_std_xy=0.02, defect_pos=defect_poss[:, i:i+1], defect_depth=defect_depths[i], defect_radius=defect_radiuses[i], defect_trans=defect_transes[i], num_p=150) 
            o3d.io.write_point_cloud('./Flat_Defect/flatDefect_' + str(i) + '.pcd', pcd)
            np.save('./Flat_Defect/label_flatDefect_' + str(i), label)

            # Optional: also generate the normal point clouds (without defect) paired with the defect point clouds
            pcd_normal, label_normal, _, _ = generate_flat(bg_size=bg_size, bg_std_depth=0.1, bg_std_xy=0.02, num_p=150)
            o3d.io.write_point_cloud('./Flat_Defect/flat_' + str(i) + '.pcd', pcd_normal)
            np.save('./Flat_Defect/label_flat_' + str(i), label_normal)
            
            # Optional: normalize and down sample the point clouds
            # processed_pcd, processed_label = normalize_and_downSampling(pcd, label, npoints=2048)
            # o3d.io.write_point_cloud('./Flat_Defect/norm2048/flatDefect_' + str(i) + '.pcd', processed_pcd)
            # np.save('./Flat_Defect/norm2048/label_flatDefect_' + str(i), processed_label)
            

