import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils import plotsample, plotsample_cmap

def curvy_z(x, y):
    return -0.023 * y ** 2 if y < 0 else 0

def generate_normal_surface(bg_size=20., bg_std_depth=0.05, bg_std_xy=0.02, num_p=150,
                            z_func=None, z_func_args=None):
    if z_func is None:
        z_func = lambda x, y: 0
    if z_func_args is None:
        z_func_args = {}
    # use provided code's xy dimensions
    xyz = np.empty((num_p ** 2, 3))
    labels = np.zeros(num_p ** 2)
    surface_index = np.zeros(num_p ** 2)
    delta_zs = np.empty(num_p ** 2)
    for i in range(num_p):
        for j in range(num_p):
            coo_x = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            coo_y = -bg_size + 2 * bg_size / num_p * j
            coo_z = z_func(coo_x, coo_y, **z_func_args)
            delta_z = np.random.normal(scale=bg_std_depth)
            xyz[i * num_p + j] = [coo_x + np.random.normal(scale=bg_std_xy),
                                  coo_y + np.random.normal(scale=bg_std_xy),
                                  coo_z + delta_z]
            delta_zs[i * num_p + j] = delta_z
    
    return xyz, labels, surface_index, delta_zs

def regular_defect_translator(xy, defect_pos, defect_depth=3, transition_depth_ratio=0.5, step=0.25,
                              defect_bot_radius=2.4, radius_top_bot_ratio=1.2):
    # params
    assert 0 < transition_depth_ratio < 1, "Invalid transition depth ratio"
    assert radius_top_bot_ratio > 1, "Invalid radius ratio"
    dist = np.linalg.norm(xy - defect_pos)
    defect_top_radius = defect_bot_radius * radius_top_bot_ratio
    defect_trans_depth = defect_depth * transition_depth_ratio
    # compute move
    if dist > defect_top_radius:
        return 0, 0
    elif dist < defect_bot_radius:
        h = np.sqrt(defect_bot_radius ** 2 - dist ** 2) 
        move = - (defect_trans_depth + (defect_depth - defect_trans_depth) * h / defect_bot_radius)
        return move, 1
    else:
        k = (defect_top_radius - dist) / (defect_top_radius - defect_bot_radius)
        move = - (step + 0.9 * k**2 * defect_trans_depth)
        return move, 2

def generate_defect_on_surface(plane_xyz: np.ndarray, plane_labels: np.ndarray, surface_index: np.ndarray,
                               defect_pos=[0, 0], def_func=None, def_args=None, move_via_normal=False):
    if def_func is None:
        def_func = regular_defect_translator
    if def_args is None:
        def_args = {}
    # get normals of the surface
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plane_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=100))
    surface_normals = np.asarray(pcd.normals)
    surface_normals[surface_normals[:,2] < 0] *= -1
    # sample
    sample_xyz = plane_xyz.copy()
    sample_labels = plane_labels.copy()
    sample_suface = surface_index.copy()
    # generate defect
    for i in range(len(plane_xyz)):
        move, surface_idx = def_func(plane_xyz[i,:2], defect_pos, **def_args)
        if surface_idx > 0: 
            sample_labels[i] = 1
            sample_suface[i] = surface_idx
            # move
            if move_via_normal:
                sample_xyz[i] += move * surface_normals[i]
            else:
                sample_xyz[i,2] += move
    # TODO: smooth the surface boundary

    return sample_xyz, sample_labels, sample_suface
    

if __name__ == "__main__":
    # generate normal surface
    xyz, labels, surface_index, delta_zs = generate_normal_surface(z_func=curvy_z)
    # plotsample(xyz, labels, show=True)
    # # generate defect
    sample_xyz, sample_labels, sample_suface = generate_defect_on_surface(xyz, labels, surface_index,
                                                                          defect_pos=[0, -10], move_via_normal=True)
    plotsample(sample_xyz, sample_labels, show=True)
    # plotsample(sample_xyz[sample_labels == 1], sample_suface[sample_labels == 1], show=True, view=[0,180])