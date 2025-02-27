import numpy as np
import open3d as o3d

def generate_normal(bg_k=0.0004, bg_size=20., bg_std_depth=0.1, bg_std_xy=0.02, spline_flag=False, spline_paras=None,
                 spline_knot=None, num_p=150):
    # numpy_array 2rd shape is the number of defects
    n_knot = spline_knot
    p_dist = 2 * bg_size / num_p
    points = np.zeros((num_p ** 2, 2))
    for i in range(num_p):
        for j in range(num_p):
            points[i * num_p + j, 0] = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            points[i * num_p + j, 1] = -bg_size + 2 * bg_size / num_p * j
    if spline_flag == True:
        knotB_u, knotB_v = Cubic_Knot_Generation(-bg_size / 1.5, bg_size / 1.5, -bg_size, bg_size, num_knot=n_knot)
        print("GENERATE BACKGROUND SPLINE!")
        B_b = BaseMatrix(points, points.shape[0], knotB_u, knotB_v)

        if spline_paras is not None:
            bg_delta_z = np.dot(B_b, spline_paras.reshape((-1, 1)))
        else:
            raise ValueError("spline_paras cannot be None when spline_flag is True")
    pts = []
    label = []
    surface_index = []
    delta_z = []
    for i in range(num_p):
        for j in range(num_p):
            coo_x = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            coo_y = -bg_size + 2 * bg_size / num_p * j
           
            if spline_flag:
                cur_point, cur_label, cur_surface, cur_delta_z = single_point_normal_SPLINE(coo_x, coo_y, bg_std_xy, bg_std_depth, bg_delta_z[i * num_p + j, 0])
            else:
                cur_point, cur_label, cur_surface, cur_delta_z = single_point_normal(coo_x, coo_y, bg_std_xy, bg_std_depth, bg_k)
            
            pts.append(cur_point)
            label.append(cur_label)
            surface_index.append(cur_surface)
            delta_z.append(cur_delta_z)

    label = np.array(label)
    surface_index = np.array(surface_index)
    delta_z = np.array(delta_z)
    defect_arr = np.array(pts)[:, :, 0]
    delta = np.zeros(defect_arr.shape)
    delta[:, 2] = np.ones((defect_arr.shape[0],)) * 50.
    defect_arr += delta
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(defect_arr[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((defect_arr.shape[0], 3)))
    return pcd, label, surface_index, delta_z


def single_point_normal(coo_x, coo_y, bg_std_xy, bg_std_depth, bg_k):
    delta_z = np.random.normal(scale=bg_std_depth)
    point = np.array(
        [[coo_x + np.random.normal(scale=bg_std_xy)], [coo_y + np.random.normal(scale=bg_std_xy)],
         [0. + delta_z]])
    label = 0
    surface = 0
    kb = (coo_x / 6) ** 2 * (coo_y / 6) * bg_k
    point += np.array([[0.], [0.], [kb * (coo_x ** 2 + coo_y ** 2)]])
    return point, label, surface, delta_z


def single_point_normal_SPLINE(coo_x, coo_y, bg_std_xy, bg_std_depth, bg_delta):
    delta_z = np.random.normal(scale=bg_std_depth)
    point = np.array(
        [[coo_x + np.random.normal(scale=bg_std_xy)], [coo_y + np.random.normal(scale=bg_std_xy)],
         [0. + delta_z]])
    label = 0
    surface = 0
    point += np.array([[0.], [0.], [bg_delta]])
    return point, label, surface, delta_z


def generate_defects(bg_k=0.0004, bg_size=20., bg_std_depth=0.1, bg_std_xy=0.02, defect_pos=[[0], [0]], defect_depth=3, defect_radius=2.4, defect_trans = 1.4, step=-0.35, spline_flag=False, spline_paras=None, spline_knot=None, num_p=150):
    # numpy_array 2rd shape is the number of defects
    n_knot = spline_knot
    p_dist = 2 * bg_size / num_p
    points = np.zeros((num_p ** 2, 2))
    for i in range(num_p):
        for j in range(num_p):
            points[i * num_p + j, 0] = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            points[i * num_p + j, 1] = -bg_size + 2 * bg_size / num_p * j
    if spline_flag == True:
        knotB_u, knotB_v = Cubic_Knot_Generation(-bg_size / 1.5, bg_size / 1.5, -bg_size, bg_size, num_knot=n_knot)
        print("GENERATE BACKGROUND SPLINE!")
        B_b = BaseMatrix(points, points.shape[0], knotB_u, knotB_v)

        if spline_paras is not None:
            bg_delta_z = np.dot(B_b, spline_paras.reshape((-1, 1)))
        else:
            raise ValueError("spline_paras cannot be None when spline_flag is True")
    pts = []
    label = []
    surface_index = []
    delta_z = []
    for i in range(num_p):
        for j in range(num_p):
            coo_x = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            coo_y = -bg_size + 2 * bg_size / num_p * j
            coo = np.array([[coo_x], [coo_y]], np.float64)
            dist = np.linalg.norm(coo - defect_pos, axis=0)

            min_dist = np.min(dist)
            if spline_flag:
                cur_point, cur_label, cur_surface, cur_delta_z = single_point_defect_SPLINE(coo_x, coo_y, min_dist, bg_std_xy, bg_std_depth, bg_k, defect_radius, defect_depth, defect_trans, step, p_dist, bg_delta_z[i * num_p + j, 0])
            else:
                cur_point, cur_label, cur_surface, cur_delta_z = single_point_defect(coo_x, coo_y, min_dist, bg_std_xy, bg_std_depth, bg_k, defect_radius, defect_depth, defect_trans, step, p_dist)

            pts.append(cur_point)
            label.append(cur_label)
            surface_index.append(cur_surface)
            delta_z.append(cur_delta_z)

    label = np.array(label)
    surface_index = np.array(surface_index)
    delta_z = np.array(delta_z)
    defect_arr = np.array(pts)[:, :, 0]
    delta = np.zeros(defect_arr.shape)
    delta[:, 2] = np.ones((defect_arr.shape[0],)) * 50.
    defect_arr += delta
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(defect_arr[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((defect_arr.shape[0], 3)))
    return pcd, label, surface_index, delta_z


def single_point_defect_SPLINE(coo_x, coo_y, min_dist, bg_std_xy, bg_std_depth, bg_k, defect_radius, defect_depth, defect_trans, step, p_dist, bg_delta):
    transition_depth = -defect_depth
    depth_trans = 0.5 * defect_depth
    delta_z = np.random.normal(scale=bg_std_depth)
    point = np.array(
        [[coo_x + np.random.normal(scale=bg_std_xy)], [coo_y + np.random.normal(scale=bg_std_xy)],
         [0. + delta_z]])
    label = 0
    surface = 0
    point += np.array([[0.], [0.], [bg_delta]])
    if min_dist >= defect_radius * defect_trans:
        pass
    elif min_dist < defect_radius:
        point += np.array([[0], [0], [-depth_trans - (defect_depth - depth_trans) * (
            np.sqrt(defect_radius ** 2 - min_dist ** 2)) / defect_radius]])
        label = 1
        surface = 1
    else:
        rt = defect_radius * defect_trans
        degree = 0.0
        k = (rt - min_dist) / (rt - defect_radius) * (1 - degree) + degree
        point += np.array([[0], [0], [step +
                                      k * (defect_trans * defect_radius - min_dist) / (
                                              (2 * defect_trans - 2) * defect_radius) * transition_depth]])
        label = 1
        surface = 2
    return point, label, surface, delta_z


def single_point_defect(coo_x, coo_y, min_dist, bg_std_xy, bg_std_depth, bg_k, defect_radius, defect_depth, defect_trans, step, p_dist):
    transition_depth = -defect_depth
    rand_index = np.random.uniform(0, 1)
    depth_trans = 0.5 * defect_depth
    delta_z = np.random.normal(scale=bg_std_depth)
    point = np.array(
        [[coo_x + np.random.normal(scale=bg_std_xy)], [coo_y + np.random.normal(scale=bg_std_xy)],
         [0. + delta_z]])
    label = 0
    surface = 0
    kb = (coo_x / 6) ** 2 * (coo_y / 6) * bg_k
    point += np.array([[0.], [0.], [kb * (coo_x ** 2 + coo_y ** 2)]])
    
    if min_dist >= defect_radius * defect_trans:
        pass
    elif min_dist < defect_radius:
        point += np.array([[0], [0], [-depth_trans - (defect_depth - depth_trans) * (
            np.sqrt(defect_radius ** 2 - min_dist ** 2)) / defect_radius]])
        label = 1
        surface = 1
    else:
        rt = defect_radius * defect_trans
        degree = 0.0
        k = (rt - min_dist) / (rt - defect_radius) * (1 - degree) + degree
        point += np.array([[0], [0], [step +
                                      k * (defect_trans * defect_radius - min_dist) / (
                                              (2 * defect_trans - 2) * defect_radius) * transition_depth]])
        label = 1
        surface = 2
    return point, label, surface, delta_z


def generate_flat(bg_size=20., bg_std_depth=0.1, bg_std_xy=0.02, num_p=150):
    p_dist = 2 * bg_size / num_p
    points = np.zeros((num_p ** 2, 2))
    for i in range(num_p):
        for j in range(num_p):
            points[i * num_p + j, 0] = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            points[i * num_p + j, 1] = -bg_size + 2 * bg_size / num_p * j
    pts = []
    labels = []
    surface_index = []
    delta_zs = []
    for i in range(num_p):
        for j in range(num_p):
            coo_x = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            coo_y = -bg_size + 2 * bg_size / num_p * j
            delta_z = np.random.normal(scale=bg_std_depth)
            point = np.array(
                [[coo_x + np.random.normal(scale=bg_std_xy)], [coo_y + np.random.normal(scale=bg_std_xy)],
                 [0. + delta_z]])
            label = 0
            surface = 0
            pts.append(point) # (n,3,1)
            labels.append(label)
            surface_index.append(surface)
            delta_zs.append(delta_z)
    labels = np.array(labels)
    surface_index = np.array(surface_index)
    delta_zs = np.array(delta_zs)
    defect_arr = np.array(pts)[:, :, 0] # (n,3)
    delta = np.zeros(defect_arr.shape)
    delta[:, 2] = np.ones((defect_arr.shape[0],)) * 50.
    defect_arr += delta # increase 50 to z axis
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(defect_arr[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((defect_arr.shape[0], 3)))
    return pcd, labels, surface_index, delta_zs


def generate_flat_defect(bg_size=20., bg_std_depth=0.1, bg_std_xy=0.02, defect_pos=[[0], [0]], defect_depth=3, defect_radius=2.4, defect_trans=1.4, step=-0.35, num_p=150):
    p_dist = 2 * bg_size / num_p
    depth_trans = 0.5 * defect_depth
    transition_depth = -defect_depth
    points = np.zeros((num_p ** 2, 2))
    for i in range(num_p):
        for j in range(num_p):
            points[i * num_p + j, 0] = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            points[i * num_p + j, 1] = -bg_size + 2 * bg_size / num_p * j
    pts = []
    labels = []
    surface_index = []
    delta_zs = []
    for i in range(num_p):
        for j in range(num_p):
            coo_x = -bg_size / 1.5 + 2 * bg_size / 1.5 / num_p * i
            coo_y = -bg_size + 2 * bg_size / num_p * j
            coo = np.array([[coo_x], [coo_y]], np.float64)
            dist = np.linalg.norm(coo - defect_pos, axis=0)
            min_dist = np.min(dist)

            delta_z = np.random.normal(scale=bg_std_depth)
            point = np.array(
                [[coo_x + np.random.normal(scale=bg_std_xy)], [coo_y + np.random.normal(scale=bg_std_xy)],
                 [0. + delta_z]])
            label = 0
            surface = 0
            if dist >= defect_radius * defect_trans:
                pass
            elif dist < defect_radius:
                point += np.array([[0], [0], [-depth_trans - (defect_depth - depth_trans) * (np.sqrt(defect_radius ** 2 - min_dist ** 2)) / defect_radius]])
                label = 1
                surface = 1
            else:
                rt = defect_radius * defect_trans
                degree = 0.0
                k = (rt - min_dist) / (rt - defect_radius) * (1 - degree) + degree
                point += np.array([[0], [0], [step + k * (defect_trans * defect_radius - min_dist) / (
                                                    (2 * defect_trans - 2) * defect_radius) * transition_depth]])
                label = 1
                surface = 2
            pts.append(point)
            labels.append(label)
            surface_index.append(surface)
            delta_zs.append(delta_z)
    labels = np.array(labels)
    surface_index = np.array(surface_index)
    delta_zs = np.array(delta_zs)
    defect_arr = np.array(pts)[:, :, 0]
    delta = np.zeros(defect_arr.shape)
    delta[:, 2] = np.ones((defect_arr.shape[0],)) * 50.
    defect_arr += delta
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(defect_arr[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((defect_arr.shape[0], 3)))
    return pcd, labels, surface_index, delta_zs


def BaseFunction(i, k, u, knot):
    Nik_u = 0
    if k == 1:
        if u >= knot[i] and u < knot[i + 1]:
            Nik_u = 1.0
        else:
            Nik_u = 0.0
    else:
        length1 = knot[i + k - 1] - knot[i]
        length2 = knot[i + k] - knot[i + 1]

        if not length1 and not length2:
            Nik_u = 0.0
        elif not length1:
            Nik_u = (knot[i + k] - u) / length2 * BaseFunction(i + 1, k - 1, u, knot)
        elif not length2:
            Nik_u = (u - knot[i]) / length1 * BaseFunction(i, k - 1, u, knot)
        else:
            Nik_u = (u - knot[i]) / length1 * BaseFunction(i, k - 1, u, knot) + \
                    (knot[i + k] - u) / length2 * BaseFunction(i + 1, k - 1, u, knot)
    return Nik_u


def Cubic_Knot_Generation(umin, umax, vmin, vmax, num_knot):
    knot_u = []
    for i in range(num_knot):
        if i <= 3:
            knot_u.append(umin)
        elif 3 < i < num_knot - 4:
            knot_u.append(umin + (umax - umin) / (num_knot - 4 - 3) * (i - 3))
        else:
            knot_u.append(umax)
    knot_v = []
    for i in range(num_knot):
        if i <= 3:
            knot_v.append(vmin)
        elif 3 < i < num_knot - 4:
            knot_v.append(vmin + (vmax - vmin) / (num_knot - 4 - 3) * (i - 3))
        else:
            knot_v.append(vmax)
    return knot_u, knot_v


# cubic B-spline
def BaseMatrix(Points, num, knot_u, knot_v, degree=3):
    num_BaseFunc = len(knot_u) - (degree + 1)
    k = num_BaseFunc ** 2
    B = np.zeros((num, k), np.float64)
    for i in range(num):
        # print(i)
        point = Points[i]
        for j in range(num_BaseFunc):
            for k in range(num_BaseFunc):
                B[i, j * num_BaseFunc + k] = BaseFunction(j, 4, point[0], knot_u) * BaseFunction(k, 4, point[1], knot_v)
    return B
