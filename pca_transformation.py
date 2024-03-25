import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import os
import json

# [X, Y, Z]
IGNORE_SCALE = [True, True, False]

def to_stdXYZ_matrix(coord_system):
    matrix = np.hstack((coord_system, np.array([[0], [0], [0]])))
    matrix = np.vstack((matrix, np.array([0, 0, 0, 1])))
    return matrix


def to_scale_matrix(scale):
    return np.array(
        [[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]]
    )


def to_translate_matrix(translate):
    return np.array(
        [
            [1, 0, 0, translate[0]],
            [0, 1, 0, translate[1]],
            [0, 0, 1, translate[2]],
            [0, 0, 0, 1],
        ]
    ).T


def cartesian_to_homogenous(cartesian_array):
    # cartesian_array: [N, 3]
    return np.hstack((cartesian_array, np.ones((cartesian_array.shape[0], 1))))


def homogenous_to_cartesian(homo_array):
    cartesian_array = []
    for e in homo_array:
        cartesian_array.append(e[:-1] / e[-1])
    return np.array(cartesian_array)


# def define_axis_of_original_points(original_points):
#     # 1.使用主成分分析获取三个主成分轴
#     # 2.规定三维坐标系的向量
#     # 3.对主成分轴进行微调，拟合到规定的向量上
#     # 4.记录微调所需的参数
#     return 0


def correct_pca_components(pca_components, bounding_box):
    correct_components = []
    correct_box = []
    z_idx = np.argmax(np.abs(np.dot(pca_components, [0, 0, 1])))
    z_axis = pca_components[z_idx]
    z_scale = bounding_box[z_idx]
    if z_axis[2] < 0:
        z_axis[2] = -z_axis[2]

    another_2_axis = []
    another_2_scale = []
    for idx in range(3):
        if idx != z_idx:
            another_2_axis.append(pca_components[idx])
            another_2_scale.append(bounding_box[idx])
    
    if np.dot(np.cross(another_2_axis[0], another_2_axis[1]), z_axis) < 0:
        another_2_axis[1] = -another_2_axis[1]
    
    correct_components.append(another_2_axis[0])
    correct_components.append(another_2_axis[1])
    correct_components.append(z_axis)

    correct_box.append(another_2_scale[0])
    correct_box.append(another_2_scale[1])
    correct_box.append(z_scale)

    return np.array(correct_components), np.array(correct_box)


# def fix_instance_components(pca_components, bounding_box, base_components, base_box):
#     correct_components = []
#     correct_box = []
#     another_2_axis = []
#     another_2_scale = []

#     z_idx = np.argmax(np.abs(np.dot(pca_components, base_components[2])))
#     z_axis = pca_components[z_idx]
#     z_scale = bounding_box[z_idx]
#     if z_axis[2] < 0:
#         z_axis[2] = -z_axis[2]

#     for idx in range(3):
#         if idx != z_idx:
#             another_2_axis.append(pca_components[idx])
#             another_2_scale.append(bounding_box[idx])

#     if np.dot(np.cross(another_2_axis[0], another_2_axis[1]), z_axis) < 0:
#         another_2_axis[1] = -another_2_axis[1]

#     correct_components.append(another_2_axis[0])
#     correct_components.append(another_2_axis[1])
#     correct_components.append(z_axis)

#     correct_box.append(base_box[0] if IGNORE_SCALE[0] else bounding_box[x_idx])
#     correct_box.append(base_box[1] if IGNORE_SCALE[1] else bounding_box[y_idx])
#     correct_box.append(base_box[2] if IGNORE_SCALE[2] else bounding_box[z_idx])
   
#     return np.array(correct_components), np.array(correct_box)


def get_base_points(original_pcd):
    original_points = np.asarray(original_pcd.points)

    # rotate
    pca = PCA(n_components=3)  # XYZ: 3 components
    pca.fit(original_points)

    # scale
    box = original_pcd.get_oriented_bounding_box()

    correct_components, correct_scale = correct_pca_components(pca.components_, box.extent)
    # correct_components = pca.components_

    print("base pca components:", correct_components)
    print("base scale:", correct_scale)

    rotate_m = to_stdXYZ_matrix(correct_components.T)

    # translate
    trans_m = to_translate_matrix(-box.center)

    # scale
    scale_m = to_scale_matrix(1.0 / correct_scale)

    m = trans_m @ rotate_m @ scale_m 

    homo_original_points = cartesian_to_homogenous(original_points)
    base_points = homogenous_to_cartesian(homo_original_points @ m)

    return base_points, correct_components, correct_scale


def get_instanced_points_and_matrix(base_points, gt_pcd, base_components=[], base_scale=[]):
    gt_points = np.asarray(gt_pcd.points)

    # scale
    box = gt_pcd.get_oriented_bounding_box()

    # rotate
    if len(base_components) == 0 and len(base_scale) == 0:
        pca = PCA(n_components=3)  # XYZ: 3 components
        pca.fit(gt_points)
        correct_components, correct_scale = correct_pca_components(pca.components_, box.extent)
    else:
        correct_components = base_components
        correct_scale = base_scale

    print("ground truth pca components:", correct_components)
    print("ground truth scale:", correct_scale)

    rotate_m = to_stdXYZ_matrix(correct_components)

    scale_m = to_scale_matrix(correct_scale)

    trans_m = to_translate_matrix(box.center)

    # For the base points:
    #   1. rotate to ground truth coords
    #   2. scale to ground truth
    #   3. translate to ground truth
    matrix = scale_m @ rotate_m @ trans_m

    homo_base_points = cartesian_to_homogenous(base_points)
    homo_transformed_points = homo_base_points @ matrix
    cartesian_transformed_points = homogenous_to_cartesian(homo_transformed_points)
    # print("------------compare------------")
    # print("result:", cartesian_transformed_points)
    # print("G.T.:", gt_points)

    return cartesian_transformed_points, matrix


# base = 313
# base = 302
# base = 324
# base = 543
base = 511

# instances = [308, 309, 311, 310, 313, 312, 315, 314, 316, 317]
# instances = [300, 301, 302, 303, 304]
# instances = [296, 307, 318, 325, 324, 323]
# instances = [543, 544, 545, 506]
instances = [511, 510, 509, 508, 507, 980, 981]

dump_data = []

original_pcd = o3d.io.read_point_cloud("./gt_instance/buildings_area49/" + str(base) + ".txt", format="xyz")
base_points, correct_components, correct_scale = get_base_points(original_pcd)
# print("base points:", base_points)
base_pc = o3d.geometry.PointCloud()
base_pc.points = o3d.utility.Vector3dVector(base_points)
o3d.io.write_point_cloud("./buildings/base/" + str(base) + ".pcd", base_pc)

# containing itself
for instance in instances:
    pcd = o3d.io.read_point_cloud("./gt_instance/buildings_area49/" + str(instance) + ".txt", format="xyz")
    instanced_points, transform_matrix = get_instanced_points_and_matrix(
        base_points, pcd
    )
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(instanced_points)
    o3d.io.write_point_cloud("./buildings/transformed/base_511/" + str(instance) + ".pcd", pc)
    dump_data.append(transform_matrix.tolist())

with open("./buildings/base_511_matrices.json", "w") as file:
    json.dump(dump_data, file)


# # small_base_tree = '69'
# middle_base_tree = '46'

# # small_trees_instances = ['112', '28', '123', '72', '143', '6', '50', '18', '139', '83', '51', '120', '34', '8', '85', '73', '47', '35', '80', '91', '98', '140', '41', '45', '137', '33', '11', '56', '62', '158', '77', '52', '79', '31', '7', '61', '133', '109', '145', '141', '129', '131', '107', '58', '16', '82', '75', '76', '60', '103', '81', '29', '1', '14', '69', '10', '22', '92', '90', '65', '106', '117', '59', '132', '42', '157', '147', '122', '30', '108', '110', '135', '113', '19', '9', '68', '84', '15', '78', '156', '70', '111_0', '111_1', '48', '134', '136', '155', '66', '43', '12', '96', '13', '32', '0', '125', '67', '17', '105', '102']
# middle_trees_instances = ['148_0', '148_1', '146_0', '146_1', '146_2', '146_3', '40_0', '40_1', '40_2', '40_3', '44', '100', '86', '25', '150', '124', '46', '39_0', '39_1', '21', '149', '89', '121', '63_0', '63_1', '142_0', '142_1', '142_2', '54', '126', '74', '95', '2', '36', '118', '154', '104_0', '104_1', '104_2', '116', '55', '153_0', '153_1', '153_2', '94', '99', '37', '23', '119', '64']

# dump_data = []

# original_pcd = o3d.io.read_point_cloud("./gt_instance/trees_/" + middle_base_tree + ".txt", format="xyz")
# base_points, correct_components = get_base_points(original_pcd)
# # print("base points:", base_points)
# base_pc = o3d.geometry.PointCloud()
# base_pc.points = o3d.utility.Vector3dVector(base_points)
# o3d.io.write_point_cloud("./trees/base/" + middle_base_tree + ".pcd", base_pc)

# # containing itself
# for instance in middle_trees_instances:
#     pcd = o3d.io.read_point_cloud("./gt_instance/trees_/" + instance + ".txt", format="xyz")
#     instanced_points, transform_matrix = get_instanced_points_and_matrix(
#         base_points, pcd, correct_components
#     )
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(instanced_points)
#     o3d.io.write_point_cloud("./trees/transformed/" + instance + ".pcd", pc)
#     dump_data.append(transform_matrix.tolist())

# with open("./trees/base_46_matrices.json", "w") as file:
#     json.dump(dump_data, file)


# dump_data = []

# original_pcd = o3d.io.read_point_cloud("./gt_instance/cars_area9/220.txt", format="xyz")
# base_points, correct_components, correct_scale = get_base_points(original_pcd)
# base_pc = o3d.geometry.PointCloud()
# base_pc.points = o3d.utility.Vector3dVector(base_points)
# o3d.io.write_point_cloud("./cars/base/220.pcd", base_pc)

# for file in os.listdir("./gt_instance/cars_area9/"):
#     print(file)
#     pcd = o3d.io.read_point_cloud("./gt_instance/cars_area9/" + file, format="xyz")
#     if len(np.asarray(pcd.points)) < 50:
#         continue
#     instanced_points, transform_matrix = get_instanced_points_and_matrix(
#         base_points, pcd
#     )
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(instanced_points)
#     o3d.io.write_point_cloud("./cars/transformed/base_220/" + file[:-4] + ".pcd", pc)
#     dump_data.append(transform_matrix.tolist())

# with open("./cars/base_220_matrices.json", "w") as file:
#     json.dump(dump_data, file)

# print("Done")



# test
# original_pcd = o3d.io.read_point_cloud("./gt_instance/buildings_/300.txt", format="xyz")
# base_points = get_base_points(original_pcd)

# test_pcd = o3d.io.read_point_cloud("./gt_instance/buildings_/303.txt", format="xyz")
# test_points = get_base_points(test_pcd)