
import open3d as o3d
import numpy as np
from scipy import cluster
from sklearn.cluster import DBSCAN, KMeans
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("mkdir success")
    else:
        print("dir exists")

COLOR = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).reshape(-1, 3)

areaID = 46

# area9 = np.loadtxt("../dataset/Area9.txt")
# area49 = np.loadtxt("../dataset/Area49.txt")
# area46 = np.loadtxt("../dataset/Area46.txt")
area = np.loadtxt("../dataset/Area" + str(areaID) + ".txt")

# X Y Z R G B Semantic_label Instance_label Fine-grained_building_category

print(area.shape)

# 'Vegetation': 1

tree_idx = []

for i in range(len(area)):
    el = area[i]
    if el[-3] == 1:
        tree_idx.append(i)

trees = area[tree_idx]

print(trees.shape)

clustering = DBSCAN(eps=1.5, min_samples=5).fit(trees[:, :3])
# kmeans = KMeans(n_clusters=200, random_state=0, n_init="auto").fit(trees[:, :3]) 
labels = clustering.labels_

print(set(labels))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

label_colors = []
for i in range(len(labels)):
    # print(COLOR[labels[i] % len(COLOR)])
    label_colors.append(COLOR[labels[i] % len(COLOR)])
label_colors = np.array(label_colors)
# print(labels.shape)
# print(trees[:,:3].shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(trees[:,:3])
pcd.colors = o3d.utility.Vector3dVector(label_colors)

# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("trees.pcd", pcd)

trees_component = [None] * len(set(labels))
for i in range(len(labels)):
    if trees_component[labels[i]] == None:
        trees_component[labels[i]] = []
    trees_component[labels[i]].append(trees[i])

area_path = "../gt_instance/trees_area" + str(areaID) + "/"
mkdir(area_path)

for id in range(len(trees_component)):
    np.savetxt(area_path + str(id) + ".txt", trees_component[id])

'''

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import os

def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

lmp = []
prefix = "../gt_instance/trees_area49/"
for file in os.listdir(prefix):
    print(file)
    pcd = o3d.io.read_point_cloud(prefix + file, format="xyz")
    print("------------------------ pcd attribute ------------------------")
    print(
        "has_colors:",
        pcd.has_colors(),
        "has_points:",
        pcd.has_points(),
    )
    points = np.asarray(pcd.points)

    num_points = points.shape[0]

    local_maxima_ind = []

    coords = points[:, 0:2]
    values = points[:, 2]
    tree = KDTree(coords, leaf_size=100)
    radius = 2
    seperate_radius = 0.5
    neighbors_ind = tree.query_radius(coords, r=radius)
    for i, n in enumerate(neighbors_ind):
        if values[i] == np.max(values[n]):  # 待选极大值点
            flag = True
            for index in local_maxima_ind:  # 如果跟之前某个极值点高度相等 且 距离过近，则排除
                if (
                    values[i] == values[index]
                    and distance(points[i], points[index]) < seperate_radius
                ):
                    flag = False
                    break
            if flag:
                local_maxima_ind.append(i)
    local_maxima_points = points[local_maxima_ind]
    for p in local_maxima_points:
        lmp.append(p)

np.save("../trees/local_maxiam_points.npy", np.array(lmp))


# local_maxima_colors = points[local_maxima_ind]
# local_maxima_pcd_data = np.hstack((local_maxima_points, local_maxima_colors))

# local_maxima_pcd = o3d.geometry.PointCloud()
# local_maxima_pcd.points = o3d.utility.Vector3dVector(local_maxima_points)
# o3d.visualization.draw_geometries([pcd])

# print("pcd colors.shape:", colors.shape)
# print("pcd points.shape:", points.shape)
# print("pcd points & colors example:", points[0], colors[0])

# pcd_data = np.hstack((points, colors))

# print("pcd hstack data:", pcd_data.shape)
# print("pcd hstack data example:", pcd_data[0])

# kmeans = KMeans(
#     n_clusters=local_maxima_points.shape[0],
#     init=local_maxima_points,
# )
# labels = kmeans.fit_predict(points)
# print("kmeans labels.shape:", labels.shape)
# print("kmeans label range:", labels.min(), labels.max())


def generate_color(label):
    random.seed(label)  # 根据分类标签设置随机数种子
    r = random.random()  # 生成[0, 1)之间的随机小数
    g = random.random()
    b = random.random()
    return [r, g, b]  # 返回RGB


# kmeans_colors_nparray = []

# for i in range(num_points):
#     kmeans_colors_nparray.append(generate_color(labels[i]))
# kmeans_colors_nparray = np.array(kmeans_colors_nparray)

# kmeans_pcd = o3d.geometry.PointCloud()
# kmeans_pcd.points = o3d.utility.Vector3dVector(points)
# kmeans_pcd.colors = o3d.utility.Vector3dVector(kmeans_colors_nparray)
# o3d.visualization.draw_geometries([kmeans_pcd])

# original_trees = np.loadtxt(prefix + "148.txt")

# trees_instance = [None] * len(set(labels))
# for i in range(len(labels)):
#     if trees_instance[labels[i]] == None:
#         trees_instance[labels[i]] = []
#     trees_instance[labels[i]].append(original_trees[i])


# print(len(cars_instance))

# for id in range(len(trees_instance)):
#     np.savetxt("../gt_instance/trees_/148_" + str(id) + ".txt", trees_instance[id])

print("Done")

'''