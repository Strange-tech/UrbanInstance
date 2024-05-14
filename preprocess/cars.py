
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

# 'Vehicle': 4

car_idx = []

for i in range(len(area)):
    el = area[i]
    if el[-3] == 4:
        car_idx.append(i)

cars = area[car_idx]

print(cars.shape)

clustering = DBSCAN(eps=0.5, min_samples=5).fit(cars[:, :3])
# kmeans = KMeans(n_clusters=200, random_state=0, n_init="auto").fit(cars[:, :3]) 
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
print(labels.shape)
print(cars[:,:3].shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cars[:,:3])
pcd.colors = o3d.utility.Vector3dVector(label_colors)

# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("cars.pcd", pcd)

cars_instance = [None] * len(set(labels))
for i in range(len(labels)):
    if cars_instance[labels[i]] == None:
        cars_instance[labels[i]] = []
    cars_instance[labels[i]].append(cars[i])

area_path = "../gt_instance/cars_area" + str(areaID) + "/"
mkdir(area_path)

for id in range(len(cars_instance)):
    np.savetxt(area_path + str(id) + ".txt", cars_instance[id])
