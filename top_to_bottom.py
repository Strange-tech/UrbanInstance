
import open3d as o3d
import numpy as np
import math
from sklearn.cluster import DBSCAN, KMeans
from utils import OBB

ROI_THRESHOLD = 10.0
SET_NEW = 1e8
REACH_BASE = False

areaID = 38
area_path = "./gt_instance/buildings_area" + str(areaID) + "/"
buildingID = 610

slice_num = 10
dbscan_min_dist = 3.0

    
def compare_ROIs(roi1, roi2):
    # roi1 = np.array(roi1)
    # roi2 = np.array(roi2)

    # is_contain = False
    # if (roi1[0] - roi2[0]).all() <= 0 and (roi1[1] - roi2[1]).all() >= 0:
    #     is_contain = True
    # d0 = np.sum(np.square(roi1[0] - roi2[0]))
    # d1 = np.sum(np.square(roi1[1] - roi2[1]))

    # return is_contain, d0 + d1
    center_diff = np.sqrt(np.sum(np.square(roi1["center"] - roi2["center"])))
    return center_diff


def inside_ROI(points, roi):
    # res = []
    # for p in points:
    #     if p[0] > roi[0][0] and p[1] > roi[0][1] and p[0] <= roi[1][0] and p[1] <= roi[1][1]:
    #         res.append(p)

    # return res
    res_idx = []

    pointArray = np.array(points)

    vect = roi["eigenvector"]
    tvect = np.transpose(vect)
    points_2d = pointArray[:, :2]
    trans_points_2d = np.dot(points_2d, np.linalg.inv(tvect))

    mina = roi["mina"]
    maxa = roi['maxa']

    for i, p in enumerate(trans_points_2d):
        if p[0] >= mina[0] and p[1] >= mina[1] and p[0] <= maxa[0] and p[1] <= maxa[1]:
            res_idx.append(i)

    return pointArray[res_idx]

def get_ROIs(instances, use_obb=True):
    rois = []
    if use_obb:
        for i in range(len(instances)):
            inst = np.array(instances[i])
            center, radius, mina, maxa, eigenvector = OBB(inst[:, :2])
            # inst = np.array(instances[i])

            # max_x = np.max(inst[:, 0])
            # min_x = np.min(inst[:, 0])
            # max_y = np.max(inst[:, 1])
            # min_y = np.min(inst[:, 1])

            # roi = [[min_x, min_y], [max_x, max_y]]

            rois.append({"center":center, "radius":radius, "mina":mina, "maxa":maxa, "eigenvector":eigenvector})
    
    return rois

building_pc = o3d.io.read_point_cloud(area_path + str(buildingID) + ".txt", format="xyz")

points = np.asarray(building_pc.points)
sorted_idx = points[:, 2].argsort()
sorted_points = points[sorted_idx][::-1]

# print(sorted_points[0], sorted_points[-1])

l = len(sorted_points)
step = math.ceil(l / slice_num)

idx = 0
last_time_idx = 0

building_instances = []
building_instances_roi = []


while(idx < l):
    print("................WHILE................")
    idx += step
    if idx >= l:
        idx = l
    block = sorted_points[last_time_idx:idx]
    clustering = DBSCAN(eps=dbscan_min_dist, min_samples=5).fit(block)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print(n_clusters_)

    block_instances = [None] * n_clusters_
    # block_instances_roi = [None] * n_clusters_

    for i in range(len(labels)):
        if block_instances[labels[i]] == None:
            block_instances[labels[i]] = []
        block_instances[labels[i]].append(block[i])
    
    print("len block instances:", len(block_instances))

    block_instances_roi = get_ROIs(block_instances)

    block2building_idx_map = {}
    map_flag = False

    print(len(block_instances_roi), len(building_instances_roi))

    # if len(building_instances_roi) == 0:
        # building_instances_roi = block_instances_roi.copy()
    if len(building_instances_roi) > 0:
        if len(block_instances_roi) < len(building_instances_roi):
            REACH_BASE = True
        elif len(block_instances_roi) > len(building_instances_roi):
            for roi1_idx, roi1 in enumerate(block_instances_roi):
                print("roi1_idx:", roi1_idx)
                map_flag = False
                
                for roi2_idx, roi2 in enumerate(building_instances_roi):
                    diff = compare_ROIs(roi1, roi2)
                    print("center diff:", diff)
                    if diff < ROI_THRESHOLD:
                        block2building_idx_map[roi1_idx] = roi2_idx
                        map_flag = True
                        break
                
                if map_flag == False:
                    block2building_idx_map[roi1_idx] = SET_NEW
        else:
            for roi1_idx, roi1 in enumerate(block_instances_roi):
                for roi2_idx, roi2 in enumerate(building_instances_roi):
                    diff = compare_ROIs(roi1, roi2)
                    print("center diff:", diff)
                    if diff < ROI_THRESHOLD:
                        block2building_idx_map[roi1_idx] = roi2_idx
                        break
        
        # if REACH_BASE == False:
        #     building_instances_roi = block_instances_roi.copy()

    print(block2building_idx_map)

    print(building_instances_roi)
    print("building inst sample:")
    for inst in building_instances:
        print(inst[0])

    print(' ')

    print(block_instances_roi)
    print("block inst sample:")
    for id, inst in enumerate(block_instances):
        print(inst[0])           

    
    if len(building_instances) == 0:
        building_instances = block_instances.copy()
    else:
        if REACH_BASE:
            for block_inst in block_instances:
                for roi_idx, roi in enumerate(building_instances_roi):
                    inside_points = inside_ROI(block_inst, roi)
                    if len(inside_points) > 0:
                        building_instances[roi_idx] = np.vstack((building_instances[roi_idx], inside_points))
        else:
            for block_idx, block_inst in enumerate(block_instances):
                print(block_idx)
                building_idx = block2building_idx_map[block_idx]
                print("block, building:", block_idx, building_idx)
                if building_idx == SET_NEW:
                    building_instances.append(block_inst)
                else:
                    building_instances[building_idx] = np.vstack((building_instances[building_idx], block_inst))              

    building_instances_roi = get_ROIs(building_instances)
    print(len(building_instances))

    last_time_idx += step

for id, inst in enumerate(building_instances):
    np.savetxt(area_path + str(buildingID) + "_" + str(id) + ".txt", inst)