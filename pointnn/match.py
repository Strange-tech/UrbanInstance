
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

from tqdm import tqdm
import argparse

from utils import *
from models import Point_NN, FeatureExtrator, LinearNet, FeatureExtractorByCentroids

import open3d as o3d
import math
import os
from sklearn.cluster import DBSCAN


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("mkdir success")
    else:
        print("dir exists")


def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    N, C = xyz.shape
    # print(xyz.shape)
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    # batch_indices = np.arange(B)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        # print(dist.shape)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    # print(centroids)
    return centroids



def get_arguments():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='mn40')
    parser.add_argument('--dataset', type=str, default='scan')

    # parser.add_argument('--split', type=int, default=1)
    # parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--split', type=int, default=3)

    parser.add_argument('--bz', type=int, default=16)  # Freeze as 16

    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--stages', type=int, default=3)
    parser.add_argument('--dim', type=int, default=72)
    parser.add_argument('--k', type=int, default=90)
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--beta', type=int, default=100)

    args = parser.parse_args()
    return args
    

def get_feature_by_net(points, net):
    # points[:, 2].fill(0)
    points = torch.tensor(points, dtype=torch.float32)
    points = points.cpu().permute(0, 2, 1)

    point_features = net(points)
    return point_features

@torch.no_grad()
def main():

    print('==> Loading args..')
    args = get_arguments()
    print(args)

    areaID = [41, 46, 47, 48, 49]

    load_prefix = "../gt_instance/buildings_area"
    save_prefix = "./data/urbanbis/buildings_area"

    all_large_features = []
    all_large_id = []

    for ID in areaID:
        large_points_features = np.load(save_prefix + str(ID) + "/large_points_features.npy")
        large_points_id = np.load(save_prefix + str(ID) + "/large_points_id.npy")
        print(large_points_features.shape)
        if len(all_large_features) == 0:
            all_large_features = np.copy(large_points_features)
        else:
            all_large_features = np.vstack((all_large_features, large_points_features))

        if len(all_large_id) == 0:
            all_large_id = np.copy(large_points_id)
        else:
            all_large_id = np.hstack((all_large_id, large_points_id))

    print(all_large_features.shape)
    print(all_large_id.shape)

    all_large_features /= np.linalg.norm(all_large_features, axis=-1, keepdims=True)

    score = np.matmul(all_large_features, all_large_features.T)

    print(score.shape)

    N = len(all_large_id)
    large_idx = range(N)

    diff_threshold = 0.5

    sorted_idx_table = np.argsort(-score, axis=-1)

    print(sorted_idx_table)

    union_flag = True
    while(union_flag):

        print("--------------------------------------while--------------------------------------")

        union_flag = False
        unionfind = UnionFind(N)

        for idx in large_idx:
            points_id = all_large_id[idx]
            print(idx, points_id)

            # choose the most similar one
            for similar_idx in sorted_idx_table[idx][1:]:
                if similar_idx in large_idx:
                    most_similar_idx = similar_idx
                    break
            
            # reverse verification
            if np.argwhere(sorted_idx_table[most_similar_idx] == idx) - 1 > N * diff_threshold:
                continue
            else:
                union_flag = True
                unionfind.Union(idx, most_similar_idx)
        
        ufset = unionfind.Print()

        for k, v in ufset.items():
            print("father:", all_large_id[k], "children:", all_large_id[v])

        # update large_idx
        updated_large_idx = []
        for idx, father in enumerate(unionfind.fa):
            if idx in large_idx and idx == father:
                updated_large_idx.append(idx)

        large_idx = updated_large_idx

        # np.save(save_prefix + "match.npy", unionfind.fa)

if __name__ == '__main__':
    main()
