
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

    load_prefix = "../gt_instance/buildings_area47/"
    save_prefix = "./data/urbanbis/buildings_area47/"

    # large: 4096 | ordinary: 1024 ｜ small: 128 | 
    large = 4096
    ordinary = 1024
    small = 128
    
    large_points = np.load(save_prefix + "large_points.npy")


    ord_points = np.load(save_prefix + "ord_points.npy")
    ord_points_id = np.load(save_prefix + "ord_points_id.npy")

    small_points = np.load(save_prefix + "small_points.npy") 
    small_points_id = np.load(save_prefix + "small_points_id.npy")

    print(large_points.shape, ord_points.shape, small_points.shape)
    # print(large_points_id.shape, ord_points_id.shape, small_points_id.shape)

    ## overall features
    large_point_nn = Point_NN(input_points=large, num_stages=args.stages,
                    embed_dim=args.dim, k_neighbors=70,
                    alpha=args.alpha, beta=args.beta).cpu()
    large_point_nn.eval()
    large_points_features = get_feature_by_net(large_points, large_point_nn)

    ord_point_nn = Point_NN(input_points=ordinary, num_stages=args.stages,
                    embed_dim=args.dim, k_neighbors=30,
                    alpha=args.alpha, beta=args.beta).cpu()
    ord_point_nn.eval()
    ord_points_features = get_feature_by_net(ord_points, ord_point_nn)

    small_point_nn = Point_NN(input_points=small, num_stages=args.stages,
                    embed_dim=args.dim, k_neighbors=8,
                    alpha=args.alpha, beta=args.beta).cpu()
    small_point_nn.eval()
    small_points_features = get_feature_by_net(small_points, small_point_nn)

    large_points_features = large_points_features.cpu().numpy()
    ord_points_features = ord_points_features.cpu().numpy()
    small_points_features = small_points_features.cpu().numpy()

    np.save(save_prefix + "large_points_features", large_points_features)
    np.save(save_prefix + "ord_points_features", ord_points_features)
    np.save(save_prefix + "small_points_features", small_points_features)

main()