import numpy as np
from tqdm import tqdm
import argparse

from utils import *

import open3d as o3d
import math
import os


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


def main():

    print('==> Loading args..')
    args = get_arguments()
    print(args)

    load_prefix = "../gt_instance/buildings_area49/"
    save_prefix = "./data/urbanbis/buildings_area49/"

    # large: 4096 | ordinary: 1024 ｜ small: 128 | 
    large = 4096
    ordinary = 1024
    small = 128
    
    
    large_points = []
    large_points_id = []

    ord_points = []
    ord_points_id = []

    small_points = []
    small_points_id = []

    for file in os.listdir(load_prefix):
        if file[:6] == "ignore":
            continue

        id = int(file[:-4])
        print(id)
        inst_points = np.loadtxt(load_prefix + file)[:, :3]
        l = len(inst_points)
        if l >= large:
            large_points.append(inst_points[furthest_point_sample(inst_points, large)])
            large_points_id.append(id)
        elif l >= ordinary:
            ord_points.append(inst_points[furthest_point_sample(inst_points, ordinary)])
            ord_points_id.append(id)
        elif l >= small:
            small_points.append(inst_points[furthest_point_sample(inst_points, small)])
            small_points_id.append(id)
    
    large_points = np.array(large_points)
    large_points_id = np.array(large_points_id)

    ord_points = np.array(ord_points)
    ord_points_id = np.array(ord_points_id)

    small_points = np.array(small_points)
    small_points_id = np.array(small_points_id)

    mkdir(save_prefix)

    np.save(save_prefix + "large_points", large_points)
    np.save(save_prefix + "large_points_id", large_points_id)

    np.save(save_prefix + "ord_points", ord_points)
    np.save(save_prefix + "ord_points_id", ord_points_id)

    np.save(save_prefix + "small_points", small_points)
    np.save(save_prefix + "small_points_id", small_points_id)

main()