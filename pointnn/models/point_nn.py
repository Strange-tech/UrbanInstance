# Non-Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn
# from pointnet2_ops import pointnet2_utils

from .model_utils import *



# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = furthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape    
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().to(xyz.device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().to(knn_xyz.device)    
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w


# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            # print("group num:", group_num)
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # print("stage:", i)
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)
        return x


# Non-Parametric Network
class Point_NN(nn.Module):
    def __init__(self, input_points, num_stages=4, embed_dim=72, k_neighbors=90, beta=1000, alpha=100):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta)
        

    def forward(self, x):
        
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        x = self.EncNP(xyz, x)
        return x


class FeatureExtrator(nn.Module):
    def __init__(self, input_points=1024, num_stages=1, embed_dim=72, k_neighbors=90, beta=1000, alpha=100):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))

    def forward(self, x):
        xyz = x.permute(0,2,1)

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # print(xyz.shape, lc_x.shape, knn_xyz.shape, knn_x.shape)
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # print(knn_x_w.shape)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)
            # print(x.shape)

        return xyz, x.permute(0,2,1)


class LinearNet(nn.Module):
    def __init__(self, n_feature=4):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 4)

    def forward(self, xyz):
        # xyz: [B, D], D = 3
        homo_add = torch.ones((xyz.shape[0], 1)).cuda()
        # homo_xyz: [B, D'], D' = 4
        homo_xyz = torch.cat((xyz, homo_add), dim=-1)
        homo_trans_xyz = self.linear(homo_xyz)
        # divide by the last element
        last_elem = homo_trans_xyz[:,-1].unsqueeze(-1)
        homo_trans_xyz = homo_trans_xyz / last_elem
    
        return homo_trans_xyz[:,:-1]

class FeatureExtractorByCentroids(nn.Module):
    def __init__(self, input_points=1024, num_stages=1, embed_dim=72, k_neighbors=90, beta=1000, alpha=100):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        self.k_neighbors = k_neighbors
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))
    
    def forward(self, centroids, x):
        xyz = x.permute(0,2,1)

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # kNN
            knn_idx = knn_point(self.k_neighbors, xyz, centroids)
            knn_xyz = index_points(xyz, knn_idx)
            knn_x = index_points(x.permute(0,2,1), knn_idx)
            # print(xyz.shape, centroids.shape, knn_xyz.shape, knn_x.shape)

            # Local Geometry Aggregation
            lc_x = torch.normal(mean=0, std=1, size=(centroids.shape[0], centroids.shape[1], knn_x.shape[-1])).cuda()
            knn_x_w = self.LGA_list[i](centroids, lc_x, knn_xyz, knn_x)
            # print(knn_x_w.shape)

            # Pooling
            x = self.Pooling_list[i](knn_x_w)
            # print(x.shape)

        return x.permute(0,2,1)