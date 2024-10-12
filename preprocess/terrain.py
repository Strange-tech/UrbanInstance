from distutils.command import build
import open3d as o3d
import numpy as np

# area9 = np.loadtxt("../dataset/Area9.txt")
area13 = np.loadtxt("../data/Area13.txt")

# X Y Z R G B Semantic_label Instance_label Fine-grained_building_category

print(area13.shape)

# 'Terrain': 0

building_dic = {}

for i in range(len(area13)):
    el = area13[i]
    if el[-3] == 0:
        if str(el[-2]) in building_dic:
            building_dic[str(el[-2])].append(el)
        else:
            building_dic[str(el[-2])] = []

# print(building_dic)
for key in building_dic.keys():
    print(key, len(building_dic[key]))
    np.savetxt("../gt_instance/terrain_area13/" + key[:-2] + ".txt", building_dic[key])



