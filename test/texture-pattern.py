import open3d as o3d
import numpy as np

terrain = o3d.io.read_point_cloud("./testcase.ply")

# print(terrain)

points = np.asarray(terrain.points)
points[:, 2].fill(0)

# terrain.points = o3d.utility.Vector3dVector(points)

# o3d.visualization.draw_geometries([terrain])

