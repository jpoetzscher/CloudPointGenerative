import numpy as np
import open3d as o3d
import os, random 
def point_cloud_from_points(points):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def visualize_point_cloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# File path to your .npy file
cates = '02691156'
npy_file_path = 'ShapeNetCore.v2.PC15k 2/' + cates + '/train/' + random.choice(os.listdir('ShapeNetCore.v2.PC15k 2/' + cates + '/train/'))  # Replace with your file path
print(npy_file_path)
#point_cloud = load_point_cloud_from_npy(npy_file_path)

def downsample_point_cloud(point_cloud, num_points=1000):
    """ Randomly sample a fixed number of points from a point cloud """
    if point_cloud.shape[0] < num_points:
        raise ValueError("The point cloud has fewer points than the target sample size")

    # Randomly select indices
    indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)

    # Sample points using the selected indices
    sampled_point_cloud = point_cloud[indices, :]

    return sampled_point_cloud

# Assuming you have loaded your point cloud into a variable named `point_cloud`
# point_cloud = np.load('path_to_point_cloud.npy')

# Downsample the point cloud
sampled_point_cloud = downsample_point_cloud(np.load(npy_file_path), 2048)
pc = point_cloud_from_points(sampled_point_cloud)

# Load and visualize the point cloud
print(np.load(npy_file_path).shape)
print(sampled_point_cloud.shape)

visualize_point_cloud(pc)
