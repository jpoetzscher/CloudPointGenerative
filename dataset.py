import open3d as o3d
import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}

cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

class ShapeNetCore(Dataset):    
    def __init__(self, path, cat_name, split, downsample=None, transform=None):
        super().__init__()
        self.path = path
        self.cat_name = cat_name
        self.cat = cate_to_synsetid[cat_name]

        self.split = split
        self.transform = transform

        self.pointclouds = []
        self.stats = None
        self.downsample = downsample

        self.get_statistics()

        self.load()

    def get_statistics(self):

        basename = self.path
        #print("BASENAME: ", basename, self.cat_name)
        stats_dir = os.path.join(os.path.dirname(self.path), 'stats')
        #print("STATS DIR: ", stats_dir)
        os.makedirs(stats_dir, exist_ok=True)

        stats_save_path = os.path.join(stats_dir, self.cat_name + '.pt')
        #print(stats_save_path)
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        #with h5py.File(self.path, 'r') as f:
        #print("F: ", f)
        pc_list = []
        for split in ('train', 'val', 'test'):
            full_path = self.path + self.cat + '/' + split
            points = os.listdir(full_path)
            for point in points:
                
                pc = full_path + '/' + point
                print(pc)
                pc_list.append(np.load(pc))
        #print(pointclouds.shape)
        all_points = torch.from_numpy(np.array(pc_list))
        #print(all_points.shape) # (B, N, 3)
        #print(all_points)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        #print(self.stats)
        return self.stats

    def load(self):
        
        full_path = self.path + self.cat + '/' + self.split
        points = os.listdir(full_path)
        for id, point in enumerate(points): 
            point_cloud_path = full_path + '/' + point
            #print(pc)
            pc = torch.from_numpy(np.load(point_cloud_path))
            if self.downsample != None and pc.shape[0] > self.downsample:
            # Randomly select indices
                indices = np.random.choice(pc.shape[0], self.downsample, replace=False)
                # Sample points using the selected indices
                pc = pc[indices, :] 
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
            pc = (pc - shift) / scale

            """ Randomly sample a fixed number of points from a point cloud """
            
            self.pointclouds.append({
                'pointcloud': pc,
                'id': id,
                'shift': shift,
                'scal': scale
            })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

train_pc = ShapeNetCore('ShapeNetCore.v2.PC15k 2/', 'cap', 'train', 1024)
"""
print(len(train_pc))
pc_example = train_pc[0]['pointcloud']



def point_cloud_from_points(points):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def visualize_point_cloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])



# Load and visualize the point cloud
#print(viz_pc.shape)
viz_pc = point_cloud_from_points(pc_example)
visualize_point_cloud(viz_pc)
"""