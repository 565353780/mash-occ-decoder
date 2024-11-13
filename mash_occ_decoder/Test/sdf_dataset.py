import sys
sys.path.append('../ma-sh/')
sys.path.append('../sdf-generate/')

import os
import numpy as np
import open3d as o3d

from ma_sh.Model.mash import Mash
from sdf_generate.Method.render import toNearSurfaceSDFPcd

from mash_occ_decoder.Dataset.sdf import SDFDataset

def test():
    dataset_root_folder_path = "/home/chli/chLi/Dataset/"
    noise_label_list = ["0_0025", "0_025", "0_25"]
    noise_label_list = ["0_0025"]

    sdf_dataset = SDFDataset(dataset_root_folder_path, noise_label_list=noise_label_list)

    for paths in sdf_dataset.paths_list:
        mash_file_path, sdf_file_path = paths

        gt_mesh_file_path = mash_file_path.replace('MashV4', 'ManifoldMesh').replace('.npy', '.obj')
        if not os.path.exists(gt_mesh_file_path):
            continue

        print('mash:', mash_file_path)
        print('sdf:', sdf_file_path)
        print('mesh:', gt_mesh_file_path)

        sdf_data = np.load(sdf_file_path)
        sdf_pcd = toNearSurfaceSDFPcd(sdf_data)

        sdf_points = np.asarray(sdf_pcd.points)
        sdf_colors = np.asarray(sdf_pcd.colors)
        filter_point_mask = sdf_colors[:, 2] == 1.0

        filter_pcd = o3d.geometry.PointCloud()
        filter_pcd.points = o3d.utility.Vector3dVector(sdf_points[filter_point_mask])
        filter_pcd.colors = o3d.utility.Vector3dVector(sdf_colors[filter_point_mask])

        mash = Mash.fromParamsFile(mash_file_path)
        mash_pcd = mash.toSamplePcd()

        o3d.visualization.draw_geometries([mash_pcd, filter_pcd])
        break

    return True
