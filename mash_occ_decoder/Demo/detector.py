import sys

sys.path.append("../ma-sh")

import torch
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash

from mash_occ_decoder.Dataset.sdf import SDFDataset
from mash_occ_decoder.Module.detector import Detector


def demo():
    model_file_path = "./output/full-v2_3/model_last.pth"
    dtype = torch.float32
    device = "cuda:0"

    dataset_root_folder_path = "/home/chli/Dataset/"
    sdf_dataset = SDFDataset(dataset_root_folder_path, "test")

    detector = Detector(model_file_path, dtype, device)

    for i in range(1):
        mash_params_file_path, sdf_file_path = sdf_dataset.paths_list[i]
        mesh_file_path = sdf_file_path
        print(mesh_file_path)
        # exit()
        mesh = detector.detectFile(mash_params_file_path)

        print(mesh)

        mesh.export("./output/test_mash_mesh" + str(i) + ".obj")

        mash = Mash.fromParamsFile(mash_params_file_path, device=device)
        mash_points = mash.toSamplePoints().detach().clone().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mash_points)
        o3d.io.write_point_cloud("./output/test_mash_pcd" + str(i) + ".ply", pcd)

    # gt_mesh_file_path = mesh_folder_path + model_id + ".obj"
    # copyfile(gt_mesh_file_path, "./output/test_mash_mesh_gt.obj")
    return True
