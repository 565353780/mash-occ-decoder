import sys

sys.path.append("../ma-sh")

import torch
import numpy as np
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash

from mash_occ_decoder.Dataset.sdf import SDFDataset
from mash_occ_decoder.Module.detector import Detector


def demo():
    model_file_path = "./output/20241022_00:34:53/model_best.pth"
    dtype = torch.float32
    device = "cuda:0"

    dataset_root_folder_path = "/home/chli/Dataset/"
    sdf_dataset = SDFDataset(dataset_root_folder_path, "val")

    detector = Detector(model_file_path, dtype, device)

    for i in range(10):
        mash_params_file_path, sdf_file_path = sdf_dataset.paths_list[i]
        gt_mesh_file_path = sdf_file_path.replace(
            sdf_dataset.sdf_folder_path_list[0] + "ShapeNet/",
            "/home/chli/chLi/Dataset/NormalizedMesh/ShapeNet/",
        ).replace(".npy", ".obj")

        if True:
            print("start export mesh", i + 1, "...")
            mesh = detector.detectFile(mash_params_file_path)
            print(mesh)
            mesh.export("./output/test_mash_mesh" + str(i) + ".obj")

        if True:
            mash = Mash.fromParamsFile(mash_params_file_path, device=device)
            mash_points = torch.cat(mash.toSamplePoints()[:2], dim=0).detach().clone().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mash_points)
            o3d.io.write_point_cloud(
                "./output/test_mash_pcd" + str(i) + ".ply", pcd, write_ascii=True
            )

        if True:
            test_mesh = o3d.io.read_triangle_mesh(gt_mesh_file_path)
            pts = np.asarray(test_mesh.vertices)
            print(np.min(pts, axis=0))
            print(np.max(pts, axis=0))

        if True:
            copyfile(gt_mesh_file_path, "./output/test_mash_mesh_gt" + str(i) + ".obj")
    return True
