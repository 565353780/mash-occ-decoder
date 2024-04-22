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
    model_file_path = "./output/20240422_17:43:08/model_best.pth"
    dtype = torch.float32
    device = "cuda:0"

    dataset_root_folder_path = "/home/chli/Dataset/"
    sdf_dataset = SDFDataset(dataset_root_folder_path, "val")

    detector = Detector(model_file_path, dtype, device)

    for i in range(1):
        mash_params_file_path, sdf_file_path = sdf_dataset.paths_list[i]
        gt_mesh_file_path = sdf_file_path.replace(
            sdf_dataset.sdf_folder_path + "ShapeNet/sdf/",
            "/home/chli/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/",
        ).replace("_obj.npy", ".obj")

        if True:
            mesh = detector.detectFile(mash_params_file_path)
            print(mesh)
            mesh.export("./output/test_mash_mesh" + str(i) + ".obj")

        if False:
            mash = Mash.fromParamsFile(mash_params_file_path, device=device)
            mash_points = mash.toSamplePoints().detach().clone().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mash_points)
            o3d.io.write_point_cloud("./output/test_mash_pcd" + str(i) + ".ply", pcd)

        if False:
            test_mesh = o3d.io.read_triangle_mesh(gt_mesh_file_path)
            pts = np.asarray(test_mesh.vertices)
            print(np.min(pts, axis=0))
            print(np.max(pts, axis=0))

        copyfile(gt_mesh_file_path, "./output/test_mash_mesh_gt" + str(i) + ".obj")
    return True
