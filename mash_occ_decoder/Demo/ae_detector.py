import sys

sys.path.append("../ma-sh")

import os
import torch
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud

from mash_occ_decoder.Dataset.mash import MashDataset
from mash_occ_decoder.Module.ae_detector import Detector


def demo():
    output_folder_path = "./output/"
    model_folder_name_list = os.listdir(output_folder_path)
    model_folder_name_list.sort()

    valid_model_folder_name_list = []
    for model_folder_name in model_folder_name_list:
        if "2024" not in model_folder_name:
            continue
        if not os.path.isdir(output_folder_path + model_folder_name):
            continue
        valid_model_folder_name_list.append(model_folder_name)

    model_file_path = "./output/" + valid_model_folder_name_list[-1] + "/model_best.pth"
    print(model_file_path)
    dtype = torch.float32
    device = "cuda:0"

    dataset_root_folder_path = "/home/chli/Dataset/"
    mash_dataset = MashDataset(dataset_root_folder_path, "val")

    detector = Detector(model_file_path, dtype, device)

    for i in range(1):
        mash_file_path = mash_dataset.paths_list[i]
        gt_mesh_file_path = mash_file_path.replace(
            mash_dataset.mash_folder_path + "ShapeNet/",
            "/home/chli/chLi/Dataset/NormalizedMesh/ShapeNet/",
        ).replace(".npy", ".obj")

        print("start export mash", i + 1, "...")
        mash = detector.detectFile(mash_file_path)

        if True:
            gt_mash = Mash.fromParamsFile(mash_file_path)

            gt_pcd = getPointCloud(toNumpy(torch.vstack(gt_mash.toSamplePoints()[:2])))
            pcd = getPointCloud(toNumpy(torch.vstack(mash.toSamplePoints()[:2])))

            gt_pcd.translate([0, -1, 0])

            o3d.io.write_point_cloud(
                "./output/ae_mash_gt" + str(i) + ".ply", gt_pcd, write_ascii=True
            )
            o3d.io.write_point_cloud(
                "./output/ae_mash" + str(i) + ".ply", pcd, write_ascii=True
            )

        if True:
            copyfile(gt_mesh_file_path, "./output/ae_mash_mesh_gt" + str(i) + ".obj")
    return True
