import sys

sys.path.append("../ma-sh/")
sys.path.append("../distribution-manage/")

import os
import torch
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash
from ma_sh.Config.custom_path import toDatasetRootPath

from mash_occ_decoder.Dataset.sdf import SDFDataset
from mash_occ_decoder.Method.time import getCurrentTime
from mash_occ_decoder.Module.detector import Detector


def demo():
    model_file_path = "../../output/20241227_16:33:30/model_best.pth".replace('../.', '')
    noise_label = "0_25"
    transformer_id = 'Objaverse_82K'
    device = "cpu"
    save_folder_path = './output/recon/' + getCurrentTime() + '/'

    os.makedirs(save_folder_path, exist_ok=True)

    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    sdf_dataset = SDFDataset(
        dataset_root_folder_path,
        "train",
        noise_label=noise_label,
        train_percent=0.9,
    )

    detector = Detector(model_file_path, transformer_id, device)

    test_idxs = [
        1, 2, 3, 4, 5, 6, 7, 8, 9,
    ]

    for i in test_idxs:
        mash_params_file_path, sdf_file_path = sdf_dataset.paths_list[i]
        gt_mesh_file_path = sdf_file_path.replace(
            "manifold_sdf_" + noise_label,
            "manifold"
        ).replace(".npy", ".obj")

        print("start export mesh", i, "...")
        mesh = detector.detectFile(mash_params_file_path)
        print(mesh)
        mesh.export(save_folder_path + "recon_" + str(i) + "_mesh.obj")

        mash = Mash.fromParamsFile(mash_params_file_path, device=device)
        mash_points = torch.cat(mash.toSamplePoints()[:2], dim=0).detach().clone().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mash_points)
        o3d.io.write_point_cloud(
            save_folder_path + "mash_" + str(i) + "_pcd.ply", pcd, write_ascii=True
        )

        copyfile(gt_mesh_file_path, save_folder_path + "gt_" + str(i) + "_mesh.obj")
    return True
