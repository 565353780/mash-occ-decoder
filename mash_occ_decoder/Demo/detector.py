import sys
sys.path.append("../ma-sh/")
sys.path.append("../distribution-manage/")

import torch
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash
from ma_sh.Config.custom_path import toDatasetRootPath

from mash_occ_decoder.Dataset.sdf import SDFDataset
from mash_occ_decoder.Module.detector import Detector


def demo():
    model_file_path = "../../output/20241227_16:33:30/model_best.pth".replace('../.', '')
    noise_label = "0_25"
    transformer_id = 'Objaverse_82K'
    device = "cuda:0"

    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    sdf_dataset = SDFDataset(
        dataset_root_folder_path,
        "train",
        noise_label=noise_label,
        train_percent=0.9,
    )

    detector = Detector(model_file_path, transformer_id, device)

    for i in range(10):
        mash_params_file_path, sdf_file_path = sdf_dataset.paths_list[i]
        gt_mesh_file_path = sdf_file_path.replace(
            "manifold_sdf_" + noise_label,
            "manifold"
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
            copyfile(gt_mesh_file_path, "./output/test_mash_mesh_gt" + str(i) + ".obj")
    return True
