import sys

sys.path.append("../ma-sh/")
sys.path.append("../distribution-manage/")

import os

from ma_sh.Model.mash import Mash
from ma_sh.Config.custom_path import toDatasetRootPath

from mash_occ_decoder.Module.detector import Detector


def demo():
    model_file_path = "../../output/20250101_01:12:59/model_best.pth".replace('../.', '')
    transformer_id = 'Objaverse_82K'
    device = "cpu"
    time_stamp = '20250104_17:29:23'
    pcd_folder_path = '../../../mash-diffusion/output/sample/'.replace('../.', '') + time_stamp + '/'
    save_folder_path = './output/recon_CFM/' + time_stamp + '/'

    os.makedirs(save_folder_path, exist_ok=True)

    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    detector = Detector(model_file_path, transformer_id, device)

    for root, _, files in os.walk(pcd_folder_path):
        for file in files:
            if not file.endswith('.npy'):
                continue

            rel_folder_path = os.path.relpath(root, pcd_folder_path)

            mash_file_path = root + '/' + file

            save_mesh_folder_path = save_folder_path + rel_folder_path + '/'
            os.makedirs(save_mesh_folder_path, exist_ok=True)

            save_mesh_file_path = save_mesh_folder_path + file.replace('mash', 'mesh').replace('.npy', '.obj')

            print("start export mesh for mash " + file + "...")
            mesh = detector.detectFile(mash_file_path)
            print(mesh)

            mesh.export(save_mesh_file_path)

            save_mash_pcd_file_path = save_mesh_folder_path + file.replace(
                'mash', 'mash_pcd').replace('.npy', '.ply')
            Mash.fromParamsFile(
                mash_file_path,
                device=device,
            ).saveAsPcdFile(
                save_mash_pcd_file_path,
                overwrite=True,
            )
    return True
