import sys

sys.path.append("../ma-sh/")
sys.path.append("../distribution-manage/")

import os

from ma_sh.Model.mash import Mash
from ma_sh.Config.custom_path import toDatasetRootPath

from mash_occ_decoder.Module.detector import Detector


def demo_file():
    model_file_path = "../../output/512dim-v4/model_best.pth".replace('../.', '')
    use_ema = True
    batch_size = 1200000
    resolution = 128
    transformer_id = 'Objaverse_82K'
    device = "cuda:0"
    mash_file_path_list = [
        '/home/chli/chLi/Dataset/Objaverse_82K/manifold_mash/000-091/897ce33a65d04bb69eb3d87d0742464f.npy',
        # '/home/chli/chLi/Dataset/Objaverse_82K/manifold_mash/000-091/aba44d3812ae4377a5347b5a482f51ab.npy',
        '/home/chli/chLi/Dataset/Objaverse_82K/manifold_mash/000-091/e7c39d9f92b94bd8a1d44986f5c37549.npy',
    ]
    save_mesh_folder_path = './output/recon_CFM/000-091/'

    os.makedirs(save_mesh_folder_path, exist_ok=True)

    detector = Detector(model_file_path, use_ema, batch_size, resolution, transformer_id, device)

    for mash_file_path in mash_file_path_list:
        print("start export mesh for mash " + mash_file_path + "...")
        mesh = detector.detectFile(mash_file_path)
        print(mesh)

        save_mesh_file_path = save_mesh_folder_path + mash_file_path.split('/')[-1].replace('.npy', '_mesh.obj')

        mesh.export(save_mesh_file_path)

        save_mash_pcd_file_path = save_mesh_file_path.replace(
            'mesh', 'mash_pcd').replace('.obj', '.ply')
        Mash.fromParamsFile(
            mash_file_path,
            device=device,
        ).saveAsPcdFile(
            save_mash_pcd_file_path,
            overwrite=True,
        )
    return True

def demo_folder():
    model_file_path = "../../output/512dim-v4/model_best.pth".replace('../.', '')
    use_ema = True
    batch_size = 1200000
    resolution = 128
    transformer_id = 'Objaverse_82K'
    device = "cuda:0"
    time_stamp = '20250108_17:45:41'
    mash_folder_path = '../../../mash-diffusion/output/sample/'.replace('../.', '') + time_stamp + '/'
    save_folder_path = './output/recon_CFM/' + time_stamp + '/'

    os.makedirs(save_folder_path, exist_ok=True)

    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    detector = Detector(model_file_path, use_ema, batch_size, resolution, transformer_id, device)

    for root, _, files in os.walk(mash_folder_path):
        for file in files:
            if not file.endswith('.npy'):
                continue

            rel_folder_path = os.path.relpath(root, mash_folder_path)

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
