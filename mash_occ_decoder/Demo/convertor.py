import sys

sys.path.append("../ma-sh")

from mash_occ_decoder.Module.convertor import Convertor


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/aro_net/data/shapenet/mash/02691156/"
    save_split_folder_path = (
        "/home/chli/Dataset/aro_net/data/shapenet/04_splits/02691156/mash/"
    )
    query_point_root_folder_path = (
        "/home/chli/Dataset/aro_net/data/shapenet/02_qry_pts_occnet/02691156/"
    )
    save_feature_folder_path = (
        "/home/chli/Dataset/aro_net/data/shapenet/anchor_feature/02691156/"
    )
    train_scale = 0.8
    val_scale = 0.1

    convertor = Convertor(dataset_root_folder_path)
    convertor.convertToSplitFiles(save_split_folder_path, train_scale, val_scale)

    if False:
        convertor.convertToAnchorFeatures(
            query_point_root_folder_path, save_feature_folder_path
        )
    return True
