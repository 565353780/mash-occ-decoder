import sys

sys.path.append("../ma-sh")

import os
from mash_occ_decoder.Module.convertor import Convertor


def demo():
    dataset_root_folder_path = "/home/chli/chLi/Dataset/SDF/ShapeNet/"
    sdf_root_folder_path = dataset_root_folder_path + "sdf/"

    class_id_list = os.listdir(sdf_root_folder_path)

    for class_id in class_id_list:
        current_dataset_root_folder_path = sdf_root_folder_path + class_id + "/"
        current_save_split_folder_path = (
            dataset_root_folder_path + "split/" + class_id + "/"
        )
        train_scale = 0.98
        val_scale = 0.01

        convertor = Convertor(current_dataset_root_folder_path)
        convertor.convertToSplitFiles(
            current_save_split_folder_path, train_scale, val_scale
        )

    return True
