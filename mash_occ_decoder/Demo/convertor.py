import sys

sys.path.append("../ma-sh")

from mash_occ_decoder.Module.convertor import Convertor


def demo():
    class_id_list = [
        "02691156",
        "03001627",
    ]

    for class_id in class_id_list:
        dataset_root_folder_path = (
            "/home/chli/Dataset/aro_net/data/shapenet/mash/" + class_id + "/"
        )
        save_split_folder_path = (
            "/home/chli/Dataset/aro_net/data/shapenet/04_splits/" + class_id + "/mash/"
        )
        train_scale = 0.8
        val_scale = 0.1

        convertor = Convertor(dataset_root_folder_path)
        convertor.convertToSplitFiles(save_split_folder_path, train_scale, val_scale)

    return True
