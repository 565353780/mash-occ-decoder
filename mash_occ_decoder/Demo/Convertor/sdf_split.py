from mash_occ_decoder.Module.Convertor.sdf_split import Convertor


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    noise_label = "0_025"
    train_scale = 0.98
    val_scale = 0.01

    convertor = Convertor(dataset_root_folder_path, noise_label)
    convertor.convertToSplitFiles(train_scale, val_scale)

    return True
