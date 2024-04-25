from mash_occ_decoder.Module.sdf_convertor import SDFConvertor


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    noise_label = "0_025"
    train_scale = 0.98
    val_scale = 0.01

    sdf_convertor = SDFConvertor(dataset_root_folder_path, noise_label)
    sdf_convertor.convertToSplitFiles(train_scale, val_scale)

    return True
