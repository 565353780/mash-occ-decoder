import torch

from mash_occ_decoder.Module.Convertor.mash_split import Convertor
from mash_occ_decoder.Module.ae_trainer import Trainer


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    batch_size = 128
    num_workers = 4
    model_file_path = "./output/t-v3-2/model_last.pth"
    model_file_path = None
    dtype = torch.float32
    device = "cuda:0"
    lr = 1e-5
    weight_decay = 1e-10
    factor = 0.99
    patience = 10000
    min_lr = 1e-6
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    train_scale = 0.9
    val_scale = 0.1

    convertor = Convertor(dataset_root_folder_path)
    convertor.convertToSplitFiles(train_scale, val_scale)

    trainer = Trainer(
        dataset_root_folder_path,
        batch_size,
        num_workers,
        model_file_path,
        dtype,
        device,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.autoTrain()
    return True
