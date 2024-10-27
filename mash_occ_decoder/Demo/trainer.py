import torch

from mash_occ_decoder.Module.Convertor.sdf_split import Convertor
from mash_occ_decoder.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    batch_size = 16
    accum_iter = 1
    num_workers = 4
    n_qry = 8000
    noise_label_list = ["0_25", "0_025", "0_0025"]
    model_file_path = "./output/mamba2-v1-1/model_last.pth"
    model_file_path = None
    dtype = torch.float32
    device = "cuda:0"
    warm_epoch_step_num = 100
    warm_epoch_num = 0
    finetune_step_num = 100000000
    lr = 1e-5
    weight_decay = 1e-10
    factor = 0.99
    patience = 10000
    min_lr = 1e-7
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    train_scale = 0.9
    val_scale = 0.1

    for noise_label in noise_label_list:
        convertor = Convertor(dataset_root_folder_path, noise_label)
        convertor.convertToSplitFiles(train_scale, val_scale)

    trainer = Trainer(
        dataset_root_folder_path,
        batch_size,
        accum_iter,
        num_workers,
        n_qry,
        noise_label_list,
        model_file_path,
        dtype,
        device,
        warm_epoch_step_num,
        warm_epoch_num,
        finetune_step_num,
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
