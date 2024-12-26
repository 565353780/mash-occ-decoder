import os
import torch

from mash_occ_decoder.Module.Convertor.sdf_split import Convertor
from mash_occ_decoder.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = os.environ['HOME'] + "/chLi/Dataset/"
    batch_size = 400
    accum_iter = 1
    num_workers = 4
    n_qry = 200
    noise_label_list = ["0_25"]
    model_file_path = None
    # model_file_path = "./output/mamba1-v1-1/model_last.pth"
    dtype = torch.float32
    device = "auto"
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 1e-2
    drop_prob = 0.75
    kl_weight = 1.0
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
        warm_step_num,
        finetune_step_num,
        lr,
        drop_prob,
        kl_weight,
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.train()
    return True
