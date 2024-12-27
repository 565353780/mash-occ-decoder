import sys
sys.path.append('../ma-sh/')
sys.path.append('../distribution-manage/')

import os
import torch

from mash_occ_decoder.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = os.environ['HOME'] + "/chLi/Dataset/"
    batch_size = 8
    accum_iter = 32
    num_workers = 16
    n_qry = 40000
    noise_label_list = ["0_25"]
    model_file_path = None
    # model_file_path = "./output/mamba1-v1-1/model_last.pth"
    dtype = torch.float32
    device = "auto"
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 2e-4
    drop_prob = 0.0
    kl_weight = 1.0
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

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
