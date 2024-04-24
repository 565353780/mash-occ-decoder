import torch

from mash_occ_decoder.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    batch_size = 20
    num_workers = 4
    n_qry = 4000
    model_file_path = "./output/mamba-v1/model_last.pth"
    dtype = torch.float32
    device = "cuda:0"
    warm_epoch_step_num = 100
    warm_epoch_num = 0
    finetune_step_num = 100000000
    lr = 1e-4
    weight_decay = 1e-10
    factor = 0.99
    patience = 10000
    min_lr = 1e-6
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    trainer = Trainer(
        dataset_root_folder_path,
        batch_size,
        num_workers,
        n_qry,
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
