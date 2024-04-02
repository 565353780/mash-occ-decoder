import torch

from mash_occ_decoder.Module.trainer import Trainer


def demo():
    model_file_path = None
    dtype = torch.float64
    device = "cuda:0"
    warm_epoch_step_num = 400
    warm_epoch_num = 10
    finetune_step_num = 100000
    lr = 1e-5
    weight_decay = 1e-4
    factor = 0.9
    patience = 400
    min_lr = 1e-8
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    trainer = Trainer(
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
