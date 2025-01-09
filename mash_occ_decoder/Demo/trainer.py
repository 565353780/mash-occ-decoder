import sys
sys.path.append('../ma-sh/')
sys.path.append('../distribution-manage/')
sys.path.append('../base-trainer/')

from ma_sh.Config.custom_path import toDatasetRootPath

from mash_occ_decoder.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    batch_size = 8
    accum_iter = 32
    num_workers = 16
    model_file_path = None
    model_file_path = "../../output/512dim-v4/model_last.pth".replace('../.', '')
    device = "auto"
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 1e-6
    lr_batch_size = 256
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = 'Accuracy'
    is_metric_lower_better = False
    sample_results_freq = -1
    use_amp = False
    n_qry = 28000
    noise_label_list = ["0_25"]
    drop_prob = 0.0

    trainer = Trainer(
        dataset_root_folder_path,
        batch_size,
        accum_iter,
        num_workers,
        model_file_path,
        device,
        warm_step_num,
        finetune_step_num,
        lr,
        lr_batch_size,
        ema_start_step,
        ema_decay_init,
        ema_decay,
        save_result_folder_path,
        save_log_folder_path,
        best_model_metric_name,
        is_metric_lower_better,
        sample_results_freq,
        use_amp,
        n_qry,
        noise_label_list,
        drop_prob,
    )

    trainer.train()
    return True
