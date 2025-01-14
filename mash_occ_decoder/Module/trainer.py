import torch
from torch import nn
from tqdm import trange
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from ma_sh.Model.mash import Mash

from mash_occ_decoder.Dataset.sdf import SDFDataset
from mash_occ_decoder.Method.tomesh import extractMesh
from mash_occ_decoder.Metric.occ import cal_occ_positive_percent, cal_occ_acc
from mash_occ_decoder.Model.mash_decoder import MashDecoder


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 8,
        accum_iter: int = 32,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "auto",
        dtype = torch.float32,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = 'auto',
        save_log_folder_path: Union[str, None] = 'auto',
        best_model_metric_name: Union[str, None] = 'Accuracy',
        is_metric_lower_better: bool = False,
        sample_results_freq: int = -1,
        use_amp: bool = False,
        quick_test: bool = False,
        n_qry: int = 28000,
        noise_label_list: list = ["0_25"],
        drop_prob: float = 0.1,
        mash_noise_level: float = 0.1,
        kl_weight: float = 1.0,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.n_qry = n_qry
        self.noise_label_list = noise_label_list
        self.drop_prob = drop_prob
        self.mash_noise_level = mash_noise_level
        self.loss_kl_weight = kl_weight

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.anchor_num = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.gt_sample_added_to_logger = False

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
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
            quick_test,
        )
        return

    def createDatasets(self) -> bool:
        if True:
            for noise_label in self.noise_label_list:
                self.dataloader_dict['sdf_' + noise_label] =  {
                    'dataset': SDFDataset(self.dataset_root_folder_path, 'train', self.n_qry, noise_label),
                    'repeat_num': 1,
                }

        if True:
            self.dataloader_dict['eval'] =  {
                'dataset': SDFDataset(self.dataset_root_folder_path, 'val', self.n_qry, self.noise_label_list[0]),
            }

        return True

    def createModel(self) -> bool:
        self.model = MashDecoder().to(self.device)
        return True

    def preProcessData(self, data_dict: dict, is_training: bool = True) -> dict:
        if is_training:
            mash_params = data_dict['mash_params']
            data_dict['mash_params'] = mash_params + self.mash_noise_level * torch.randn_like(mash_params)
            data_dict['drop_prob'] = self.drop_prob
            data_dict['deterministic'] = self.loss_kl_weight == 0
        else:
            data_dict['drop_prob'] = 0.0
            data_dict['deterministic'] = True
        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        gt_occ = data_dict['occ']
        occ = result_dict['occ']
        loss_occ = self.loss_fn(occ, gt_occ)

        loss_kl = 0.0
        if 'kl' in result_dict.keys() and self.loss_kl_weight > 0.0:
            kl = result_dict['kl']
            loss_kl = torch.sum(kl) / kl.shape[0]

        weighted_loss_kl = self.loss_kl_weight * loss_kl

        loss = loss_occ + weighted_loss_kl

        acc = cal_occ_acc(occ, gt_occ)
        positive_occ_percent = cal_occ_positive_percent(gt_occ)

        loss_dict = {
            "LossOCC": loss_occ,
            "LossKL": loss_kl,
            "Loss": loss,
            "Accuracy": acc,
            "PositiveOCC": positive_occ_percent,
        }

        return loss_dict

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        if self.local_rank != 0:
            return True

        sample_num = 3
        resolution = 128
        batch_size = 1200000
        dataset = self.dataloader_dict['eval']["dataset"]

        model.eval()

        print("[INFO][Trainer::sampleModelStep]")
        print("\t start recon", sample_num, "mashs....")
        for i in trange(sample_num):
            data_dict = dataset.__getitem__(i)

            mash_params = data_dict['mash_params'].to(self.device)

            mesh = extractMesh(mash_params, model, resolution, batch_size, 'odc')

            self.logger.addMesh(model_name + "/recon_mesh_" + str(i), mesh, self.step)

            mash_model = Mash(
                self.anchor_num,
                self.mask_degree,
                self.sh_degree,
                20,
                800,
                0.4,
                dtype=torch.float64,
                device=self.device,
            )

            if not self.gt_sample_added_to_logger:
                gt_mash = dataset.normalizeInverse(mash_params)

                sh2d = 2 * self.mask_degree + 1
                ortho_poses = gt_mash[:, :6]
                positions = gt_mash[:, 6:9]
                mask_params = gt_mash[:, 9 : 9 + sh2d]
                sh_params = gt_mash[:, 9 + sh2d :]

                mash_model.loadParams(
                    mask_params=mask_params,
                    sh_params=sh_params,
                    positions=positions,
                    ortho6d_poses=ortho_poses,
                )

                pcd = mash_model.toSamplePcd()

                self.logger.addPointCloud("GT_MASH/gt_mash_" + str(i), pcd, self.step)

        self.gt_sample_added_to_logger = True

        return True
