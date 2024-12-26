import os
import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from typing import Union
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from mash_occ_decoder.Dataset.sdf import SDFDataset
from mash_occ_decoder.Method.time import getCurrentTime
from mash_occ_decoder.Method.path import createFileFolder, removeFile, renameFile
from mash_occ_decoder.Metric.occ import cal_occ_positive_percent, cal_occ_acc
from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Module.logger import Logger


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def check_and_replace_nan_in_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradient: {name}")
            param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
    return True


class Trainer(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 400,
        accum_iter: int = 1,
        num_workers: int = 4,
        n_qry: int = 200,
        noise_label_list: list = ["0_25"],
        model_file_path: Union[str, None] = None,
        dtype=torch.float32,
        device: str = "auto",
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 1e-2,
        drop_prob: float = 0.75,
        kl_weight: float = 1.0,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.local_rank = setup_distributed()

        self.accum_iter = accum_iter
        self.dtype = dtype
        if device == 'auto':
            self.device = torch.device('cuda:' + str(self.local_rank))
        else:
            self.device = device

        self.warm_step_num = warm_step_num / accum_iter
        self.finetune_step_num = finetune_step_num
        self.lr = lr * batch_size / 256 * self.accum_iter * dist.get_world_size()
        self.drop_prob = drop_prob
        self.loss_kl_weight = kl_weight

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.step = 0
        self.epoch = 0
        self.loss_dict_list = []
        self.loss_min = float("inf")

        self.logger = None
        if self.local_rank == 0:
            self.logger = Logger()

        self.dataloader_dict = {}

        if True:
            for noise_label in noise_label_list:
                self.dataloader_dict['sdf_' + noise_label] =  {
                    'dataset': SDFDataset(dataset_root_folder_path, 'train', n_qry, noise_label),
                    'repeat_num': 1,
                }

        if True:
            self.dataloader_dict['eval'] =  {
                'dataset': SDFDataset(dataset_root_folder_path, 'val', n_qry, noise_label_list[0]),
            }

        for key, item in self.dataloader_dict.items():
            if key == 'eval':
                self.dataloader_dict[key]['dataloader'] = DataLoader(
                    item['dataset'],
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                continue

            self.dataloader_dict[key]['sampler'] = DistributedSampler(item['dataset'])
            self.dataloader_dict[key]['dataloader'] = DataLoader(
                item['dataset'],
                sampler=self.dataloader_dict[key]['sampler'],
                batch_size=batch_size,
                num_workers=num_workers,
            )


        self.model = MashDecoder(
            dtype=self.dtype,
            device=self.device,
        ).to(self.device)

        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        if model_file_path is not None:
            self.loadModel(model_file_path)

        self.optim = AdamW(self.model.parameters(), lr=self.lr)
        self.sched = LambdaLR(self.optim, lr_lambda=self.warmup_lr)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.initRecords()

        self.min_lr_reach_time = 0

        self.gt_sample_added_to_logger = False
        return

    def initRecords(self) -> bool:
        if self.logger is None:
            return True

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_state_dict = torch.load(model_file_path)
        if 'model' in model_state_dict.keys():
            self.model.module.load_state_dict(model_state_dict["model"])
        if 'step' in model_state_dict.keys():
            self.step = model_state_dict['step']
        if 'loss_min' in model_state_dict.keys():
            self.loss_min = model_state_dict['loss_min']

        print('[INFO][Trainer::loadModel]')
        print('\t model loaded from:', model_file_path)

        return True

    def getLr(self) -> float:
        return self.optim.state_dict()["param_groups"][0]["lr"]

    def warmup_lr(self, step: int) -> float:
        if self.warm_step_num == 0:
            return 1.0

        return min(step, self.warm_step_num) / self.warm_step_num

    def trainStep(
        self,
        data: dict,
    ) -> dict:
        self.model.train()

        for key in data.keys():
            data[key] = data[key].to(self.device)

        gt_occ = data["occ"]

        occ, kl = self.model(data)

        loss_occ = self.loss_fn(occ, gt_occ)

        loss_kl = torch.sum(kl) / kl.shape[0]
        weighted_loss_kl = self.loss_kl_weight * loss_kl

        loss = loss_occ + weighted_loss_kl

        loss_item = loss.clone().detach().cpu().numpy()

        accum_loss = loss / self.accum_iter

        accum_loss.backward()

        if not check_and_replace_nan_in_grad(self.model):
            print('[ERROR][Trainer::trainStep]')
            print('\t check_and_replace_nan_in_grad failed!')
            exit()

        if (self.step + 1) % self.accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.sched.step()
            self.optim.zero_grad()

        acc = cal_occ_acc(occ, gt_occ)
        positive_occ_percent = cal_occ_positive_percent(gt_occ)

        loss_dict = {
            "LossOCC": loss_occ.clone().detach().cpu().numpy(),
            "LossKL": loss_kl.clone().detach().cpu().numpy(),
            "Loss": loss_item,
            "Accuracy": acc,
            "PositiveOCC": positive_occ_percent,
        }

        return loss_dict

    @torch.no_grad()
    def evalStep(self, data: dict) -> dict:
        self.model.eval()

        for key in data.keys():
            data[key] = data[key].to(self.device)

        occ, _ = self.model.module(data, drop_prob=0.0, deterministic=True)

        gt_occ = data["occ"]

        loss = self.loss_fn(occ, gt_occ)

        loss_item = loss.item()
        acc = cal_occ_acc(occ, gt_occ)
        positive_occ_percent = cal_occ_positive_percent(gt_occ)

        loss_dict = {
            'Loss': loss_item,
            'Accuracy': acc,
            'PositiveOCC': positive_occ_percent,
        }

        return loss_dict

    def trainEpoch(self, data_name: str) -> bool:
        if data_name not in self.dataloader_dict.keys():
            print('[ERROR][Trainer::trainEpoch]')
            print('\t data not exist!')
            print('\t data_name:', data_name)
            return False

        dataloader_dict = self.dataloader_dict[data_name]
        dataloader_dict['sampler'].set_epoch(self.epoch)

        dataloader = dataloader_dict['dataloader']

        if self.local_rank == 0:
            pbar = tqdm(total=len(dataloader))
        for data in dataloader:
            train_loss_dict = self.trainStep(data)

            self.loss_dict_list.append(train_loss_dict)

            lr = self.getLr()

            if (self.step + 1) % self.accum_iter == 0 and self.local_rank == 0:
                for key in train_loss_dict.keys():
                    value = 0
                    for i in range(len(self.loss_dict_list)):
                        value += self.loss_dict_list[i][key]
                    value /= len(self.loss_dict_list)
                    self.logger.addScalar("Train/" + key, value, self.step)
                self.logger.addScalar("Train/Lr", lr, self.step)

                self.loss_dict_list = []

            if self.local_rank == 0:
                pbar.set_description(
                    "EPOCH %d LOSS %.6f LR %.4f"
                    % (
                        self.epoch,
                        train_loss_dict["Loss"],
                        self.getLr() / self.lr,
                    )
                )

            self.step += 1

            if self.local_rank == 0:
                pbar.update(1)

        if self.local_rank == 0:
            pbar.close()

        self.epoch += 1

        return True

    @torch.no_grad()
    def evalEpoch(self) -> bool:
        if self.local_rank != 0:
            return True

        if 'eval' not in self.dataloader_dict.keys():
            return True

        print('[INFO][Trainer::evalEpoch]')
        print('\t start evaluating ...')

        dataloader = self.dataloader_dict['eval']['dataloader']

        avg_loss = 0
        avg_acc = 0
        avg_positive_occ_percent = 0
        ni = 0

        print("[INFO][Trainer::evalEpoch]")
        print("\t start eval loss and acc...")
        for data in tqdm(dataloader, total=len(dataloader)):
            loss_dict = self.evalStep(data)

            avg_loss += loss_dict['Loss']
            avg_acc += loss_dict['Accuracy']
            avg_positive_occ_percent += loss_dict['PositiveOCC']

            ni += 1

        avg_loss /= ni
        avg_acc /= ni
        avg_positive_occ_percent /= ni

        eval_loss_dict = {
            "Loss": avg_loss,
            "Accuracy": avg_acc,
            "PositiveOCC": avg_positive_occ_percent,
        }

        for key, item in eval_loss_dict.items():
            self.logger.addScalar("Eval/" + key, item, self.step)

        self.autoSaveModel('best', eval_loss_dict["Accuracy"], False)

        return True

    def train(self) -> bool:
        final_step = self.step + self.finetune_step_num

        if self.local_rank == 0:
            print("[INFO][Trainer::train]")
            print("\t start training ...")

        while self.step < final_step or self.finetune_step_num < 0:

            for data_name in self.dataloader_dict.keys():
                if data_name == 'eval':
                    continue

                repeat_num = self.dataloader_dict[data_name]['repeat_num']

                for i in range(repeat_num):
                    if self.local_rank == 0:
                        print('[INFO][Trainer::train]')
                        print('\t start training on dataset [', data_name, '] ,', i + 1, '/', repeat_num, '...')

                    if not self.trainEpoch(data_name):
                        print('[ERROR][Trainer::train]')
                        print('\t trainEpoch failed!')
                        return False

                    self.autoSaveModel("last")

                    if not self.evalEpoch():
                        print('[ERROR][Trainer::train]')
                        print('\t evalEpoch failed!')
                        return False

                    if self.epoch % 1 == 0:
                        #self.sampleStep()
                        pass

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.state_dict(),
            "step": self.step,
            "loss_min": self.loss_min,
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, name: str, value: Union[float, None] = None, check_lower: bool = True) -> bool:
        if self.local_rank != 0:
            return True

        if self.save_result_folder_path is None:
            return False

        if value is not None:
            if self.loss_min == float("inf"):
                if not check_lower:
                    self.loss_min = -float("inf")

            if check_lower:
                if value > self.loss_min:
                    return False
            elif value < self.loss_min:
                return False

            self.loss_min = value

        save_model_file_path = self.save_result_folder_path + "model_" + name + ".pth"

        tmp_save_model_file_path = save_model_file_path[:-4] + "_tmp.pth"

        self.saveModel(tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)

        return True
