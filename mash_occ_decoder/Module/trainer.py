import os
import torch
from torch import nn
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from mash_occ_decoder.Dataset.sdf import SDFDataset
from mash_occ_decoder.Method.time import getCurrentTime
from mash_occ_decoder.Method.path import createFileFolder
from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Module.logger import Logger


def cal_occ_acc(occ, gt_occ):
    positive_acc_num = torch.where((occ.sigmoid() > 0.5) & (gt_occ > 0.5))[0].shape[0]
    negative_acc_num = torch.where((occ.sigmoid() < 0.5) & (gt_occ < 0.5))[0].shape[0]

    occ_num = 1
    for size in gt_occ.shape:
        occ_num *= size

    acc = (positive_acc_num + negative_acc_num) / occ_num
    return acc


def cal_sdf_acc(sdf, gt_occ):
    occ = torch.ones_like(sdf)
    occ[sdf > 0.0] = 0.0
    positive_acc_num = torch.where((occ > 0.5) & (gt_occ > 0.5))[0].shape[0]
    negative_acc_num = torch.where((occ < 0.5) & (gt_occ < 0.5))[0].shape[0]

    occ_num = 1
    for size in gt_occ.shape:
        occ_num *= size

    acc = (positive_acc_num + negative_acc_num) / occ_num
    return acc


class Trainer(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 400,
        num_workers: int = 4,
        n_qry: int = 200,
        model_file_path: Union[str, None] = None,
        dtype=torch.float64,
        device: str = "cpu",
        warm_epoch_step_num: int = 20,
        warm_epoch_num: int = 10,
        finetune_step_num: int = 400,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        factor: float = 0.9,
        patience: int = 1,
        min_lr: float = 1e-4,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.dtype = dtype
        self.device = device

        self.warm_epoch_step_num = warm_epoch_step_num
        self.warm_epoch_num = warm_epoch_num

        self.finetune_step_num = finetune_step_num

        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.train_loader = DataLoader(
            SDFDataset(dataset_root_folder_path, "train", n_qry),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            SDFDataset(dataset_root_folder_path, "val", n_qry),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.model = MashDecoder(dtype=self.dtype, device=self.device).to(self.device)

        self.occ_loss_fn = nn.BCEWithLogitsLoss()
        self.sdf_loss_fn = nn.SmoothL1Loss()

        self.occ_loss_weight = 1.0
        self.sdf_loss_weight = 100.0

        self.initRecords()

        if model_file_path is not None:
            self.loadModel(model_file_path)

        self.min_lr_reach_time = 0
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

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
        self.model.load_state_dict(model_state_dict["model"])
        return True

    def getLr(self, optimizer) -> float:
        return optimizer.state_dict()["param_groups"][0]["lr"]

    def toTrainStepNum(self, scheduler: LRScheduler) -> int:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            return self.finetune_step_num

        if scheduler.T_mult == 1:
            warm_epoch_num = scheduler.T_0 * self.warm_epoch_num
        else:
            warm_epoch_num = int(
                scheduler.T_mult
                * (1.0 - pow(scheduler.T_mult, self.warm_epoch_num))
                / (1.0 - scheduler.T_mult)
            )

        return self.warm_epoch_step_num * warm_epoch_num

    def trainStep(
        self,
        data: dict,
        optimizer: Optimizer,
    ) -> dict:
        for key in data.keys():
            data[key] = data[key].to(self.device)
        optimizer.zero_grad()

        gt_occ = data["occ"]

        occ, sdf = self.model(data)

        occ_loss = self.occ_loss_fn(occ, gt_occ) * self.occ_loss_weight
        sdf_loss = self.sdf_loss_fn(sdf, data["sdf"]) * self.sdf_loss_weight

        loss = occ_loss + sdf_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            occ_acc = cal_occ_acc(occ, gt_occ)
            sdf_acc = cal_sdf_acc(sdf, gt_occ)

        loss_dict = {
            "OCCLoss": occ_loss.item(),
            "SDFLoss": sdf_loss.item(),
            "Loss": loss.item(),
            "OCCAccuracy": occ_acc,
            "SDFAccuracy": sdf_acc,
        }

        return loss_dict

    @torch.no_grad()
    def valStep(self) -> dict:
        avg_occ_loss = 0
        avg_sdf_loss = 0
        avg_loss = 0
        avg_occ_acc = 0
        avg_sdf_acc = 0
        avg_positive_occ_percent = 0
        ni = 0

        print("[INFO][Trainer::valStep]")
        print("\t start val loss and acc...")
        for data in tqdm(self.val_loader):
            for key in data.keys():
                data[key] = data[key].to(self.device)

            occ, sdf = self.model(data)

            gt_occ = data["occ"]
            gt_sdf = data["sdf"]

            occ_loss = self.occ_loss_fn(occ, gt_occ) * self.occ_loss_weight
            sdf_loss = self.sdf_loss_fn(sdf, gt_sdf) * self.sdf_loss_weight

            loss = occ_loss + sdf_loss

            occ_acc = cal_occ_acc(occ, gt_occ)
            sdf_acc = cal_sdf_acc(sdf, gt_occ)

            avg_occ_loss += occ_loss.item()
            avg_sdf_loss += sdf_loss.item()
            avg_loss += loss.item()
            avg_occ_acc += occ_acc
            avg_sdf_acc += sdf_acc

            gt_occ_shape = gt_occ.shape
            positive_occ_num = torch.where(gt_occ > 0.5)[0].shape[0]
            occ_num = 1
            for size in gt_occ_shape:
                occ_num *= size

            positive_occ_percent = 1.0 * positive_occ_num / occ_num
            avg_positive_occ_percent += positive_occ_percent

            ni += 1

        avg_occ_loss /= ni
        avg_sdf_loss /= ni
        avg_loss /= ni
        avg_occ_acc /= ni
        avg_sdf_acc /= ni
        avg_positive_occ_percent /= ni

        loss_dict = {
            "OCCLoss": avg_occ_loss,
            "SDFLoss": avg_sdf_loss,
            "Loss": avg_loss,
            "OCCAccuracy": avg_occ_acc,
            "SDFAccuracy": avg_sdf_acc,
            "PositiveOCC": avg_positive_occ_percent,
        }

        return loss_dict

    def checkStop(
        self, optimizer: Optimizer, scheduler: LRScheduler, loss_dict: dict
    ) -> bool:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(loss_dict["Loss"])

            if self.getLr(optimizer) == self.min_lr:
                self.min_lr_reach_time += 1

            return self.min_lr_reach_time > self.patience

        current_warm_epoch = self.step / self.warm_epoch_step_num
        scheduler.step(current_warm_epoch)

        return current_warm_epoch >= self.warm_epoch_num

    def train(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> bool:
        train_step_num = self.toTrainStepNum(scheduler)
        final_step = self.step + train_step_num

        eval_step = 0

        print("[INFO][Trainer::train]")
        print("\t start training ...")
        pbar = tqdm(total=final_step)
        pbar.update(self.step)
        while self.step < final_step:
            self.model.train()

            for data in tqdm(self.train_loader):
                train_loss_dict = self.trainStep(data, optimizer)

                lr = self.getLr(optimizer)

                if self.logger.isValid():
                    for key, item in train_loss_dict.items():
                        self.logger.addScalar("Train/" + key, item, self.step)
                    self.logger.addScalar("Train/Lr", lr, self.step)

                    gt_occ = data["occ"]
                    occ_shape = gt_occ.shape
                    positive_occ_num = torch.where(gt_occ > 0.5)[0].shape[0]
                    occ_num = 1
                    for size in occ_shape:
                        occ_num *= size

                    positive_occ_percent = 1.0 * positive_occ_num / occ_num
                    self.logger.addScalar(
                        "Train/PositiveOCC", positive_occ_percent, self.step
                    )

                pbar.set_description(
                    "LOSS %.6f LR %.4f"
                    % (
                        train_loss_dict["Loss"],
                        self.getLr(optimizer) / self.lr,
                    )
                )

                self.step += 1
                pbar.update(1)

                if self.checkStop(optimizer, scheduler, train_loss_dict):
                    break

                if self.step >= final_step:
                    break

            eval_step += 1
            if eval_step % 1 == 0:
                print("[INFO][Trainer::train]")
                print("\t start eval on val dataset...")
                self.model.eval()
                eval_loss_dict = self.valStep()

                if self.logger.isValid():
                    for key, item in eval_loss_dict.items():
                        self.logger.addScalar("Eval/" + key, item, self.step)

                print(
                    " loss:",
                    eval_loss_dict["Loss"],
                    " acc:",
                    eval_loss_dict["OCCAccuracy"],
                    " positive occ:",
                    eval_loss_dict["PositiveOCC"],
                )

                self.autoSaveModel(eval_loss_dict["OCCAccuracy"], False)

            # self.autoSaveModel(train_loss_dict['OCCAccuracy'], False)

        return True

    def autoTrain(
        self,
    ) -> bool:
        print("[INFO][Trainer::autoTrain]")
        print("\t start auto train mash occ decoder...")

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        warm_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)
        finetune_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )

        self.train(optimizer, warm_scheduler)
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr
        self.train(optimizer, finetune_scheduler)

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.state_dict(),
            "loss_min": self.loss_min,
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, value: float, check_lower: bool = True) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_last_model_file_path = self.save_result_folder_path + "model_last.pth"

        self.saveModel(save_last_model_file_path)

        if self.loss_min == float("inf"):
            if not check_lower:
                self.loss_min = -float("inf")

        if check_lower:
            if value > self.loss_min:
                return False
        else:
            if value < self.loss_min:
                return False

        self.loss_min = value

        save_best_model_file_path = self.save_result_folder_path + "model_best.pth"

        self.saveModel(save_best_model_file_path)

        return True
