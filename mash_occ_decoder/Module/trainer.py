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

from mash_occ_decoder.Config.config import MASH_DECODER_CONFIG
from mash_occ_decoder.Dataset.mash import MashDataset
from mash_occ_decoder.Method.time import getCurrentTime
from mash_occ_decoder.Method.path import createFileFolder
from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Model.mash_decoder_v2 import MashDecoderV2
from mash_occ_decoder.Module.logger import Logger


def cal_acc(occ, gt_occ):
    positive_acc_num = torch.where((occ.sigmoid() > 0.5) & (gt_occ > 0.5))[0].shape[0]
    negative_acc_num = torch.where((occ.sigmoid() < 0.5) & (gt_occ < 0.5))[0].shape[0]

    occ_num = 1
    for size in gt_occ.shape:
        occ_num *= size

    acc = (positive_acc_num + negative_acc_num) / occ_num
    return acc


class Trainer(object):
    def __init__(
        self,
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
            MashDataset("train"),
            shuffle=True,
            batch_size=MASH_DECODER_CONFIG.n_bs,
            num_workers=MASH_DECODER_CONFIG.n_wk,
        )
        self.val_loader = DataLoader(
            MashDataset("val"),
            shuffle=False,
            batch_size=MASH_DECODER_CONFIG.n_bs,
            num_workers=MASH_DECODER_CONFIG.n_wk,
        )

        self.model = MashDecoderV2().to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()

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

        occ = self.model(data)

        loss = self.loss_fn(occ, data["occ"])

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = cal_acc(occ, data["occ"])

        loss_dict = {
            "loss": loss.item(),
            "acc": acc,
        }

        return loss_dict

    @torch.no_grad()
    def valStep(self) -> dict:
        avg_loss = 0
        avg_acc = 0
        avg_positive_occ_percent = 0
        ni = 0

        print("[INFO][Trainer::valStep]")
        print("\t start val loss and acc...")
        for data in tqdm(self.val_loader):
            for key in data:
                data[key] = data[key].to(self.device)

            occ = self.model(data)

            gt_occ = data["occ"]

            loss = self.loss_fn(occ, gt_occ)

            acc = cal_acc(occ, gt_occ)

            avg_loss += loss.item()
            avg_acc += acc

            gt_occ_shape = gt_occ.shape
            positive_occ_num = torch.where(gt_occ > 0.5)[0].shape[0]
            occ_num = 1
            for size in gt_occ_shape:
                occ_num *= size

            positive_occ_percent = 1.0 * positive_occ_num / occ_num
            avg_positive_occ_percent += positive_occ_percent

            ni += 1

        avg_loss /= ni
        avg_acc /= ni
        avg_positive_occ_percent /= ni

        loss_dict = {
            "loss": avg_loss,
            "acc": avg_acc,
            "positive_occ_percent": avg_positive_occ_percent,
        }

        return loss_dict

    def checkStop(
        self, optimizer: Optimizer, scheduler: LRScheduler, loss_dict: dict
    ) -> bool:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(loss_dict["loss"])

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

        print("[INFO][Trainer::train]")
        print("\t start training ...")
        pbar = tqdm(total=final_step)
        pbar.update(self.step)
        while self.step < final_step:
            self.model.train()

            for data in self.train_loader:
                train_loss_dict = self.trainStep(data, optimizer)
                train_loss = train_loss_dict["loss"]
                train_acc = train_loss_dict["acc"]

                lr = self.getLr(optimizer)

                if self.logger.isValid():
                    self.logger.addScalar("Train/Loss", train_loss, self.step)
                    self.logger.addScalar("Train/Accuracy", train_acc, self.step)
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
                        train_loss,
                        self.getLr(optimizer) / self.lr,
                    )
                )

                self.step += 1
                pbar.update(1)

                if self.checkStop(optimizer, scheduler, train_loss_dict):
                    break

                if self.step >= final_step:
                    break

            if False:
                print("[INFO][Trainer::train]")
                print("\t start eval on val dataset...")
                self.model.eval()
                eval_loss_dict = self.valStep()
                eval_loss = eval_loss_dict["loss"]
                eval_acc = eval_loss_dict["acc"]
                eval_positive_occ_percent = eval_loss_dict["positive_occ_percent"]

                if self.logger.isValid():
                    self.logger.addScalar("Eval/Loss", eval_loss, self.step)
                    self.logger.addScalar("Eval/Accuracy", eval_acc, self.step)
                    self.logger.addScalar(
                        "Eval/PositiveOCC",
                        eval_positive_occ_percent,
                        self.step,
                    )

                print(
                    " loss:",
                    eval_loss,
                    " acc:",
                    eval_acc,
                    " positive occ:",
                    eval_positive_occ_percent,
                )

                self.autoSaveModel(eval_loss_dict)

            self.autoSaveModel(train_loss_dict)

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

    def autoSaveModel(self, loss_dict: dict) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_last_model_file_path = self.save_result_folder_path + "model_last.pth"

        self.saveModel(save_last_model_file_path)

        if loss_dict["loss"] > self.loss_min:
            return False

        self.loss_min = loss_dict["loss"]

        save_best_model_file_path = self.save_result_folder_path + "model_best.pth"

        self.saveModel(save_best_model_file_path)

        return True
