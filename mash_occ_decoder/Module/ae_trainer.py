import os
import torch
from torch import nn
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    ReduceLROnPlateau,
)

from mash_occ_decoder.Dataset.mash import MashDataset
from mash_occ_decoder.Method.time import getCurrentTime
from mash_occ_decoder.Method.path import createFileFolder
from mash_occ_decoder.Model.sh_ae import SHAutoEncoder
from mash_occ_decoder.Module.logger import Logger


class Trainer(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 400,
        num_workers: int = 4,
        model_file_path: Union[str, None] = None,
        dtype=torch.float64,
        device: str = "cpu",
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
            MashDataset(dataset_root_folder_path, "train", True),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            MashDataset(dataset_root_folder_path, "val", True),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.model = SHAutoEncoder().to(self.device)

        self.initRecords()

        self.loss_fn = nn.MSELoss()

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

    def trainStep(
        self,
        data: dict,
        optimizer: Optimizer,
    ) -> dict:
        for key in data.keys():
            data[key] = data[key].to(self.device)

        gt_mash = data["mash_params"]

        mash = self.model(gt_mash)

        loss = self.loss_fn(mash, gt_mash[:, :, 6:])

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_dict = {
            "Loss": loss.item(),
        }

        return loss_dict

    @torch.no_grad()
    def valStep(self) -> dict:
        avg_loss = 0
        ni = 0

        print("[INFO][Trainer::valStep]")
        print("\t start val loss and acc...")
        for data in tqdm(self.val_loader):
            for key in data.keys():
                data[key] = data[key].to(self.device)

            gt_mash = data["mash_params"]

            mash = self.model(gt_mash)

            loss = self.loss_fn(mash, gt_mash[:, :, 6:])

            avg_loss += loss.item()

            ni += 1

        avg_loss /= ni

        loss_dict = {
            "Loss": avg_loss,
        }

        return loss_dict

    def train(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> bool:
        train_step_num = 1000000000
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

                for key, value in train_loss_dict.items():
                    self.logger.addScalar("Train/" + key, value, self.step)
                self.logger.addScalar("Train/Lr", lr, self.step)

                pbar.set_description(
                    "LOSS %.6f LR %.4f"
                    % (
                        train_loss_dict["Loss"],
                        self.getLr(optimizer) / self.lr,
                    )
                )

                self.step += 1
                pbar.update(1)

                scheduler.step(train_loss_dict["Loss"])

            eval_step += 1
            if eval_step % 1 == 0:
                print("[INFO][Trainer::train]")
                print("\t start eval on val dataset...")
                self.model.eval()
                eval_loss_dict = self.valStep()

                for key, item in eval_loss_dict.items():
                    self.logger.addScalar("Eval/" + key, item, self.step)

                print(" loss:", eval_loss_dict["Loss"])

                self.autoSaveModel(eval_loss_dict["Loss"])

            # self.autoSaveModel(train_loss_dict['Loss'])

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
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )
        self.train(optimizer, scheduler)
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
