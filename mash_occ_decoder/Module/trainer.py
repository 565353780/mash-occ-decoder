import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader


from mash_occ_decoder.Config.config import MASH_DECODER_CONFIG
from mash_occ_decoder.Dataset.mash import MashDataset
from mash_occ_decoder.Method.time import getCurrentTime
from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Module.logger import Logger

CONFIG = MASH_DECODER_CONFIG
DATASET = MashDataset
NET = MashDecoder


def cal_acc(x, gt):
    acc = ((x.sigmoid() > 0.5) == (gt["occ"] > 0.5)).float().sum(dim=-1) / x.shape[1]
    acc = acc.mean(-1)
    return acc


def cal_loss_pred(x, gt):
    loss_pred = F.binary_cross_entropy_with_logits(x, gt["occ"])
    return loss_pred


class Trainer(object):
    def __init__(self, model_file_path: Union[str, None] = None) -> None:
        current_time = getCurrentTime()

        self.dir_ckpt = "./output/" + current_time + "/"
        self.log_folder_path = "./logs/" + current_time + "/"

        self.train_loader = DataLoader(
            DATASET("train"),
            shuffle=True,
            batch_size=CONFIG.n_bs,
            num_workers=CONFIG.n_wk,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            DATASET("val"),
            shuffle=False,
            batch_size=CONFIG.n_bs,
            num_workers=CONFIG.n_wk,
            drop_last=True,
        )

        self.model = NET().to(CONFIG.device)

        self.writer = Logger(self.log_folder_path)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_state_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_state_dict["model"])
        return True

    def train_step(self, batch, opt):
        for key in batch:
            batch[key] = batch[key].to(CONFIG.device)
        opt.zero_grad()
        x = self.model(batch)

        loss = cal_loss_pred(x, batch)

        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = cal_acc(x, batch)
        return loss.item(), acc.item()

    @torch.no_grad()
    def val_step(self):
        avg_loss_pred = 0
        avg_acc = 0
        ni = 0

        print("[INFO][Trainer::val_step]")
        print("\t start val loss and acc...")
        for batch in tqdm(self.val_loader):
            for key in batch:
                try:
                    batch[key] = batch[key].to(CONFIG.device)
                except:
                    pass
            x = self.model(batch)

            loss_pred = cal_loss_pred(x, batch)

            acc = cal_acc(x, batch)

            avg_loss_pred = avg_loss_pred + loss_pred.item()
            avg_acc = avg_acc + acc.item()
            ni += 1

        avg_loss_pred /= ni
        avg_acc /= ni
        return avg_loss_pred, avg_acc

    def train(self):
        os.makedirs(self.dir_ckpt, exist_ok=True)

        opt = optim.Adam(self.model.parameters(), lr=CONFIG.lr)

        fnames_ckpt = glob.glob(os.path.join(self.dir_ckpt, "*"))
        if len(fnames_ckpt) > 0:
            fname_ckpt_latest = max(fnames_ckpt, key=os.path.getctime)
            # path_ckpt = os.path.join(dir_ckpt, fname_ckpt_latest)
            ckpt = torch.load(fname_ckpt_latest)
            self.model.module.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            epoch_latest = ckpt["n_epoch"] + 1
            n_iter = ckpt["n_iter"]
            n_epoch = epoch_latest
        else:
            epoch_latest = 0
            n_iter = 0
            n_epoch = 0

        for i in range(epoch_latest, CONFIG.n_epochs):
            self.model.train()
            print("[INFO][Trainer::train]")
            print("\t start train mash occ itr", i + 1, "...")
            for batch in tqdm(self.train_loader):
                loss, acc = self.train_step(batch, opt)
                lr = opt.state_dict()["param_groups"][0]["lr"]

                self.writer.addScalar("Loss/train", loss, n_iter)
                self.writer.addScalar("Acc/train", acc, n_iter)
                self.writer.addScalar("Acc/lr", lr, n_iter)

                n_iter += 1

            self.model.eval()
            avg_loss_pred, avg_acc = self.val_step()
            self.writer.addScalar("Loss/val", avg_loss_pred, n_iter)
            self.writer.addScalar("Acc/val", avg_acc, n_iter)
            print(
                "[val] epcho:",
                n_epoch,
                " ,iter:",
                n_iter,
                " avg_loss_pred:",
                avg_loss_pred,
                " acc:",
                avg_acc,
            )

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "opt": opt.state_dict(),
                    "n_epoch": n_epoch,
                    "n_iter": n_iter,
                },
                f"{self.dir_ckpt}/{n_epoch}_{n_iter}.ckpt",
            )

            if n_epoch > 0 and n_epoch % CONFIG.freq_decay == 0:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * CONFIG.weight_decay

            n_epoch += 1

        return True
