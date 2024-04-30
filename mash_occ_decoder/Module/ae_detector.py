import os
import torch
import numpy as np
from math import sqrt
from typing import Union

from ma_sh.Model.mash import Mash

from mash_occ_decoder.Model.sh_ae import SHAutoEncoder


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        dtype=torch.float32,
        device: str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device

        self.model = SHAutoEncoder().to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")["model"]

        self.model.load_state_dict(state_dict)
        self.model.eval()
        return True

    @torch.no_grad()
    def detectFile(self, mash_params_file_path: str) -> Mash:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        gt_mash_params_dict = np.load(mash_params_file_path, allow_pickle=True).item()

        gt_rotate_vectors = gt_mash_params_dict["rotate_vectors"]
        gt_positions = gt_mash_params_dict["positions"]
        gt_mask_params = gt_mash_params_dict["mask_params"]
        gt_sh_params = gt_mash_params_dict["sh_params"]
        use_inv = gt_mash_params_dict["use_inv"]

        anchor_num = gt_mask_params.shape[0]
        mask_degree_max = int((gt_mask_params.shape[1] - 1) / 2)
        sh_degree_max = int(sqrt(gt_sh_params.shape[1] - 1))

        gt_mash_params = (
            torch.from_numpy(
                np.hstack(
                    [gt_rotate_vectors, gt_positions, gt_mask_params, gt_sh_params]
                )
            )
            .unsqueeze(0)
            .to(self.device)
        )

        mash_params = self.model(gt_mash_params)[0]

        rotate_vectors = gt_rotate_vectors
        positions = gt_positions
        sh2d = 2 * mask_degree_max + 1
        mask_params = mash_params[:, :sh2d]
        sh_params = mash_params[:, sh2d:]

        mash = Mash(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
        )
        mash.loadParams(mask_params, sh_params, rotate_vectors, positions, use_inv)

        return mash
