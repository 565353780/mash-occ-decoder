import os
import torch
import trimesh
from typing import Union

from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Method.io import loadMashTensor
from mash_occ_decoder.Method.tomesh import extractMesh


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = True,
        batch_size: int = 1200000,
        resolution: int = 128,
        device: str = "cpu",
    ) -> None:
        self.batch_size = batch_size
        self.resolution = resolution
        self.device = device

        self.model = MashDecoder().to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path, use_ema)
        return

    def loadModel(self, model_file_path: str, use_ema: bool = True) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")

        if use_ema:
            model_state_dict = state_dict["ema_model"]
        else:
            model_state_dict = state_dict["model"]

        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def detect(self, mash_params: torch.Tensor) -> Union[trimesh.Trimesh, None]:
        mesh = extractMesh(
            mash_params, self.model, self.resolution, self.batch_size, "odc"
        )

        return mesh

    @torch.no_grad()
    def detectFile(self, mash_params_file_path: str) -> Union[trimesh.Trimesh, None]:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        mash_params = loadMashTensor(mash_params_file_path)
        if mash_params is None:
            print("[ERROR][Detector::detectFile]")
            print("\t loadMashTensor failed!")
            return None

        mash_params = mash_params.to(self.device)

        mesh = self.detect(mash_params)

        return mesh
