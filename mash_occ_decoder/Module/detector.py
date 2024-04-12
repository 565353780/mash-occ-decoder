import os
import torch
import trimesh
from typing import Union

from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Module.generator_3d import Generator3D


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        dtype=torch.float32,
        device: str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device

        self.model = MashDecoder(dtype=self.dtype, device=self.device)

        self.generator = Generator3D(self.model, device=self.device)

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
        self.model.to(self.device)
        self.model.eval()
        return True

    @torch.no_grad()
    def detectFile(self, mash_params_file_path: str) -> Union[trimesh.Trimesh, None]:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        out = self.generator.generate_mesh(mash_params_file_path)

        if isinstance(out, trimesh.Trimesh):
            mesh = out
        else:
            mesh = out[0]

        return mesh
