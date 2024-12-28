import os
import torch
import trimesh
from typing import Union

from ma_sh.Method.io import loadMashFileParamsTensor
from ma_sh.Method.transformer import getTransformer

from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Module.generator_3d import Generator3D


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        transformer_id: str = 'Objaverse_82K',
        device: str = "cpu",
    ) -> None:
        self.device = device

        self.transformer = getTransformer(transformer_id)
        assert self.transformer is not None

        self.model = MashDecoder().to(self.device)

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

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        print('[INFO][Detector::loadModel]')
        print('\t load model success!')
        print('\t model_file_path:', model_file_path)
        return True

    @torch.no_grad()
    def detectFile(self, mash_params_file_path: str) -> Union[trimesh.Trimesh, None]:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        mash_params = loadMashFileParamsTensor(mash_params_file_path, torch.float32, self.device)

        mash_params = self.transformer.transform(mash_params)

        out = self.generator.generate_mesh(mash_params)

        if isinstance(out, trimesh.Trimesh):
            mesh = out
        else:
            mesh = out[0]

        return mesh
