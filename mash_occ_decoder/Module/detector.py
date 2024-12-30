import os
import torch
import trimesh
from typing import Union

from ma_sh.Method.io import loadMashFileParamsTensor
from ma_sh.Method.transformer import getTransformer

from mash_occ_decoder.Model.mash_decoder import MashDecoder
from mash_occ_decoder.Lib.ODC.occupancy_dual_contouring import occupancy_dual_contouring


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

        self.odc = occupancy_dual_contouring(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str, use_ema: bool = True) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")

        if use_ema:
            model_state_dict = state_dict['ema_model']
        else:
            model_state_dict = state_dict['model']

        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        print('[INFO][Detector::loadModel]')
        print('\t load model success!')
        print('\t model_file_path:', model_file_path)
        return True

    @torch.no_grad()
    def detect(self, mash_params: torch.Tensor) -> Union[trimesh.Trimesh, None]:
        mash_params = self.transformer.transform(mash_params)

        mash_params = mash_params.unsqueeze(0)

        def toOCC(xyz: torch.Tensor) -> torch.Tensor:
            data = {
                'mash_params': mash_params,
                'qry': xyz.to(self.device, dtype=mash_params.dtype).unsqueeze(0),
                'drop_prob': 0.0,
            }

            results = self.model(data)

            occ = results['occ'].reshape(-1)

            return occ.to(xyz.dtype)

        vertices, triangles = self.odc.extract_mesh(
            imp_func=toOCC,
            num_grid=128,
            isolevel=0.5,
            batch_size=1200000,
            outside=False,
        )

        mesh = trimesh.Trimesh(vertices.cpu(), triangles.cpu())

        return mesh

    @torch.no_grad()
    def detectFile(self, mash_params_file_path: str) -> Union[trimesh.Trimesh, None]:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        mash_params = loadMashFileParamsTensor(mash_params_file_path, torch.float32, self.device)

        mesh = self.detect(mash_params)

        return mesh
