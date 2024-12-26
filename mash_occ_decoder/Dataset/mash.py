import os
import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Method.io import loadMashFileParamsTensor

from mash_occ_decoder.Config.transformer import getTransformer


class MashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
        train_percent: float = 0.9,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split

        self.mash_folder_path = self.dataset_root_folder_path + "Objaverse_82K/manifold_mash/"
        assert os.path.exists(self.mash_folder_path)

        self.paths_list = []

        print("[INFO][MashDataset::__init__]")
        print("\t start load dataset:", self.mash_folder_path)
        for root, _, files in os.walk(self.mash_folder_path):

            for file in files:
                if not file.endswith('.npy'):
                    continue

                mash_file_path = root + '/' + file

                self.paths_list.append(mash_file_path)

        self.paths_list.sort()

        train_data_num = max(int(len(self.paths_list) * train_percent), 1)

        if self.split == 'train':
            self.paths_list = self.paths_list[:train_data_num]
        else:
            self.paths_list = self.paths_list[train_data_num:]

        self.transformer = getTransformer('Objaverse_82K')
        assert self.transformer is not None
        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.transform(mash_params, False)

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(mash_params, False)

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_path = self.paths_list[index]

        mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')

        mash_params = self.normalize(mash_params)

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        feed_dict = {
            "mash_params": mash_params,
        }

        return feed_dict
