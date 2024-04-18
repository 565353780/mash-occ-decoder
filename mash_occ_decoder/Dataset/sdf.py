import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
        n_qry: int = 200,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.n_qry = n_qry

        self.mash_folder_path = self.dataset_root_folder_path + "Mash/"
        self.sdf_folder_path = self.dataset_root_folder_path + "SDF/"
        self.split_folder_path = self.dataset_root_folder_path + "Split/"

        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.sdf_folder_path)
        assert os.path.exists(self.split_folder_path)

        self.paths_list = []

        dataset_name_list = os.listdir(self.split_folder_path + "sdf/")

        for dataset_name in dataset_name_list:
            sdf_split_folder_path = self.split_folder_path + "sdf/" + dataset_name + "/"

            categories = os.listdir(sdf_split_folder_path)
            # FIXME: for detect test only
            if self.split == "test":
                categories = ["03001627"]

            for i, category in enumerate(categories):
                rel_file_path_list_file_path = (
                    sdf_split_folder_path + category + "/" + self.split + ".txt"
                )
                if not os.path.exists(rel_file_path_list_file_path):
                    continue

                with open(rel_file_path_list_file_path, "r") as f:
                    rel_file_path_list = f.read().split()

                print("[INFO][SDFDataset::__init__]")
                print(
                    "\t start load dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(i + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for rel_file_path in tqdm(rel_file_path_list):
                    mash_file_path = (
                        self.mash_folder_path
                        + dataset_name
                        + "/mash/"
                        + category
                        + "/"
                        + rel_file_path
                    )

                    sdf_file_path = (
                        self.sdf_folder_path
                        + dataset_name
                        + "/sdf/"
                        + category
                        + "/"
                        + rel_file_path
                    )

                    self.paths_list.append([mash_file_path, sdf_file_path])
        return

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        mash_file_path, sdf_file_path = self.paths_list[index]

        mash_params = np.load(mash_file_path, allow_pickle=True).item()
        sdf_data = np.load(sdf_file_path)

        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]
        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]

        mash_params = np.hstack([mask_params, sh_params, rotate_vectors, positions])

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        points = sdf_data[:, :3]
        sdf = sdf_data[:, 3]

        positive_sdf_idxs = np.where(sdf_data[:, 3] <= 0.0)[0]
        negative_sdf_idxs = np.where(sdf_data[:, 3] > 0.0)[0]

        occ = np.ones_like(sdf)
        occ[negative_sdf_idxs] = 0.0

        positive_sdf_num = self.n_qry // 2

        if positive_sdf_num > positive_sdf_idxs.shape[0]:
            positive_sdf_num = positive_sdf_idxs.shape[0]

        negative_sdf_num = self.n_qry - positive_sdf_num

        if negative_sdf_num > negative_sdf_idxs.shape[0]:
            negative_sdf_num = negative_sdf_idxs.shape[0]

        positive_idxs = np.random.choice(positive_sdf_idxs, positive_sdf_num)
        negative_idxs = np.random.choice(negative_sdf_idxs, negative_sdf_num)

        idxs = np.hstack([positive_idxs, negative_idxs])

        perm = idxs[np.random.permutation(positive_sdf_num + negative_sdf_num)]

        qry = points[perm]
        occ = occ[perm]
        sdf = sdf[perm]
        # sdf = np.clip(sdf, -1.0, 1.0)
        mash_params = mash_params[np.random.permutation(mash_params.shape[0])]

        feed_dict = {
            "qry": torch.tensor(qry).float(),
            "mash_params": torch.tensor(mash_params).float(),
            "occ": torch.tensor(occ).float(),
            "sdf": torch.tensor(sdf).float(),
        }

        return feed_dict
