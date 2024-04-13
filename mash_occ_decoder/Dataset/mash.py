import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
        n_qry: int = 200,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.n_qry = n_qry

        self.files = []

        categories = os.listdir(dataset_root_folder_path + "04_splits/")

        for category in categories:
            split_file_path = (
                self.dataset_root_folder_path
                + "04_splits/"
                + category
                + "/mash/"
                + self.split
                + ".lst"
            )
            with open(split_file_path, "r") as f:
                id_shapes = f.read().split()

            for shape_id in id_shapes:
                qry_file_path = f"{self.dataset_root_folder_path}/02_qry_pts_occnet/{category}/{shape_id[:-4]}.npy"
                occ_file_path = f"{self.dataset_root_folder_path}/03_qry_occs_occnet/{category}/{shape_id[:-4]}.npy"

                if not os.path.exists(qry_file_path):
                    continue
                if not os.path.exists(occ_file_path):
                    continue

                self.files.append((category, shape_id))
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        category, full_shape_id = self.files[index]
        shape_id = full_shape_id[:-4]

        qry = np.load(
            f"{self.dataset_root_folder_path}/02_qry_pts_occnet/{category}/{shape_id}.npy"
        )
        occ = np.load(
            f"{self.dataset_root_folder_path}/03_qry_occs_occnet/{category}/{shape_id}.npy"
        )

        mash_params_file_path = (
            f"{self.dataset_root_folder_path}/mash/{category}/{full_shape_id}.npy"
        )

        mash_params = np.load(mash_params_file_path, allow_pickle=True).item()

        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]
        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]

        mash_params = np.hstack([mask_params, sh_params, rotate_vectors, positions])

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        positive_occ_idxs = np.where(occ > 0.5)[0]
        negative_occ_idxs = np.where(occ < 0.5)[0]

        positive_occ_num = self.n_qry // 2

        if positive_occ_num > positive_occ_idxs.shape[0]:
            positive_occ_num = positive_occ_idxs.shape[0]

        negative_occ_num = self.n_qry - positive_occ_num

        positive_idxs = np.random.choice(positive_occ_idxs, positive_occ_num)
        negative_idxs = np.random.choice(negative_occ_idxs, negative_occ_num)

        idxs = np.hstack([positive_idxs, negative_idxs])

        perm = idxs[np.random.permutation(positive_occ_num + negative_occ_num)]

        qry = qry[perm]
        occ = occ[perm]
        mash_params = mash_params[np.random.permutation(mash_params.shape[0])]

        feed_dict = {
            "qry": torch.tensor(qry).float(),
            "mash_params": torch.tensor(mash_params).float(),
            "occ": torch.tensor(occ).float(),
        }

        return feed_dict
