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
        noise_label_list: list = ["0_25"],
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.n_qry = n_qry

        self.mash_folder_path = self.dataset_root_folder_path + "MashV3/"
        assert os.path.exists(self.mash_folder_path)

        self.sdf_folder_path_list = []
        self.split_folder_path_list = []
        for noise_label in noise_label_list:
            sdf_folder_path = (
                self.dataset_root_folder_path + "SampledSDF_" + noise_label + "/"
            )
            split_folder_path = (
                self.dataset_root_folder_path + "MashOCCSplit_" + noise_label + "/"
            )
            assert os.path.exists(sdf_folder_path)
            assert os.path.exists(split_folder_path)

            self.sdf_folder_path_list.append(sdf_folder_path)
            self.split_folder_path_list.append(split_folder_path)

        self.paths_list = []

        for i in range(len(self.sdf_folder_path_list)):
            split_folder_path = self.split_folder_path_list[i]
            sdf_folder_path = self.sdf_folder_path_list[i]

            dataset_name_list = os.listdir(split_folder_path)

            for dataset_name in dataset_name_list:
                sdf_split_folder_path = split_folder_path + dataset_name + "/"

                categories = os.listdir(sdf_split_folder_path)
                # FIXME: for detect test only
                if self.split == "test":
                    # categories = ["02691156"]
                    categories = ["03001627"]

                for j, category in enumerate(categories):
                    modelid_list_file_path = (
                        sdf_split_folder_path + category + "/" + self.split + ".txt"
                    )
                    if not os.path.exists(modelid_list_file_path):
                        continue

                    with open(modelid_list_file_path, "r") as f:
                        modelid_list = f.read().split()

                    print("[INFO][SDFDataset::__init__]")
                    print(
                        "\t start load dataset: "
                        + dataset_name
                        + "["
                        + category
                        + "], "
                        + str(j + 1)
                        + "/"
                        + str(len(categories))
                        + "..."
                    )
                    for modelid in tqdm(modelid_list):
                        mash_file_path = (
                            self.mash_folder_path
                            + dataset_name
                            + "/"
                            + category
                            + "/"
                            + modelid
                            + ".npy"
                        )

                        sdf_file_path = (
                            sdf_folder_path
                            + dataset_name
                            + "/"
                            + category
                            + "/"
                            + modelid
                            + ".npy"
                        )

                        self.paths_list.append([mash_file_path, sdf_file_path])
        return

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_path, sdf_file_path = self.paths_list[index]

        mash_params = np.load(mash_file_path, allow_pickle=True).item()
        sdf_data = np.load(sdf_file_path)

        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        scale_range = [0.5, 2.0]
        move_range = [-0.6, 0.6]

        random_scale = (
            scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.rand()
        )
        random_translate = move_range[0] + (
            move_range[1] - move_range[0]
        ) * np.random.rand(3)

        mash_params = np.hstack(
            [
                rotate_vectors,
                positions * random_scale + random_translate,
                mask_params,
                sh_params * random_scale,
            ]
        )

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

        qry = points[perm] * random_scale + random_translate
        occ = occ[perm]
        mash_params = mash_params[np.random.permutation(mash_params.shape[0])]

        feed_dict = {
            "qry": torch.tensor(qry).float(),
            "mash_params": torch.tensor(mash_params).float(),
            "occ": torch.tensor(occ).float(),
        }

        return feed_dict
