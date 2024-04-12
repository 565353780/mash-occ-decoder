import os
import numpy as np
from math import ceil
from typing import Tuple


class Convertor(object):
    def __init__(self, dataset_root_folder_path: str) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        return

    def toRelFilePathList(self) -> list:
        rel_file_path_list = []
        print("[INFO][Convertor::toRelFilePathList]")
        print("\t start search npy files...")
        print("\t dataset_root_folder_path:", self.dataset_root_folder_path)
        for root, _, files in os.walk(self.dataset_root_folder_path):
            rel_folder_path = root.split(self.dataset_root_folder_path)[1] + "/"

            for file_name in files:
                if file_name[-4:] != ".npy":
                    continue

                rel_file_path = rel_folder_path + file_name
                rel_file_path_list.append(rel_file_path)

        return rel_file_path_list

    def convertToSplits(
        self,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> Tuple[list, list, list]:
        if not os.path.exists(self.dataset_root_folder_path):
            print("[ERROR][Convertor::convertToSplits]")
            print("\t dataset root folder not exist!")
            print("\t dataset_root_folder_path:", self.dataset_root_folder_path)
            return [], [], []

        rel_file_path_list = self.toRelFilePathList()

        permut_rel_file_path_list = np.random.permutation(rel_file_path_list)

        rel_file_path_num = len(rel_file_path_list)

        train_split_num = ceil(train_scale * rel_file_path_num)
        val_split_num = ceil(val_scale * rel_file_path_num)

        train_split = permut_rel_file_path_list[:train_split_num]
        val_split = permut_rel_file_path_list[
            train_split_num : train_split_num + val_split_num
        ]
        test_split = permut_rel_file_path_list[train_split_num + val_split_num :]

        return train_split.tolist(), val_split.tolist(), test_split.tolist()

    def convertToSplitFiles(
        self,
        save_split_folder_path: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        train_split, val_split, test_split = self.convertToSplits(
            train_scale, val_scale
        )

        if len(train_split) + len(val_split) + len(test_split) == 0:
            print("[ERROR][Convertor::convertToSplitFiles]")
            print("\t convertToSplits failed!")
            return False

        os.makedirs(save_split_folder_path, exist_ok=True)

        with open(save_split_folder_path + "train.txt", "w") as f:
            for train_name in train_split:
                f.write(train_name + "\n")

        with open(save_split_folder_path + "val.txt", "w") as f:
            for val_name in val_split:
                f.write(val_name + "\n")

        with open(save_split_folder_path + "test.txt", "w") as f:
            for test_name in test_split:
                f.write(test_name + "\n")
        return True
