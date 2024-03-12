import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.class_folders = sorted(os.listdir(data_folder))
        self.data = []
        self.labels = []

        for class_idx, class_folder in enumerate(self.class_folders):
            class_path = os.path.join(data_folder, class_folder)
            npy_files = [file for file in os.listdir(class_path) if file.endswith('.npy')]
            for npy_file in npy_files:
                file_path = os.path.join(class_path, npy_file)
                keypoints_data = np.load(file_path)
                min_values = keypoints_data.min(axis=(0, 1), keepdims=True)
                max_values = keypoints_data.max(axis=(0, 1), keepdims=True)

                # 将数据缩放到 [0, 1] 范围内
                keypoints_data = (keypoints_data - min_values) / (max_values - min_values)
                self.data.append(keypoints_data)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
