import torch
import wget
from torch.utils.data import Dataset
import numpy as np
import os

class CorruptMnist(Dataset):
    def __init__(self, train):
        self.download_data(train)
        if train:
            content = [ ]
            for i in range(5):
                content.append(np.load(f"train_{i}.npz", allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load("test.npz", allow_pickle=True)
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
            
        self.data = data
        self.targets = targets
    
    def download_data(self, train):
        files = os.listdir()
        if train:
            for file_idx in range(5):
                if f'train_{file_idx}.npy' not in files:
                    wget.download(f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz")
        else:
            if "test.npy" not in files:    
                wget.download("https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz")
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]


if __name__ == "__main__":
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)
''' from typing import Union, Any, List

import numpy
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST



#def mnist(path: str = None):
#    train = MNIST("./temp/", train=True, download=True)
#    test = MNIST("./temp/", train=False, download=True)
#    return train, test


def _load_and_concat_all_train_data_files(path):
    train_images = []
    train_labels = []
    for i in range(5):
        train = np.load(path + f"train_{i}.npz")
        train_images.append(train["images"])
        train_labels.append(train["labels"])

    return {"images": np.array(train_images).reshape(-1, 28, 28),
            "labels": np.array(train_labels).reshape(-1, )}


def mnist(path: str = None):
    path = "C:/Users/Lenovo/Documents/dtu_mlops/dtu_mlops/s1_development_environment/exercise_files/final_exercise/data/corruptmnist/" if path is None else path
    train_temp = _load_and_concat_all_train_data_files(path)
    test_temp = np.load(path + "test.npz")
    
    train_x, train_y = torch.Tensor(train_temp["images"]), torch.Tensor(train_temp["labels"])
    test_x, test_y = torch.Tensor(test_temp["images"]), torch.Tensor(test_temp["labels"])
    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    return train, test


def convert_mnist_to_tensor_dataset(train, test):
    train_x, train_y = torch.Tensor(train["images"]), torch.Tensor(train["labels"])
    test_x, test_y = torch.Tensor(test["images"]), torch.Tensor(test["labels"])
    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    return train, test '''