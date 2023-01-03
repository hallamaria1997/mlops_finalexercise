from typing import Union, Any, List

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
    return train, test