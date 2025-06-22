import torch
from torch.utils.data import TensorDataset, random_split

def get_datasets(split_ratio=0.8, input_dim=10, dataset_size=500):
    torch.manual_seed(42)
    X = torch.randn(dataset_size, input_dim)
    y = 2 * X[:, :1] + 0.1 * torch.randn(dataset_size, 1)  # linear target with noise

    dataset = TensorDataset(X, y)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
