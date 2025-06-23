import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split


def patchify(img, patch_size=14):
    """
    Input:  (1, 28, 28) tensor (MNIST image)
    Output: (num_patches, patch_dim), here (4, 196)
    """
    # img: torch.Tensor with shape (1, 28, 28)
    patches = img.unfold(1, patch_size, patch_size) \
                 .unfold(2, patch_size, patch_size)
    # patches shape: (1, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size * patch_size)
    return patches  # shape: (num_patches=4, patch_dim=196)


def unpatchify(patches, patch_size=14):
    """
    Reconstruct a 28x28 image from (4, 196) flattened patches
    Output: (1, 28, 28) tensor
    """
    # (4, 196) -> (4, 14, 14)
    patches = patches.view(2, 2, patch_size, patch_size)
    
    # Rearrange into image: (2 * 14, 2 * 14) = (28, 28)
    top = torch.cat([patches[0,0], patches[0,1]], dim=1)
    bottom = torch.cat([patches[1,0], patches[1,1]], dim=1)
    full_img = torch.cat([top, bottom], dim=0)

    return full_img.unsqueeze(0)  # shape: (1, 28, 28)


class PatchifiedMNIST(Dataset):
    def __init__(self, root, train=True, download=True, patch_size=14, transform=None):
        self.base_dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform if transform else transforms.ToTensor()
        )
        self.patch_size = patch_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # img: (1, 28, 28)
        patches = patchify(img, self.patch_size)  # patches: (4, 196)
        return patches, label


def load_mnist_dataloaders(cache_dir, batch_size=64, valid_fraction=0.2, patch_size=14, seed=42, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # These are mean and standard deviation of the MNIST dataset computed over the training set:
        # 0.1307 ≈ mean pixel intensity of MNIST
        # 0.3081 ≈ standard deviation of pixel intensity
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Full training set
    full_dataset = PatchifiedMNIST(root=cache_dir, train=True, download=True, patch_size=patch_size, transform=transform)

    # Train/val split
    valid_size = int(valid_fraction * len(full_dataset))
    train_size = len(full_dataset) - valid_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, valid_size], generator=generator)

    # Test set
    test_dataset = PatchifiedMNIST(root=cache_dir, train=False, download=True, patch_size=patch_size, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# def get_train_valid_data_loader(cache_dir, valid_fraction=0.2):
#     full_dataset = PatchifiedMNIST(root=cache_dir, train=True, download=True, transform=transform)

#     valid_size = int(valid_fraction * len(full_dataset))
#     train_size = len(full_dataset) - valid_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, valid_size])

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     valid_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#     return train_loader, valid_loader


# def get_test_data_loader(cache_dir):
#     full_dataset = PatchifiedMNIST(root=cache_dir, train=False, download=True, transform=transform)
#     return DataLoader(full_dataset, batch_size=64, shuffle=True)
