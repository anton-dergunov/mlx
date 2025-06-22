from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(cfg):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers, shuffle=True)
