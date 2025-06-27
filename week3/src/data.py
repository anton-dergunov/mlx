import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from torch.utils.data import random_split
from PIL import Image
import hashlib
import pickle
from tqdm import tqdm
import numpy as np
import random
import os


# TODO:
# 3. don't hardcode size 28.
# 4. Replace transforms.Normalize((0.1307,), (0.3081,)) with the real mean and std for the dataset.


PAD_TOKEN = 10
START_TOKEN = 11
END_TOKEN = 12


def patchify(img, patch_size=14):
    """
    Converts an image into flatenned patches.
    Args:
        img (Tensor):     shape (1, H, W);                        example (1, 28, 28)
        patch_size (int): width and height of each patch;         example 14
    Returns:
        Tensor: (num_patches, patch_dim) flattened patches image; example (4, 196)
    """
    # img: torch.Tensor with shape (1, H, W)
    patches = img.unfold(1, patch_size, patch_size) \
                 .unfold(2, patch_size, patch_size)
    # patches shape: (1, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size * patch_size)
    return patches  # shape: (num_patches, patch_dim)


def unpatchify(patches, patch_size=14):
    """
    Reconstruct an image from flattened patches.
    Args:
        patches (Tensor): shape (num_patches, patch_dim); example (4, 196)
        patch_size (int): width and height of each patch; example 14
    Returns:
        Tensor: (1, H, W) reconstructed image;            example (1, 28, 28)
    """
    num_patches, patch_dim = patches.shape
    assert patch_dim == patch_size ** 2, f"Patch dim mismatch: {patch_dim} != {patch_size}^2"

    # Infer grid size
    grid_size = int(np.sqrt(num_patches))
    assert grid_size ** 2 == num_patches, "Number of patches must be a perfect square"

    # (num_patches, patch_size, patch_size)
    patches = patches.view(grid_size, grid_size, patch_size, patch_size)

    # Reconstruct rows
    rows = [torch.cat([patches[i, j] for j in range(grid_size)], dim=1) for i in range(grid_size)]

    # Combine all rows
    full_img = torch.cat(rows, dim=0)  # (H, W)
    return full_img.unsqueeze(0)  # (1, H, W)


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


class CompositePatchifiedMNIST(Dataset):
    def __init__(self, root, train=True, download=True, transform=None,
                 canvas_size=(56, 56), grid_rows=2, grid_cols=2, patch_size=14,
                 num_digits=4, placement='grid', num_images=10000,
                 num_digits_range=None, cache=True, cache_dir=None):

        self.canvas_w, self.canvas_h = canvas_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.patch_size = patch_size
        self.num_digits = num_digits
        self.num_digits_range = num_digits_range
        self.placement = placement
        self.num_images = num_images
        self.transform = transform if transform else transforms.ToTensor()
        self.cache = cache

        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())

        # Generate a hash to identify the cache
        self.cache_dir = cache_dir or os.path.join(root, 'MNIST')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_key = self._make_cache_key(canvas_size, patch_size, num_digits, num_digits_range,
                                              placement, num_images, train)
        self.cache_file = os.path.join(self.cache_dir, self.cache_key + '.pkl')

        if cache and os.path.exists(self.cache_file):
            print(f"Loading cached dataset: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print(f"Generating composite dataset: {self.cache_file}")
            self.samples = self._precompute()
            if cache:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.samples, f)

    def _make_cache_key(self, *args):
        hash_input = str(args).encode()
        return hashlib.md5(hash_input).hexdigest()

    def _precompute(self):
        from torchvision.transforms import functional as TF

        samples = []
        for _ in tqdm(range(self.num_images), desc="Generating composite MNIST"):
            img = Image.new('L', (self.canvas_w, self.canvas_h), color=0)
            labels = []

            num_digits = self.num_digits
            if self.num_digits_range:
                num_digits = random.randint(*self.num_digits_range)

            if self.placement == 'grid':
                patch_w, patch_h = self.canvas_w // self.grid_cols, self.canvas_h // self.grid_rows

                for i in range(num_digits):
                    idx = random.randint(0, len(self.mnist) - 1)
                    digit, label = self.mnist[idx]
                    digit = transforms.Resize((patch_h, patch_w))(digit)
                    row, col = divmod(i, self.grid_cols)
                    x = col * patch_w
                    y = row * patch_h
                    img.paste(transforms.ToPILImage()(digit), (x, y))
                    labels.append(label)

            elif self.placement == 'random':
                total_cells = self.grid_rows * self.grid_cols

                # Randomly choose cells where digits will go
                chosen_cells = random.sample(range(total_cells), num_digits)
                chosen_cells.sort()

                for i, cell_index in enumerate(chosen_cells):
                    idx = random.randint(0, len(self.mnist) - 1)
                    digit_tensor, label = self.mnist[idx]
                    digit_img = transforms.ToPILImage()(digit_tensor)

                    # === Augment ===

                    # Random rotation (normal distribution, std 30 degrees)
                    angle = random.gauss(0, 30)
                    digit_img = TF.rotate(digit_img, angle, fill=0)

                    # Random scale (normal distribution, std 1.3, mean 1)
                    scale_factor = max(0.5, random.gauss(1.0, 0.3))
                    new_size = max(10, int(28 * scale_factor))
                    digit_img = TF.resize(digit_img, (new_size, new_size))

                    # Random affine stretch/shear
                    shear_x = random.uniform(-10, 10)
                    shear_y = random.uniform(-10, 10)
                    digit_img = TF.affine(digit_img, angle=0, translate=(0, 0), scale=1.0, shear=[shear_x, shear_y], fill=0)

                    # Random intensity bump: brighten non-black pixels
                    if random.random() < 0.8:
                        arr = np.array(digit_img, dtype=np.float32)
                        arr[arr > 0] *= random.uniform(1.1, 1.5)
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                        digit_img = Image.fromarray(arr)

                    # === Placement ===

                    row = cell_index // self.grid_cols
                    col = cell_index % self.grid_cols

                    cell_w = self.canvas_w // self.grid_cols
                    cell_h = self.canvas_h // self.grid_rows
                    base_x = col * cell_w
                    base_y = row * cell_h

                    # Small extra jitter inside cell
                    max_x_jitter = max(cell_w - digit_img.size[0], 0)
                    max_y_jitter = max(cell_h - digit_img.size[1], 0)
                    jitter_x = random.randint(0, max_x_jitter) if max_x_jitter > 0 else 0
                    jitter_y = random.randint(0, max_y_jitter) if max_y_jitter > 0 else 0

                    x = base_x + jitter_x
                    y = base_y + jitter_y

                    img.paste(digit_img, (x, y))
                    labels.append(label)

            else:
                raise ValueError(f"Unknown placement mode: {self.placement}")

            tensor_img = self.transform(img)
            patches = patchify(tensor_img, self.patch_size)
            samples.append((patches, torch.tensor(labels)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NumberScribblesDataset(Dataset):
    """
    Custom dataset for hand-drawn number grids saved as PNGs.
    Filenames encode the ground truth sequence with underscores for blanks.
    """
    def __init__(self, source_dir, patch_size=14, transform=None):
        self.source_dir = source_dir
        self.patch_size = patch_size
        self.transform = transform if transform else transforms.ToTensor()
        self.files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img_path = os.path.join(self.source_dir, file)
        img = Image.open(img_path).convert('L')  # (H, W)

        tensor_img = self.transform(img).unsqueeze(0) if img.mode != 'L' else self.transform(img)
        patches = patchify(tensor_img, self.patch_size)

        # Parse filename: remove underscores, extract digits
        base = os.path.splitext(file)[0]
        label_str = base.replace('_', '')
        label = torch.tensor([int(ch) for ch in label_str], dtype=torch.long)

        return patches, label


def padded_collate_fn(batch):
    inputs, targets = zip(*batch)  # list of (input, target)

    # Collate input tensors (images or patchified inputs)
    inputs = default_collate(inputs)

    # Convert all targets to tensors
    targets = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in targets]

    # If all targets are scalar (single value), stack them
    if all(t.ndim == 0 for t in targets):
        targets = torch.stack(targets)  # Shape: (B,)
    else:
        # Otherwise treat them as sequences. Include <START> and <END> tokens and pad
        processed_targets = []
        for seq in targets:
            seq_tensor = torch.tensor([START_TOKEN] + list(seq) + [END_TOKEN], dtype=torch.long)
            processed_targets.append(seq_tensor)

        targets = pad_sequence(processed_targets, batch_first=True, padding_value=PAD_TOKEN)  # Shape: (B, T)

    return inputs, targets


def load_mnist_dataloaders(cache_dir, dataset_name=None, source_dir=None, batch_size=64, valid_fraction=0.2, patch_size=14,
                           seed=42, num_workers=2, composite_mode=False,
                           canvas_size=(56, 56), grid_rows=2, grid_cols=2, num_digits=4, placement='grid',
                           num_digits_range=None, num_images=10000, num_images_test=2000):

    # TODO Do not hardcode these parameters
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_name and dataset_name == 'number-scribbles':
        full_dataset = NumberScribblesDataset(
            source_dir=source_dir,
            patch_size=patch_size,
            transform=transform
        )
        # TODO Use a different dataset for testing
        test_dataset = NumberScribblesDataset(
            source_dir=source_dir,
            patch_size=patch_size,
            transform=transform
        )
    else:
        if composite_mode:
            full_dataset = CompositePatchifiedMNIST(
                root=cache_dir,
                train=True,
                download=True,
                transform=transform,
                canvas_size=canvas_size,
                grid_rows = grid_rows,
                grid_cols = grid_cols,
                num_digits=num_digits,
                placement=placement,
                num_digits_range=num_digits_range,
                num_images=num_images
            )
            test_dataset = CompositePatchifiedMNIST(
                root=cache_dir,
                train=False,
                download=True,
                transform=transform,
                canvas_size=canvas_size,
                grid_rows = grid_rows,
                grid_cols = grid_cols,
                num_digits=num_digits,
                placement=placement,
                num_digits_range=num_digits_range,
                num_images=num_images_test
            )
        else:
            full_dataset = PatchifiedMNIST(root=cache_dir, train=True, download=True, patch_size=patch_size, transform=transform)
            test_dataset = PatchifiedMNIST(root=cache_dir, train=False, download=True, patch_size=patch_size, transform=transform)

    valid_size = int(valid_fraction * len(full_dataset))
    train_size = len(full_dataset) - valid_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, valid_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=padded_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=padded_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=padded_collate_fn)

    return train_loader, val_loader, test_loader
