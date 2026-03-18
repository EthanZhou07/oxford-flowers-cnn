import os
import tarfile
import urllib.request

import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, ROOT_DIR, SEED

# Allow loading slightly truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def download_file(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"Already exists: {dst}")
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"Saved to: {dst}")


def safe_extract_tar(tar_path: str, dst_dir: str) -> None:
    print("Extracting image archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        try:
            tar.extractall(path=dst_dir, filter="data")
        except TypeError:
            tar.extractall(path=dst_dir)
    print("Extraction completed.")


def download_dataset(root_dir: str = ROOT_DIR) -> None:
    """Download and extract the Oxford 102 Flowers dataset if needed."""
    image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

    os.makedirs(root_dir, exist_ok=True)

    image_tar_path = os.path.join(root_dir, "102flowers.tgz")
    labels_path = os.path.join(root_dir, "imagelabels.mat")
    jpg_dir = os.path.join(root_dir, "jpg")

    download_file(image_url, image_tar_path)
    download_file(labels_url, labels_path)

    if not os.path.isdir(jpg_dir):
        safe_extract_tar(image_tar_path, root_dir)
    else:
        print(f"Images already extracted: {jpg_dir}")

    print("\nDataset is ready!")
    print(f"Root directory: {os.path.abspath(root_dir)}")


class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir: str, indices=None, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "jpg")
        self.transform = transform

        labels_mat = scipy.io.loadmat(os.path.join(root_dir, "imagelabels.mat"))
        all_labels = labels_mat["labels"][0] - 1  # convert to 0..101

        if indices is None:
            indices = np.arange(len(all_labels))

        self.indices = np.array(indices, dtype=int)
        self.labels = all_labels[self.indices]

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        real_idx = int(self.indices[item])
        img_name = f"image_{real_idx + 1:05d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.labels[item])

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(f"[Warning] Failed to load {img_name}: {e}")
            return None

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def safe_collate(batch):
    """Drop unreadable samples instead of letting DataLoader crash."""
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def split_indices(n_samples: int, train_ratio=0.7, val_ratio=0.15, seed: int = SEED):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def build_dataloaders(root_dir: str = ROOT_DIR, batch_size: int = BATCH_SIZE):
    labels_mat = scipy.io.loadmat(os.path.join(root_dir, "imagelabels.mat"))
    n_samples = len(labels_mat["labels"][0])
    train_idx, val_idx, test_idx = split_indices(n_samples)

    train_dataset = OxfordFlowersDataset(root_dir=root_dir, indices=train_idx, transform=train_transform)
    val_dataset = OxfordFlowersDataset(root_dir=root_dir, indices=val_idx, transform=eval_transform)
    test_dataset = OxfordFlowersDataset(root_dir=root_dir, indices=test_idx, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
