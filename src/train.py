import os
import random
import time
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from PIL import Image, ImageFile
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Allow loading slightly truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


ROOT_DIR = "flower_data"
NUM_CLASSES = 102
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
SEED = 42
BEST_MODEL_PATH = "best_flower_cnn.pth"


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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            # For older Python versions that do not support the filter argument
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


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch in loader:
        if batch is None:
            continue

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("All training batches were empty. Please check dataset files.")

    return running_loss / total_samples, running_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch in loader:
        if batch is None:
            continue

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("All evaluation batches were empty. Please check dataset files.")

    return running_loss / total_samples, running_correct / total_samples


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


@torch.no_grad()
def show_random_predictions(model, test_loader, device, num_images=6):
    model.eval()

    valid_batch = None
    for batch in test_loader:
        if batch is not None:
            valid_batch = batch
            break

    if valid_batch is None:
        print("No valid test images available to display.")
        return

    images, labels = valid_batch
    num_images = min(num_images, len(images))
    indices = random.sample(range(len(images)), num_images)

    rows = (num_images + 2) // 3
    plt.figure(figsize=(12, 4 * rows))

    for i, idx in enumerate(indices):
        img = images[idx].unsqueeze(0).to(device)
        label = labels[idx].item()
        outputs = model(img)
        pred = outputs.argmax(dim=1).item()

        img_display = denormalize(images[idx])
        plt.subplot(rows, 3, i + 1)
        plt.imshow(img_display)
        plt.axis("off")
        color = "green" if pred == label else "red"
        plt.title(f"True: {label}\nPred: {pred}", color=color)

    plt.tight_layout()
    plt.show()


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


def main():
    set_seed(SEED)
    download_dataset(ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(ROOT_DIR, BATCH_SIZE)

    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved to {BEST_MODEL_PATH}")

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    print("\nRandom test predictions:")
    show_random_predictions(model, test_loader, device)


if __name__ == "__main__":
    main()
