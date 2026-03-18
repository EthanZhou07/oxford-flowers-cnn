import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
