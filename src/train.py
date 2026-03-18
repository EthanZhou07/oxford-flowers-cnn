import os
import time

import torch
from torch import nn, optim

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
    ROOT_DIR,
    SEED,
    WEIGHT_DECAY,
)
from dataset import build_dataloaders, download_dataset
from model import SimpleCNN
from utils import (
    evaluate,
    plot_training_curves,
    set_seed,
    show_random_predictions,
    train_one_epoch,
)


def main():
    set_seed(SEED)
    download_dataset(ROOT_DIR)

    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

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
