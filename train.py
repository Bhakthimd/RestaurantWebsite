"""
Training script for Food Image Classification with EfficientNet-B0.

Features:
- Checks if checkpoint already exists to skip redundant training
- 80/20 stratified train/validation split from food20dataset/train_set
- Phase 1: Freeze backbone, train classifier head (5 epochs)
- Phase 2: Unfreeze backbone, fine-tune with lower learning rate (15 epochs)
- AdamW optimizer, label smoothing 0.1, weight decay, early stopping on validation loss
- Final evaluation on the held-out test_set (300 images) with accuracy, precision, recall, F1, confusion matrix
- Saves best checkpoint and test results cache
"""

import os
import sys
import json
import time
import argparse
from typing import List, Tuple, Dict, Any
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from preprocess import (
    get_class_names,
    get_train_transforms,
    get_eval_transforms,
)
from model import build_model


class FoodImageDataset(Dataset):
    """
    Dataset for loading food images from filepaths with labels and transforms.
    Handles potentially corrupted images safely.
    """
    def __init__(self, filepaths: List[str], labels: List[int], transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as e:
            # Fallback to black image if corrupted
            print(f"Warning: Corrupted image at {path}, error: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def collect_dataset_samples(dataset_dir: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    """
    Collects image filepaths and corresponding integer class labels.
    """
    filepaths = []
    labels = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_folder):
            print(f"Warning: Missing class folder {class_folder}")
            continue

        for fname in sorted(os.listdir(class_folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                filepaths.append(os.path.join(class_folder, fname))
                labels.append(class_idx)

    return filepaths, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Runs one training epoch. Returns (avg_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets).item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates model on validation dataloader. Returns (avg_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += targets.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def evaluate_test_set(
    model: nn.Module,
    test_dir: str = "food20dataset/test_set",
    class_names: List[str] = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Evaluates model on held-out test_set (300 images).
    Returns comprehensive metrics including accuracy, macro precision/recall/F1,
    confusion matrix, and classification report dict.
    """
    if class_names is None:
        class_names = get_class_names()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_files, test_labels = collect_dataset_samples(test_dir, class_names)
    if not test_files:
        raise ValueError(f"No test images found in {test_dir}")

    eval_transform = get_eval_transforms()
    test_dataset = FoodImageDataset(test_files, test_labels, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    results = {
        "accuracy": acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "class_names": class_names,
        "total_test_samples": len(test_files),
    }

    return results


def train_model(
    train_dir: str = "food20dataset/train_set",
    test_dir: str = "food20dataset/test_set",
    checkpoint_path: str = "EfficientNet_B0_best.pt",
    results_cache_path: str = "test_results.json",
    phase1_epochs: int = 5,
    phase2_epochs: int = 15,
    batch_size: int = 16,
    patience: int = 4,
    force_train: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Full training pipeline following specified requirements:
      - Stratified 80/20 train/validation split
      - Phase 1: Train classifier head (backbone frozen)
      - Phase 2: Fine-tune backbone and head (unfrozen) with lower LR
      - Label smoothing 0.1, AdamW, weight decay, early stopping on val loss
      - Final evaluation on test_set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for existing checkpoint
    if os.path.exists(checkpoint_path) and not force_train:
        print(f"Checkpoint '{checkpoint_path}' already exists. Loading checkpoint...")
        class_names = get_class_names(train_dir)
        model = build_model(num_classes=len(class_names), pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        
        # Load or generate test evaluation results
        if os.path.exists(results_cache_path):
            with open(results_cache_path, "r") as f:
                test_results = json.load(f)
        else:
            print("Running test evaluation on held-out test set...")
            test_results = evaluate_test_set(model, test_dir, class_names, device)
            with open(results_cache_path, "w") as f:
                json.dump(test_results, f, indent=2)
                
        return model, test_results

    # 1. Dataset & Stratified Split
    class_names = get_class_names(train_dir)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    train_files, train_labels = collect_dataset_samples(train_dir, class_names)
    print(f"Total training images found: {len(train_files)}")

    # 80/20 Stratified train/val split
    train_paths, val_paths, y_train, y_val = train_test_split(
        train_files,
        train_labels,
        test_size=0.20,
        stratify=train_labels,
        random_state=42
    )
    print(f"Train split: {len(train_paths)} samples, Validation split: {len(val_paths)} samples")

    train_tf = get_train_transforms()
    eval_tf = get_eval_transforms()

    train_dataset = FoodImageDataset(train_paths, y_train, transform=train_tf)
    val_dataset = FoodImageDataset(val_paths, y_val, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 2. Build Model
    model = build_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    # Criterion with label smoothing 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = None
    epochs_no_improve = 0

    total_start_time = time.time()

    # ==========================================
    # Phase 1: Freeze backbone, train head only
    # ==========================================
    print("\n" + "=" * 50)
    print("PHASE 1: Training classifier head (Backbone frozen)")
    print("=" * 50)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_phase1 = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_phase1, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Phase 1 - Epoch [{epoch}/{phase1_epochs}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> Best checkpoint updated (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc * 100:.2f}%)")

    # ==========================================
    # Phase 2: Unfreeze backbone & fine-tune
    # ==========================================
    print("\n" + "=" * 50)
    print("PHASE 2: Fine-tuning entire network (Backbone unfrozen)")
    print("=" * 50)

    # Restore best weights from Phase 1 before fine-tuning
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    for param in model.parameters():
        param.requires_grad = True

    # Differential learning rate: lower for backbone features, slightly higher for head
    optimizer_phase2 = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ],
        weight_decay=1e-2
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=phase2_epochs, eta_min=1e-6
    )

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_phase2, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Phase 2 - Epoch [{epoch}/{phase2_epochs}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  --> Best checkpoint updated (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc * 100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without validation loss improvement.")
                break

    # Save best checkpoint
    print(f"\nSaving best checkpoint to '{checkpoint_path}'...")
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "model_architecture": "EfficientNet-B0",
            "classifier": "Dropout(p=0.3) + Linear(1280, 10)",
        },
        checkpoint_path,
    )
    print(f"Checkpoint successfully saved! Best Val Acc: {best_val_acc * 100:.2f}%, Best Val Loss: {best_val_loss:.4f}")

    # ==========================================
    # Held-out Test Set Evaluation (Once at end)
    # ==========================================
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON HELD-OUT TEST SET (300 images)")
    print("=" * 50)
    test_results = evaluate_test_set(model, test_dir, class_names, device)

    print(f"Test Accuracy:        {test_results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:      {test_results['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:         {test_results['macro_recall'] * 100:.2f}%")
    print(f"Macro F1 Score:       {test_results['macro_f1'] * 100:.2f}%")

    with open(results_cache_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved evaluation cache to '{results_cache_path}'.")

    total_time = time.time() - total_start_time
    print(f"\nTraining pipeline completed in {total_time / 60:.2f} minutes.")
    return model, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Food20 Dataset")
    parser.add_argument("--force", action="store_true", help="Force retraining even if checkpoint exists")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Number of head training epochs")
    parser.add_argument("--phase2-epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=4, help="Patience for early stopping")
    args = parser.parse_args()

    train_model(
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        force_train=args.force,
    )
