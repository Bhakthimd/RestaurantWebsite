"""
Diagnostic script to verify the EfficientNet-B0 checkpoint on food20dataset/test_set.

Evaluates all 300 test images using:
- Model and classifier loaded from checkpoint
- Class names loaded from ckpt["class_names"]
- Preprocessing: Resize((224, 224)), ToTensor(), Normalize(...)
- ImageOps.exif_transpose(img)
Prints:
- Overall accuracy
- Per-class accuracy
- Top 10 most confident wrong predictions
"""

import os
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    checkpoint_path = "EfficientNet_B0_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' does not exist.")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        print("Error: Checkpoint does not contain 'model_state_dict'.")
        return

    # Load class names from checkpoint
    class_names = ckpt.get("class_names")
    print(f"Loaded class names from checkpoint ({len(class_names)} classes):")
    print(class_names)
    
    if "best_val_loss" in ckpt:
        print(f"Checkpoint best_val_loss: {ckpt['best_val_loss']}")

    # Build model exactly as specified
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(class_names))
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocessing exactly as specified:
    # Resize((224, 224)), ToTensor(), Normalize(...)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: ImageOps.exif_transpose(img)),
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = "food20dataset/test_set"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        return

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    total_samples = 0
    total_correct = 0
    per_class_stats = {name: {"correct": 0, "total": 0} for name in class_names}
    wrong_predictions = []

    test_folders = sorted(os.listdir(test_dir))
    for folder in test_folders:
        folder_path = os.path.join(test_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        if folder not in class_to_idx:
            print(f"Warning: Folder '{folder}' not found in ckpt['class_names']. Skipping.")
            continue
            
        true_idx = class_to_idx[folder]
        
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            img_path = os.path.join(folder_path, fname)
            try:
                img = Image.open(img_path)
                tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    
                conf, pred_idx = torch.max(probs, dim=0)
                conf = conf.item()
                pred_idx = pred_idx.item()
                pred_class = class_names[pred_idx]
                
                total_samples += 1
                per_class_stats[folder]["total"] += 1
                
                if pred_idx == true_idx:
                    total_correct += 1
                    per_class_stats[folder]["correct"] += 1
                else:
                    wrong_predictions.append({
                        "file_name": os.path.join(folder, fname),
                        "true_class": folder,
                        "pred_class": pred_class,
                        "confidence": conf
                    })
            except Exception as e:
                print(f"Error processing image '{img_path}': {e}")

    overall_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"OVERALL ACCURACY: {overall_acc:.2f}% ({total_correct}/{total_samples})")
    print("=" * 60)

    print("\nPER-CLASS ACCURACY:")
    print("-" * 45)
    print(f"{'Class Name':<20} {'Accuracy':<10} {'Fraction':<15}")
    print("-" * 45)
    for name in class_names:
        stats = per_class_stats[name]
        c = stats["correct"]
        t = stats["total"]
        acc = (c / t) * 100 if t > 0 else 0.0
        print(f"{name:<20} {acc:>6.2f}%     ({c}/{t})")
    print("-" * 45)

    # Sort wrong predictions by confidence descending
    wrong_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_10_wrong = wrong_predictions[:10]

    print(f"\nTOP {len(top_10_wrong)} MOST CONFIDENT WRONG PREDICTIONS:")
    print("-" * 75)
    print(f"{'File Name':<30} {'True Class':<15} {'Predicted':<15} {'Confidence':<10}")
    print("-" * 75)
    for w in top_10_wrong:
        print(f"{w['file_name']:<30} {w['true_class']:<15} {w['pred_class']:<15} {w['confidence']*100:>6.2f}%")
    print("-" * 75)

if __name__ == "__main__":
    check()
