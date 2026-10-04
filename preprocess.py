"""
Preprocessing utilities for food image classification.

Handles dataset transforms, EXIF orientation correction,
standard evaluation transforms, and intermediate preprocessing step
extraction for UI visualization.
"""

import os
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_class_names(train_dir: str = "food20dataset/train_set") -> List[str]:
    """
    Retrieve sorted list of class names from dataset directory for training.
    In the app, class names are always loaded from ckpt['class_names'].
    """
    if not os.path.exists(train_dir):
        return []
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])


def prepare_image(img: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Applies EXIF transpose so phone photos are correctly oriented,
    shrinks very large images to max_size preserving aspect ratio,
    and converts to RGB.
    """
    img = ImageOps.exif_transpose(img)
    # thumbnail modifies in-place, preserving aspect ratio
    img_copy = img.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")
    return img_copy


def get_eval_transforms() -> transforms.Compose:
    """
    Validation, test, and inference transforms matching exact training protocol:
    Resize((224, 224)) -> ToTensor() -> Normalize(mean, std)
    (No random augmentation, no center crop).
    """
    return transforms.Compose([
        # exif_transpose is retained because prepare_image is not always called
        # before get_eval_transforms (e.g., in FoodImageDataset in train.py).
        transforms.Lambda(lambda img: ImageOps.exif_transpose(img)),
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transforms() -> transforms.Compose:
    """
    Training transforms with augmentation.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: ImageOps.exif_transpose(img)),
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_augmentation_preview_transform() -> transforms.Compose:
    """
    Augmentation transform that returns PIL Image for visual inspection.
    Matches get_train_transforms parameters.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    De-normalize an image tensor with shape (3, H, W) or (1, 3, H, W)
    back to [0, 1] RGB numpy array of shape (H, W, 3).
    """
    t = tensor.detach().cpu().clone()
    if t.ndim == 4:
        t = t.squeeze(0)
    
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    
    # Transpose to (H, W, 3)
    np_img = t.permute(1, 2, 0).numpy()
    return np_img


def extract_preprocessing_steps(image: Image.Image, num_aug_samples: int = 5) -> Dict[str, Any]:
    """
    Extract intermediate representations of the image for UI visualization:
      - 'original': PIL Image and size (W, H)
      - 'resized': PIL Image resized to (224, 224)
      - 'raw_tensor': torch.Tensor with shape [3, 224, 224] before normalization
      - 'tensor_shape': tuple of tensor shape
      - 'normalized_tensor': torch.Tensor after normalization
      - 'denormalized_image': numpy array (224, 224, 3)
      - 'channels': dict of 'R', 'G', 'B' numpy arrays (224, 224)
      - 'augmentations': list of augmented PIL Images
      - 'calculations': dictionary containing step-by-step mathematical calculations and statistics
    """
    # Prepare image: EXIF orientation correction and RGB conversion
    prepared_img = prepare_image(image)
    orig_w, orig_h = prepared_img.size
    orig_size = (orig_w, orig_h)
    orig_pixels = orig_w * orig_h
    orig_aspect_ratio = orig_w / orig_h if orig_h > 0 else 1.0
    orig_memory_bytes = orig_pixels * 3  # 3 channels (RGB) x 1 byte (uint8)
    
    # Step b: Direct Resize to (224, 224) (exact match to model training)
    resized_224 = transforms.Resize((224, 224))(prepared_img)
    resized_w, resized_h = resized_224.size
    resized_pixels = resized_w * resized_h  # 50,176 pixels
    scale_x = resized_w / orig_w if orig_w > 0 else 1.0
    scale_y = resized_h / orig_h if orig_h > 0 else 1.0
    pixel_change_pct = ((resized_pixels - orig_pixels) / orig_pixels) * 100.0 if orig_pixels > 0 else 0.0
    
    # Step c: Converted to tensor (values in [0, 1])
    to_tensor = transforms.ToTensor()(resized_224)
    tensor_shape = list(to_tensor.shape)  # [3, 224, 224]
    
    # Step d: Normalized with ImageNet mean/std
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    norm_tensor = normalize(to_tensor.clone())
    denorm_img = denormalize_tensor(norm_tensor)
    
    # Separate R, G, B channels
    np_channels = to_tensor.numpy()  # shape (3, 224, 224)
    r_channel = np_channels[0]
    g_channel = np_channels[1]
    b_channel = np_channels[2]

    # Sample pixel at center (112, 112) for concrete step-by-step arithmetic verification
    cx, cy = 112, 112
    raw_center_rgb = resized_224.getpixel((cx, cy))  # (r_uint8, g_uint8, b_uint8)
    
    tensor_center_r = float(to_tensor[0, cy, cx])
    tensor_center_g = float(to_tensor[1, cy, cx])
    tensor_center_b = float(to_tensor[2, cy, cx])
    
    norm_center_r = float(norm_tensor[0, cy, cx])
    norm_center_g = float(norm_tensor[1, cy, cx])
    norm_center_b = float(norm_tensor[2, cy, cx])
    
    recon_center_r = float(denorm_img[cy, cx, 0])
    recon_center_g = float(denorm_img[cy, cx, 1])
    recon_center_b = float(denorm_img[cy, cx, 2])

    # Per-channel raw stats [0.0, 1.0]
    raw_stats = {
        "R": {
            "min": float(to_tensor[0].min()),
            "max": float(to_tensor[0].max()),
            "mean": float(to_tensor[0].mean()),
            "std": float(to_tensor[0].std()),
        },
        "G": {
            "min": float(to_tensor[1].min()),
            "max": float(to_tensor[1].max()),
            "mean": float(to_tensor[1].mean()),
            "std": float(to_tensor[1].std()),
        },
        "B": {
            "min": float(to_tensor[2].min()),
            "max": float(to_tensor[2].max()),
            "mean": float(to_tensor[2].mean()),
            "std": float(to_tensor[2].std()),
        },
    }

    # Per-channel normalized stats
    norm_stats = {
        "R": {
            "min": float(norm_tensor[0].min()),
            "max": float(norm_tensor[0].max()),
            "mean": float(norm_tensor[0].mean()),
            "std": float(norm_tensor[0].std()),
        },
        "G": {
            "min": float(norm_tensor[1].min()),
            "max": float(norm_tensor[1].max()),
            "mean": float(norm_tensor[1].mean()),
            "std": float(norm_tensor[1].std()),
        },
        "B": {
            "min": float(norm_tensor[2].min()),
            "max": float(norm_tensor[2].max()),
            "mean": float(norm_tensor[2].mean()),
            "std": float(norm_tensor[2].std()),
        },
    }

    # Energy calculations across channels
    energy_r = float(r_channel.sum())
    energy_g = float(g_channel.sum())
    energy_b = float(b_channel.sum())
    total_energy = energy_r + energy_g + energy_b if (energy_r + energy_g + energy_b) > 0 else 1.0
    pct_r = (energy_r / total_energy) * 100.0
    pct_g = (energy_g / total_energy) * 100.0
    pct_b = (energy_b / total_energy) * 100.0

    # Reconstruction Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(to_tensor.permute(1, 2, 0).numpy() - denorm_img)))

    # Step e: Training augmentation examples
    aug_tf = get_augmentation_preview_transform()
    augmentations = [aug_tf(resized_224) for _ in range(num_aug_samples)]
    
    calculations = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "orig_pixels": orig_pixels,
        "orig_aspect_ratio": orig_aspect_ratio,
        "orig_memory_bytes": orig_memory_bytes,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "resized_pixels": resized_pixels,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "pixel_change_pct": pixel_change_pct,
        "center_coord": (cx, cy),
        "raw_center_rgb": raw_center_rgb,
        "tensor_center_rgb": (tensor_center_r, tensor_center_g, tensor_center_b),
        "norm_center_rgb": (norm_center_r, norm_center_g, norm_center_b),
        "recon_center_rgb": (recon_center_r, recon_center_g, recon_center_b),
        "raw_stats": raw_stats,
        "norm_stats": norm_stats,
        "energy_r": energy_r,
        "energy_g": energy_g,
        "energy_b": energy_b,
        "pct_r": pct_r,
        "pct_g": pct_g,
        "pct_b": pct_b,
        "mae": mae,
        "total_tensor_elements": 1 * 3 * 224 * 224,
        "tensor_memory_bytes": 1 * 3 * 224 * 224 * 4,  # float32 = 4 bytes
    }

    return {
        "original_image": prepared_img,
        "original_size": orig_size,
        "resized_image": resized_224,
        "resized_size": resized_224.size,
        "raw_tensor": to_tensor,
        "tensor_shape": tensor_shape,
        "normalized_tensor": norm_tensor,
        "denormalized_image": denorm_img,
        "channel_r": r_channel,
        "channel_g": g_channel,
        "channel_b": b_channel,
        "augmentations": augmentations,
        "calculations": calculations,
    }
