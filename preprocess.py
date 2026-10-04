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
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
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
    """
    # Prepare image: EXIF orientation correction and RGB conversion
    prepared_img = prepare_image(image)
    orig_size = prepared_img.size  # (width, height)
    
    # Step b: Direct Resize to (224, 224) (exact match to model training)
    resized_224 = transforms.Resize((224, 224))(prepared_img)
    
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
    
    # Step e: Training augmentation examples
    aug_tf = get_augmentation_preview_transform()
    augmentations = [aug_tf(resized_224) for _ in range(num_aug_samples)]
    
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
    }
