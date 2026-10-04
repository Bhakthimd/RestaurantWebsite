"""
Model definition and Grad-CAM utilities for food classification.

Defines the EfficientNet-B0 architecture with a custom classification head:
  Dropout(p=0.3) + Linear(1280, num_classes)
Loads class names directly from ckpt["class_names"] and provides inference functions.
"""

import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

from preprocess import get_eval_transforms, prepare_image


def build_model(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Constructs an EfficientNet-B0 model with custom classifier:
      Dropout(p=0.3) + Linear(1280, num_classes)
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_checkpoint(
    checkpoint_path: str = "EfficientNet_B0_best.pt",
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """
    Loads model weights and class_names directly from a checkpoint file.
    
    Returns:
        (model, class_names, meta_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found."
        )
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {e}")
        
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint format in '{checkpoint_path}'. Expected dict with 'model_state_dict'.")

    # Extract class names directly from checkpoint
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(f"No 'class_names' key found in checkpoint '{checkpoint_path}'.")

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    
    model.to(device)
    model.eval()
    return model, class_names, meta


def predict_image(
    model: nn.Module,
    image: Image.Image,
    class_names: List[str],
    device: Optional[torch.device] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Runs model inference on a single PIL Image.
    Applies EXIF transpose, thumbnail scaling (1024x1024), and exact Resize((224, 224)).
    
    Returns:
        Dict containing:
            - 'top_class': name of predicted top class
            - 'top_confidence': confidence float (0.0 to 1.0)
            - 'top_k': list of (class_name, confidence) sorted desc
            - 'all_probs': dict mapping class_name -> confidence
            - 'input_tensor': preprocessed input tensor [1, 3, 224, 224]
    """
    if device is None:
        device = next(model.parameters()).device
        
    # Pre-shrink and correct orientation
    prepared_img = prepare_image(image)
    
    transform = get_eval_transforms()
    input_tensor = transform(prepared_img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        
    probs_cpu = probs.cpu().numpy()
    
    sorted_indices = np.argsort(probs_cpu)[::-1]
    top_k_results = [
        (class_names[idx], float(probs_cpu[idx]))
        for idx in sorted_indices[:top_k]
    ]
    
    all_probs = {class_names[i]: float(probs_cpu[i]) for i in range(len(class_names))}
    
    return {
        "top_class": top_k_results[0][0],
        "top_confidence": top_k_results[0][1],
        "top_index": int(sorted_indices[0]),
        "top_k": top_k_results,
        "all_probs": all_probs,
        "input_tensor": input_tensor,
    }


class GradCAM:
    """
    Grad-CAM implementation for visual explanations of model predictions.
    Targets the final convolutional block of EfficientNet-B0 (model.features[-1]).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, inp, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes 2D Grad-CAM heatmap normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        
        tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(tensor)
        
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())
            
        score = logits[0, target_class_idx]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")
            
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam
        
    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def overlay_gradcam(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet"
) -> Image.Image:
    """
    Resizes the 2D CAM heatmap to match the image dimensions and blends it with the image.
    """
    img_rgb = prepare_image(image)
    w, h = img_rgb.size
    
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0
    
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap_name)
    colored_cam = cmap(cam_resized)[:, :, :3]
    
    img_arr = np.array(img_rgb, dtype=np.float32) / 255.0
    blended = alpha * colored_cam + (1 - alpha) * img_arr
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)
