"""
Streamlit Web Application for Food Image Classification.

Matches exact training configuration:
- Class names loaded directly from checkpoint (ckpt["class_names"])
- Preprocessing: ImageOps.exif_transpose -> thumbnail((1024, 1024)) -> Resize((224, 224)) -> ToTensor() -> Normalize
- Cached model loaded once via @st.cache_resource
- Inference under torch.no_grad() and model.eval()
- Real-time prediction, top-3 confidence bar chart, Grad-CAM attention overlay,
  and image processing steps breakdown.
"""

import os
import random
import json
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps

import streamlit as st
import torch

from preprocess import (
    prepare_image,
    extract_preprocessing_steps,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from model import load_checkpoint, predict_image, GradCAM, overlay_gradcam
from train import evaluate_test_set


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Indian Food Classifier | EfficientNet-B0",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, modern cards and badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .pred-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 5px solid #2563EB;
        padding: 1.1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .correct-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .wrong-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .info-tag {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Resources & Data
# ==============================================================================
@st.cache_resource(show_spinner="Loading EfficientNet-B0 model checkpoint...")
def get_cached_model(checkpoint_path: str = "EfficientNet_B0_best.pt") -> Tuple[Optional[torch.nn.Module], Optional[List[str]], Optional[str]]:
    """
    Loads and caches the model once for fast inference.
    Class names are retrieved directly from ckpt["class_names"].
    """
    if not os.path.exists(checkpoint_path):
        return None, None, f"Checkpoint '{checkpoint_path}' not found."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, class_names, _ = load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device
        )
        return model, class_names, None
    except Exception as e:
        return None, None, f"Error loading checkpoint '{checkpoint_path}': {e}"


@st.cache_data(show_spinner="Evaluating model on test dataset (300 images)...")
def get_cached_evaluation(checkpoint_path: str = "EfficientNet_B0_best.pt", cache_file: str = "test_results.json") -> Tuple[Optional[dict], Optional[str]]:
    """
    Loads test set evaluation results from disk or calculates them.
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data, None
        except Exception:
            pass

    if not os.path.exists(checkpoint_path):
        return None, f"Checkpoint '{checkpoint_path}' not found."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, class_names, _ = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
        results = evaluate_test_set(model, test_dir="food20dataset/test_set", class_names=class_names, device=device)
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
        return results, None
    except Exception as e:
        return None, f"Evaluation failed: {e}"


# ==============================================================================
# Helper Visualization Functions
# ==============================================================================
def plot_top_k_predictions(top_k: List[Tuple[str, float]]):
    """
    Renders horizontal bar chart of top-3 predictions.
    """
    categories = [item[0].title() for item in reversed(top_k)]
    probabilities = [item[1] * 100 for item in reversed(top_k)]

    fig, ax = plt.subplots(figsize=(6, 2.4))
    bars = ax.barh(categories, probabilities, color=["#93C5FD", "#60A5FA", "#2563EB"], height=0.55)
    
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1E293B",
        )

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm_data: list, class_names: list):
    """
    Renders styled confusion matrix heatmap.
    """
    cm_arr = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.title() for c in class_names],
        yticklabels=[c.title() for c in class_names],
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#E2E8F0",
    )
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.title("Confusion Matrix (Test Set - 300 Images)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


# ==============================================================================
# Page 1: Food Classifier
# ==============================================================================
def render_food_classifier_page(model: torch.nn.Module, class_names: List[str]):
    st.markdown('<div class="main-title">🍲 Food Image Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a food photo or select a test sample to classify with EfficientNet-B0 and inspect deep learning preprocessing steps.</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state keys
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "image_source" not in st.session_state:
        st.session_state.image_source = None
    if "image_name" not in st.session_state:
        st.session_state.image_name = None
    if "sample_true_class" not in st.session_state:
        st.session_state.sample_true_class = None
    if "last_upload_signature" not in st.session_state:
        st.session_state.last_upload_signature = None

    # Input controls section
    st.markdown("### 📥 Select or Upload Input Image")
    col_input1, col_input2 = st.columns([1, 1], gap="medium")

    with col_input1:
        st.markdown("**Option A: Upload Image**")
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            key="food_uploader",
        )
        if uploaded_file is not None:
            upload_sig = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_upload_signature != upload_sig:
                try:
                    raw_img = Image.open(uploaded_file)
                    # Apply EXIF transpose and shrink very large uploads with thumbnail((1024, 1024))
                    processed_img = prepare_image(raw_img, max_size=(1024, 1024))
                    st.session_state.current_image = processed_img
                    st.session_state.image_source = "upload"
                    st.session_state.image_name = uploaded_file.name
                    st.session_state.sample_true_class = None
                    st.session_state.last_upload_signature = upload_sig
                except Exception as e:
                    st.error(f"⚠️ Corrupt or unreadable image file: {e}")

    with col_input2:
        st.markdown("**Option B: Pick Sample from Test Dataset**")
        sample_class = st.selectbox(
            "Select Food Class:",
            options=class_names,
            format_func=lambda x: x.title(),
            key="sample_class_selector",
        )
        pick_btn = st.button("🎲 Pick Random Sample", width="stretch")

        if pick_btn:
            test_dir = os.path.join("food20dataset/test_set", sample_class)
            if not os.path.exists(test_dir):
                st.error(f"Folder not found: {test_dir}")
            else:
                valid_images = [
                    f for f in sorted(os.listdir(test_dir))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if not valid_images:
                    st.error(f"No images found in {test_dir}")
                else:
                    chosen_file = random.choice(valid_images)
                    sample_path = os.path.join(test_dir, chosen_file)
                    try:
                        raw_sample = Image.open(sample_path)
                        processed_sample = prepare_image(raw_sample, max_size=(1024, 1024))
                        st.session_state.current_image = processed_sample
                        st.session_state.image_source = "sample"
                        st.session_state.sample_true_class = sample_class
                        st.session_state.image_name = chosen_file
                        st.session_state.last_upload_signature = None
                    except Exception as e:
                        st.error(f"⚠️ Failed to open sample image '{sample_path}': {e}")

    st.markdown("---")

    active_img = st.session_state.current_image
    if active_img is None:
        st.info("👆 Please upload an image or click **'🎲 Pick Random Sample'** to begin classification.")
        return

    # Automatic inference under torch.no_grad() and model.eval()
    try:
        pred_results = predict_image(
            model=model,
            image=active_img,
            class_names=class_names,
            top_k=3
        )
    except Exception as e:
        st.error(f"⚠️ Inference error: {e}")
        return

    top_class = pred_results["top_class"]
    top_confidence = pred_results["top_confidence"]
    top_k = pred_results["top_k"]

    # ==========================================
    # Main Layout: Left = Image, Right = Results
    # ==========================================
    col_img, col_pred = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 🖼️ Input Image")
        source_label = (
            f"Sample ({st.session_state.sample_true_class.title()} - {st.session_state.image_name})"
            if st.session_state.image_source == "sample"
            else f"Uploaded ({st.session_state.image_name})"
        )
        st.image(active_img, caption=source_label, width="stretch")

        # Grad-CAM option
        show_gradcam = st.checkbox("🔥 Show Grad-CAM Attention Heatmap", value=False)
        if show_gradcam:
            try:
                cam_generator = GradCAM(model)
                cam = cam_generator.generate_heatmap(
                    pred_results["input_tensor"],
                    target_class_idx=pred_results["top_index"]
                )
                cam_generator.remove()
                cam_overlay = overlay_gradcam(active_img, cam, alpha=0.55, colormap_name="jet")
                st.image(
                    cam_overlay,
                    caption=f"Grad-CAM Heatmap overlay for predicted '{top_class.title()}'",
                    width="stretch"
                )
            except Exception as cam_err:
                st.warning(f"Grad-CAM could not be computed: {cam_err}")

    with col_pred:
        st.markdown("#### 🎯 Prediction Results")
        
        # Large and clear predicted food name
        st.markdown(
            f"""
            <div class="pred-box">
                <span style="font-size:0.85rem; color:#6B7280; font-weight:700; text-transform:uppercase;">Predicted Food</span>
                <div style="font-size:2.3rem; font-weight:800; color:#1E3A8A; line-height:1.2;">
                    {top_class.title()}
                </div>
                <div style="font-size:1.15rem; font-weight:600; color:#2563EB; margin-top:0.35rem;">
                    Confidence: {top_confidence * 100:.2f}%
                </div>
                <div class="info-tag">Note: Model was trained with label smoothing 0.1; the maximum possible confidence is about 91%, so 70-90% is normal for high certainty.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Ground truth comparison for sample images
        if st.session_state.image_source == "sample" and st.session_state.sample_true_class:
            true_cls = st.session_state.sample_true_class
            is_correct = (top_class.lower() == true_cls.lower())
            if is_correct:
                st.markdown(
                    f'<div class="correct-badge">✅ Correct Prediction! Ground truth: <strong>{true_cls.title()}</strong></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="wrong-badge">❌ Incorrect Prediction! Ground truth: <strong>{true_cls.title()}</strong></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='margin-bottom:0.75rem;'></div>", unsafe_allow_html=True)

        # Low confidence warning (< 50%)
        if top_confidence < 0.50:
            st.warning("⚠️ Low confidence, the image may not be one of the 10 classes.")

        # Top-3 predictions bar chart
        st.markdown("##### 📊 Top-3 Predictions")
        fig_bar = plot_top_k_predictions(top_k)
        st.pyplot(fig_bar, width="stretch")
        plt.close(fig_bar)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================================================================
    # Image Processing Steps Expander
    # ==============================================================================
    with st.expander("🔍 Show image processing steps & calculations", expanded=True):
        st.markdown("Detailed step-by-step image transformation stages applied during inference with exact mathematical calculations:")
        
        try:
            steps = extract_preprocessing_steps(active_img, num_aug_samples=5)
        except Exception as e:
            st.error(f"Error extracting image processing steps: {e}")
            return

        calc = steps.get("calculations", {})

        # ----------------------------------------------------------------------
        # Step 1: Input Dimensions & EXIF Orientation Correction
        # ----------------------------------------------------------------------
        st.markdown("### 1️⃣ Step 1: Input Acquisition & Dimension Preparation")
        st.markdown(
            "Phone camera orientation metadata (EXIF) is transposed to ensure the image is upright. "
            "If dimensions exceed 1024 px, proportional thumbnail downscaling is applied to preserve aspect ratio without distortion. "
            "The sizes shown below are after the 1024 px shrink, not necessarily the uploaded file size."
        )

        col_s1_img, col_s1_calc = st.columns([1, 1.3], gap="medium")
        with col_s1_img:
            st.image(steps["original_image"], caption=f"Prepared Image ({calc.get('orig_w', steps['original_size'][0])} × {calc.get('orig_h', steps['original_size'][1])} px)", width="stretch")
        
        with col_s1_calc:
            orig_w = calc.get("orig_w", steps["original_size"][0])
            orig_h = calc.get("orig_h", steps["original_size"][1])
            orig_ar = calc.get("orig_aspect_ratio", orig_w / orig_h if orig_h > 0 else 1.0)
            orig_pix = calc.get("orig_pixels", orig_w * orig_h)
            orig_mem = calc.get("orig_memory_bytes", orig_pix * 3)

            st.markdown("**Mathematical Calculations:**")
            st.latex(r"\text{Aspect Ratio (AR)} = \frac{W_{\text{orig}}}{H_{\text{orig}}} = \frac{" + f"{orig_w}" + r"}{" + f"{orig_h}" + r"} = " + f"{orig_ar:.3f} : 1")
            st.latex(r"N_{\text{orig}} = W_{\text{orig}} \times H_{\text{orig}} = " + f"{orig_w} \\times {orig_h} = {orig_pix:,}" + r"\text{ pixels}")
            st.latex(r"\text{Raw Memory (RGB uint8)} = \frac{N_{\text{orig}} \times 3 \text{ bytes}}{1024^2} = " + f"{orig_mem / (1024 * 1024):.2f}" + r"\text{ MB}")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Width × Height", f"{orig_w} × {orig_h} px")
            with m2:
                st.metric("Aspect Ratio", f"{orig_ar:.2f} : 1")
            with m3:
                st.metric("Total Pixels", f"{orig_pix:,}")

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Step 2: Spatial Resizing to Fixed Network Input (224 x 224)
        # ----------------------------------------------------------------------
        st.markdown("### 2️⃣ Step 2: Spatial Resizing to (224 × 224)")
        st.markdown(
            "EfficientNet-B0 requires a fixed input resolution of $224 \\times 224$ pixels. "
            "Bilinear interpolation uses distance-weighted averages of nearby pixels. "
            "The image is resized directly to 224x224 with no cropping, so non-square images are stretched (as in training)."
        )

        col_s2_img, col_s2_calc = st.columns([1, 1.3], gap="medium")
        with col_s2_img:
            st.image(steps["resized_image"], caption="Resized Image (224 × 224 px)", width="stretch")

        with col_s2_calc:
            scale_x = calc.get("scale_x", 224 / orig_w if orig_w > 0 else 1.0)
            scale_y = calc.get("scale_y", 224 / orig_h if orig_h > 0 else 1.0)
            pix_change = calc.get("pixel_change_pct", ((50176 - orig_pix) / orig_pix) * 100 if orig_pix > 0 else 0.0)

            st.markdown("**Mathematical Calculations:**")
            st.latex(r"S_x = \frac{W_{\text{target}}}{W_{\text{orig}}} = \frac{224}{" + f"{orig_w}" + r"} = " + f"{scale_x:.4f}")
            st.latex(r"S_y = \frac{H_{\text{target}}}{H_{\text{orig}}} = \frac{224}{" + f"{orig_h}" + r"} = " + f"{scale_y:.4f}")
            st.latex(r"\Delta \text{ Pixel Count} = \frac{224 \times 224 - N_{\text{orig}}}{N_{\text{orig}}} \times 100\% = " + f"{pix_change:+.1f}\\%")
            st.latex(r"I_{\text{target}}(x, y) = \sum_{i \in \{0, 1\}} \sum_{j \in \{0, 1\}} w_{i, j} \cdot I_{\text{orig}}(x_i, y_j)")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Target Size", "224 × 224 px")
            with m2:
                st.metric("Resized Pixels", "50,176 px")
            with m3:
                st.metric("Pixel Scale Δ", f"{pix_change:+.1f}%")

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Step 3: PyTorch Tensor Conversion & Value Range Scaling
        # ----------------------------------------------------------------------
        st.markdown("### 3️⃣ Step 3: Tensor Conversion & [0.0, 1.0] Range Scaling")
        st.markdown(
            "Converts the PIL uint8 image $([0, 255])$ into a 32-bit floating point PyTorch tensor $([0.0, 1.0])$. "
            "The memory layout is permuted from $(H, W, C)$ to PyTorch's native channel-first order $(C, H, W)$."
        )

        col_s3_img, col_s3_calc = st.columns([1, 1.3], gap="medium")
        with col_s3_img:
            st.image(steps["raw_tensor"].permute(1, 2, 0).numpy(), caption="PyTorch Float Tensor (3, 224, 224)", width="stretch")

        with col_s3_calc:
            st.markdown("**Mathematical Formula:**")
            st.latex(r"x_{\text{tensor}}[c, y, x] = \frac{x_{\text{uint8}}[y, x, c]}{255.0} \quad \in [0.0, 1.0]")
            st.latex(r"\text{Dimension Permute: } (H=224, W=224, C=3) \xrightarrow{\text{permute}(2, 0, 1)} (C=3, H=224, W=224)")
            
            raw_r, raw_g, raw_b = calc.get("raw_center_rgb", (128, 128, 128))
            t_r, t_g, t_b = calc.get("tensor_center_rgb", (0.5, 0.5, 0.5))
            cx, cy = calc.get("center_coord", (112, 112))
            st.markdown(f"**Sample Center Pixel Calculation at $(x={cx}, y={cy})$:**")
            st.markdown(
                f"- **Red:** $R_{{\\text{{raw}}}} = {raw_r} \\implies R_{{\\text{{tensor}}}} = \\frac{{{raw_r}}}{{255.0}} = \\mathbf{{{t_r:.4f}}}$\n"
                f"- **Green:** $G_{{\\text{{raw}}}} = {raw_g} \\implies G_{{\\text{{tensor}}}} = \\frac{{{raw_g}}}{{255.0}} = \\mathbf{{{t_g:.4f}}}$\n"
                f"- **Blue:** $B_{{\\text{{raw}}}} = {raw_b} \\implies B_{{\\text{{tensor}}}} = \\frac{{{raw_b}}}{{255.0}} = \\mathbf{{{t_b:.4f}}}$"
            )

        # Pre-normalization statistics table
        raw_stats = calc.get("raw_stats", {
            "R": {"min": 0, "max": 1, "mean": 0.5, "std": 0.2},
            "G": {"min": 0, "max": 1, "mean": 0.5, "std": 0.2},
            "B": {"min": 0, "max": 1, "mean": 0.5, "std": 0.2},
        })
        pct_r = calc.get("pct_r", 33.3)
        pct_g = calc.get("pct_g", 33.3)
        pct_b = calc.get("pct_b", 33.3)

        df_raw_stats = pd.DataFrame([
            {"Channel": "🔴 Red (R)", "Min [0.0, 1.0]": f"{raw_stats['R']['min']:.4f}", "Max [0.0, 1.0]": f"{raw_stats['R']['max']:.4f}", "Mean (μ)": f"{raw_stats['R']['mean']:.4f}", "Std Dev (σ)": f"{raw_stats['R']['std']:.4f}", "Intensity share": f"{pct_r:.1f}%"},
            {"Channel": "🟢 Green (G)", "Min [0.0, 1.0]": f"{raw_stats['G']['min']:.4f}", "Max [0.0, 1.0]": f"{raw_stats['G']['max']:.4f}", "Mean (μ)": f"{raw_stats['G']['mean']:.4f}", "Std Dev (σ)": f"{raw_stats['G']['std']:.4f}", "Intensity share": f"{pct_g:.1f}%"},
            {"Channel": "🔵 Blue (B)", "Min [0.0, 1.0]": f"{raw_stats['B']['min']:.4f}", "Max [0.0, 1.0]": f"{raw_stats['B']['max']:.4f}", "Mean (μ)": f"{raw_stats['B']['mean']:.4f}", "Std Dev (σ)": f"{raw_stats['B']['std']:.4f}", "Intensity share": f"{pct_b:.1f}%"},
        ])
        st.markdown("**Live Pre-Normalization Channel Statistics:**")
        st.dataframe(df_raw_stats, width="stretch", hide_index=True)

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Step 4: ImageNet Z-Score Normalization (Standardization)
        # ----------------------------------------------------------------------
        st.markdown("### 4️⃣ Step 4: ImageNet Z-Score Normalization")
        st.markdown(
            "Each color channel is standardized using the official ImageNet dataset statistics: "
            f"Mean $\\mu = {IMAGENET_MEAN}$ and Standard Deviation $\\sigma = {IMAGENET_STD}$. "
            "These are fixed dataset-wide values, so a single image will not have mean exactly 0 and std exactly 1; they bring inputs to the range the pretrained model expects."
        )

        col_s4_eq, col_s4_calc = st.columns([1, 1.3], gap="medium")
        with col_s4_eq:
            st.markdown("**Mathematical Standardization Equations:**")
            st.latex(r"z = \frac{x - \mu}{\sigma}")
            st.latex(r"z_R = \frac{x_R - 0.485}{0.229}, \quad z_G = \frac{x_G - 0.456}{0.224}, \quad z_B = \frac{x_B - 0.406}{0.225}")
            st.latex(r"\text{Batch Tensor: } (B=1, C=3, H=224, W=224)")
            tot_elements = calc.get("total_tensor_elements", 150528)
            tot_mem = calc.get("tensor_memory_bytes", 602112)
            st.caption(
                f"Total Elements: **{tot_elements:,}** float32 values | "
                f"Memory: **{tot_mem / 1024:.1f} KB**"
            )

        with col_s4_calc:
            norm_r, norm_g, norm_b = calc.get("norm_center_rgb", (0.0, 0.0, 0.0))
            st.markdown(f"**Step-by-Step Arithmetic for Center Pixel $(x={cx}, y={cy})$:**")
            st.latex(
                r"z_R = \frac{" + f"{t_r:.4f}" + r" - 0.485}{0.229} = \frac{" + f"{t_r - 0.485:+.4f}" + r"}{0.229} = \mathbf{" + f"{norm_r:+.4f}" + r"}"
            )
            st.latex(
                r"z_G = \frac{" + f"{t_g:.4f}" + r" - 0.456}{0.224} = \frac{" + f"{t_g - 0.456:+.4f}" + r"}{0.224} = \mathbf{" + f"{norm_g:+.4f}" + r"}"
            )
            st.latex(
                r"z_B = \frac{" + f"{t_b:.4f}" + r" - 0.406}{0.225} = \frac{" + f"{t_b - 0.406:+.4f}" + r"}{0.225} = \mathbf{" + f"{norm_b:+.4f}" + r"}"
            )

        # Post-normalization statistics table
        norm_stats = calc.get("norm_stats", {
            "R": {"min": -2.0, "max": 2.0, "mean": 0.0, "std": 1.0},
            "G": {"min": -2.0, "max": 2.0, "mean": 0.0, "std": 1.0},
            "B": {"min": -2.0, "max": 2.0, "mean": 0.0, "std": 1.0},
        })
        df_norm_stats = pd.DataFrame([
            {"Channel": "🔴 Red (R)", "Normalized Min": f"{norm_stats['R']['min']:+.4f}", "Normalized Max": f"{norm_stats['R']['max']:+.4f}", "Normalized Mean (μ_z)": f"{norm_stats['R']['mean']:+.4f}", "Normalized Std (σ_z)": f"{norm_stats['R']['std']:.4f}"},
            {"Channel": "🟢 Green (G)", "Normalized Min": f"{norm_stats['G']['min']:+.4f}", "Normalized Max": f"{norm_stats['G']['max']:+.4f}", "Normalized Mean (μ_z)": f"{norm_stats['G']['mean']:+.4f}", "Normalized Std (σ_z)": f"{norm_stats['G']['std']:.4f}"},
            {"Channel": "🔵 Blue (B)", "Normalized Min": f"{norm_stats['B']['min']:+.4f}", "Normalized Max": f"{norm_stats['B']['max']:+.4f}", "Normalized Mean (μ_z)": f"{norm_stats['B']['mean']:+.4f}", "Normalized Std (σ_z)": f"{norm_stats['B']['std']:.4f}"},
        ])
        st.markdown("**Live Post-Normalization Channel Statistics:**")
        st.dataframe(df_norm_stats, width="stretch", hide_index=True)

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Step 5: Channel Decompositions & Intensity Share Distribution
        # ----------------------------------------------------------------------
        st.markdown("### 5️⃣ Step 5: Color Channel Decompositions & Intensity Share Calculation")
        st.markdown(
            "Decomposes the image into individual Red, Green, and Blue intensity arrays to analyze color representation (intensity share = channel pixel sum / total pixel sum)."
        )

        col_d1, col_d2, col_d3, col_d4 = st.columns(4, gap="small")
        with col_d1:
            st.markdown("**De-normalized Image**")
            st.image(steps["denormalized_image"], width="stretch")
            mae_val = calc.get("mae", 0.0)
            st.caption(f"Reconstructed (MAE: **{mae_val:.6f}**)")

        with col_d2:
            st.markdown("**Red Channel (R)**")
            fig_r, ax_r = plt.subplots(figsize=(3, 3))
            ax_r.imshow(steps["channel_r"], cmap="Reds")
            ax_r.axis("off")
            st.pyplot(fig_r, width="stretch")
            plt.close(fig_r)
            st.caption(f"Intensity share: **{pct_r:.1f}%** | Mean: **{raw_stats['R']['mean']:.3f}**")

        with col_d3:
            st.markdown("**Green Channel (G)**")
            fig_g, ax_g = plt.subplots(figsize=(3, 3))
            ax_g.imshow(steps["channel_g"], cmap="Greens")
            ax_g.axis("off")
            st.pyplot(fig_g, width="stretch")
            plt.close(fig_g)
            st.caption(f"Intensity share: **{pct_g:.1f}%** | Mean: **{raw_stats['G']['mean']:.3f}**")

        with col_d4:
            st.markdown("**Blue Channel (B)**")
            fig_b, ax_b = plt.subplots(figsize=(3, 3))
            ax_b.imshow(steps["channel_b"], cmap="Blues")
            ax_b.axis("off")
            st.pyplot(fig_b, width="stretch")
            plt.close(fig_b)
            st.caption(f"Intensity share: **{pct_b:.1f}%** | Mean: **{raw_stats['B']['mean']:.3f}**")

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Step 6: Numerical Reversibility (De-normalization) Verification
        # ----------------------------------------------------------------------
        st.markdown("### 6️⃣ Step 6: Numerical Inversion & Reconstruction Verification")
        st.markdown(
            "Verifies mathematical reversibility by inverting normalized values back to the original $[0.0, 1.0]$ range. "
            "The Mean Absolute Error (MAE) confirms that normalization is fully reversible (the resize in Step 2 is not reversible)."
        )

        col_rev1, col_rev2 = st.columns([1, 1.3], gap="medium")
        with col_rev1:
            st.latex(r"x_{\text{reconstructed}} = \text{clamp}(z \cdot \sigma + \mu, 0.0, 1.0)")
            st.latex(r"\text{MAE} = \frac{1}{3 \times 224 \times 224} \sum |x_{\text{raw}} - x_{\text{reconstructed}}| = \mathbf{" + f"{mae_val:.6f}" + r"}")
        with col_rev2:
            recon_r, recon_g, recon_b = calc.get("recon_center_rgb", (t_r, t_g, t_b))
            st.markdown(f"**Center Pixel Inverse Calculation at $(x={cx}, y={cy})$:**")
            st.markdown(
                f"- $R_{{\\text{{recon}}}} = ({norm_r:+.4f} \\times 0.229) + 0.485 = \\mathbf{{{recon_r:.4f}}}$ (Original: ${t_r:.4f}$)\n"
                f"- $G_{{\\text{{recon}}}} = ({norm_g:+.4f} \\times 0.224) + 0.456 = \\mathbf{{{recon_g:.4f}}}$ (Original: ${t_g:.4f}$)\n"
                f"- $B_{{\\text{{recon}}}} = ({norm_b:+.4f} \\times 0.225) + 0.406 = \\mathbf{{{recon_b:.4f}}}$ (Original: ${t_b:.4f}$)"
            )
            st.success(f"✅ Inversion verified: Mean Absolute Error is {mae_val:.6e} (normalization is lossless).")

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Step 7: Training Augmentations & Mathematical Parameters
        # ----------------------------------------------------------------------
        st.markdown("### 7️⃣ Step 7: Training Augmentation Pipeline & Formulations")
        st.markdown(
            "During model training, stochastic data augmentations are applied to synthesize variations and prevent overfitting. "
            "Below are sample augmentations dynamically generated from this input image:"
        )

        aug_cols = st.columns(len(steps["augmentations"]), gap="small")
        for idx, (col_aug, aug_img) in enumerate(zip(aug_cols, steps["augmentations"])):
            with col_aug:
                st.image(aug_img, width="stretch")
                st.caption(f"Augmentation #{idx + 1}")

        st.markdown(
            """
            **Training Augmentation Mathematical Parameters:**
            1. **RandomResizedCrop(224, scale=(0.75, 1.0))**: Cropped crop area $A_{\\text{crop}} \\in [0.75 \\cdot A, 1.0 \\cdot A]$ with random aspect ratio $r \\in [3/4, 4/3]$, bilinearly resized to $224 \\times 224$.
            2. **RandomRotation(degrees=15)**: Rotation angle $\\theta \\sim \\mathcal{U}(-15^\\circ, +15^\\circ)$ with affine rotation matrix $\\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{bmatrix}$.
            3. **RandomHorizontalFlip(p=0.5)**: Reflection matrix $x' = W - 1 - x$ with probability $0.5$.
            4. **ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)**: Multiplicative and additive color factors sampled uniformly: Brightness factor $\\in [0.8, 1.2]$, Contrast $\\in [0.8, 1.2]$, Saturation $\\in [0.8, 1.2]$, Hue shift $\\in [-0.1, 0.1]$.
            """
        )


# ==============================================================================
# Page 2: Model Results
# ==============================================================================
def render_model_results_page(class_names: List[str]):
    st.markdown('<div class="main-title">📊 Model Evaluation Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Quantitative performance benchmark on the held-out test_set (300 images across 10 classes).</div>',
        unsafe_allow_html=True,
    )

    results, err = get_cached_evaluation()
    if err:
        st.error(f"⚠️ {err}")
        return

    # Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Overall Accuracy", value=f"{results['accuracy'] * 100:.2f}%")
    with m2:
        st.metric(label="Macro Precision", value=f"{results['macro_precision'] * 100:.2f}%")
    with m3:
        st.metric(label="Macro Recall", value=f"{results['macro_recall'] * 100:.2f}%")
    with m4:
        st.metric(label="Macro F1-Score", value=f"{results['macro_f1'] * 100:.2f}%")

    st.markdown("---")

    col_cm, col_report = st.columns([1.1, 1], gap="large")

    with col_cm:
        st.markdown("### 🔲 Confusion Matrix")
        st.caption("Rows: Ground truth classes | Columns: Model predictions")
        fig_cm = plot_confusion_matrix(results["confusion_matrix"], results.get("class_names", class_names))
        st.pyplot(fig_cm, width="stretch")
        plt.close(fig_cm)

    with col_report:
        st.markdown("### 📋 Per-Class Classification Report")
        st.caption("Precision, recall, and F1 scores per food category.")
        
        raw_report = results["classification_report"]
        table_rows = []
        for cls in results.get("class_names", class_names):
            if cls in raw_report:
                metrics = raw_report[cls]
                table_rows.append({
                    "Food Class": cls.title(),
                    "Precision": f"{metrics['precision'] * 100:.1f}%",
                    "Recall": f"{metrics['recall'] * 100:.1f}%",
                    "F1-Score": f"{metrics['f1-score'] * 100:.1f}%",
                    "Support": int(metrics["support"]),
                })

        if table_rows:
            df_report = pd.DataFrame(table_rows)
            st.dataframe(df_report, width="stretch", hide_index=True)
            
            macro_avg = raw_report.get("macro avg", {})
            weighted_avg = raw_report.get("weighted avg", {})
            st.markdown(
                f"""
                - **Macro Average**: Precision `{macro_avg.get('precision', 0)*100:.1f}%` | Recall `{macro_avg.get('recall', 0)*100:.1f}%` | F1 `{macro_avg.get('f1-score', 0)*100:.1f}%`
                - **Weighted Average**: Precision `{weighted_avg.get('precision', 0)*100:.1f}%` | Recall `{weighted_avg.get('recall', 0)*100:.1f}%` | F1 `{weighted_avg.get('f1-score', 0)*100:.1f}%`
                - **Total Test Samples**: `{results.get('total_test_samples', 300)}` images
                """
            )


# ==============================================================================
# Main Entry Point & Sidebar Navigation
# ==============================================================================
def main():
    try:
        st.sidebar.image(
            "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=300&auto=format&fit=crop&q=80",
            caption="Indian Culinary AI",
            width="stretch",
        )
    except Exception:
        pass
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio(
        "Go to page:",
        ["Food Classifier", "Model Results"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Information")
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"**Compute Device:** {device_name}\n\n**Backbone:** EfficientNet-B0\n\n**Classes:** 10 Indian foods")

    # Load Model once
    model, class_names, err = get_cached_model()
    if err:
        st.error(f"⚠️ Model status: {err}")
        if page_selection == "Model Results":
            render_model_results_page(class_names or [])
        return

    # Render selected page
    if page_selection == "Food Classifier":
        render_food_classifier_page(model, class_names)
    elif page_selection == "Model Results":
        render_model_results_page(class_names)


if __name__ == "__main__":
    main()
