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
    page_title="South Indian Food Classifier | EfficientNet-B0",
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
        pick_btn = st.button("🎲 Pick Random Sample", use_container_width=True)

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
        st.image(active_img, caption=source_label, use_container_width=True)

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
                    use_container_width=True
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
                <div class="info-tag">Note: Model was trained with label smoothing 0.1, making 70–90% normal for high certainty.</div>
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
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================================================================
    # Image Processing Steps Expander
    # ==============================================================================
    with st.expander("🔍 Show image processing steps", expanded=False):
        st.markdown("Detailed step-by-step image transformation stages applied during inference:")
        
        try:
            steps = extract_preprocessing_steps(active_img, num_aug_samples=5)
        except Exception as e:
            st.error(f"Error extracting image processing steps: {e}")
            return

        # Row 1: Steps a, b, c
        st.markdown("#### 1. EXIF Correction, Resizing & Tensor Conversion")
        col_a, col_b, col_c = st.columns(3, gap="medium")

        with col_a:
            st.markdown("**a. Original Image (EXIF transposed)**")
            st.image(steps["original_image"], use_container_width=True)
            st.caption(f"Dimensions: **{steps['original_size'][0]} x {steps['original_size'][1]} px** (RGB)")

        with col_b:
            st.markdown("**b. Exact Resize to (224, 224)**")
            st.image(steps["resized_image"], use_container_width=True)
            st.caption(f"Dimensions: **{steps['resized_size'][0]} x {steps['resized_size'][1]} px** (No center crop)")

        with col_c:
            st.markdown("**c. Converted to Tensor**")
            st.image(steps["raw_tensor"].permute(1, 2, 0).numpy(), use_container_width=True)
            shape_str = str(steps["tensor_shape"])
            st.caption(f"Shape: **{shape_str}** | Normalized range: **[0.0, 1.0]**")

        st.markdown("---")

        # Row 2: Step d - Normalization & Channels
        st.markdown("#### 2. Normalization & Channel Decompositions")
        st.caption(
            f"Normalized using ImageNet mean: {IMAGENET_MEAN} and std: {IMAGENET_STD}. "
            "Displayed after de-normalizing, plus separate Red, Green, and Blue channel intensity views."
        )
        col_d1, col_d2, col_d3, col_d4 = st.columns(4, gap="small")

        with col_d1:
            st.markdown("**De-normalized Image**")
            st.image(steps["denormalized_image"], use_container_width=True)
            st.caption("Reconstructed from normalized tensor")

        with col_d2:
            st.markdown("**Red Channel (R)**")
            fig_r, ax_r = plt.subplots(figsize=(3, 3))
            ax_r.imshow(steps["channel_r"], cmap="Reds")
            ax_r.axis("off")
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)
            st.caption("Intensity of Red channel")

        with col_d3:
            st.markdown("**Green Channel (G)**")
            fig_g, ax_g = plt.subplots(figsize=(3, 3))
            ax_g.imshow(steps["channel_g"], cmap="Greens")
            ax_g.axis("off")
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)
            st.caption("Intensity of Green channel")

        with col_d4:
            st.markdown("**Blue Channel (B)**")
            fig_b, ax_b = plt.subplots(figsize=(3, 3))
            ax_b.imshow(steps["channel_b"], cmap="Blues")
            ax_b.axis("off")
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)
            st.caption("Intensity of Blue channel")

        st.markdown("---")

        # Row 3: Step e - Training Augmentations
        st.markdown("#### 3. Training Augmentation Examples")
        st.caption("Sample augmented versions generated using RandomResizedCrop, RandomHorizontalFlip, RandomRotation(15°), and ColorJitter:")
        
        aug_cols = st.columns(len(steps["augmentations"]), gap="small")
        for idx, (col_aug, aug_img) in enumerate(zip(aug_cols, steps["augmentations"])):
            with col_aug:
                st.image(aug_img, use_container_width=True)
                st.caption(f"Augmentation #{idx + 1}")


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
        st.pyplot(fig_cm, use_container_width=True)
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
            st.dataframe(df_report, use_container_width=True, hide_index=True)
            
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
    st.sidebar.image(
        "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=300&auto=format&fit=crop&q=80",
        caption="South Indian Culinary AI",
        use_container_width=True,
    )
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio(
        "Go to page:",
        ["Food Classifier", "Model Results"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Information")
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"**Compute Device:** {device_name}\n\n**Backbone:** EfficientNet-B0\n\n**Classes:** 10 South Indian foods")

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
