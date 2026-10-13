🍲 Indian Food Image Classifier | EfficientNet-B0

A production-ready deep learning application for classifying 10 iconic Indian food dishes using EfficientNet-B0 in PyTorch, deployed with an interactive Streamlit dashboard.

The application features real-time inference, Top-3 confidence scoring, Grad-CAM visual explainability, a complete test set evaluation benchmark (92.67% accuracy), and a transparent 7-step deep learning preprocessing pipeline with live mathematical calculations and aspect ratio tracking for every single step.

📌 Table of Contents
Overview & Key Features
Supported Food Classes
Model Architecture & Training
Live Mathematical Calculations & Preprocessing Pipeline
Model Evaluation & Benchmark Results
Project Structure
Installation & Quick Start
Running the Streamlit Application
Testing & Verification
🚀 Overview & Key Features
EfficientNet-B0 Backbone: High-accuracy, parameter-efficient convolutional neural network pretrained on ImageNet and fine-tuned on Indian culinary imagery.
Streamlit Interactive UI:
Option A: Upload your own image (JPG, JPEG, PNG).
Option B: Pick random test samples from any of the 10 food categories.
Top-3 Confidence Distributions: Real-time probability chart for top predictions (trained with 
0.1
0.1 label smoothing).
Grad-CAM Explainability: Live gradient-weighted class activation heatmaps showing which spatial features and regions the neural network focuses on.
Live Step-by-Step Mathematical Calculations: Complete breakdown of the computer vision pipeline with exact live arithmetic and aspect ratio calculations for every stage.
Quantitative Benchmark Page: Interactive confusion matrix heatmap and per-class precision, recall, and F1-score evaluation on 300 held-out test images.
🍽️ Supported Food Classes

The model classifies 10 authentic Indian dishes:

#	Food Class	Description
1	Bisi Bele Bath	Traditional Karnataka spiced rice and lentil dish
2	Dhokla	Savory steamed fermented chickpea snack from Gujarat
3	Dosa	Crispy fermented rice and lentil crepe from South India
4	Idly	Steamed savory rice cake staple
5	Kathi Roll	Kolkata street-food skewer-roasted kebab rolled in paratha
6	Medu Vadai	Crispy, savory doughnut-shaped South Indian lentil fritter
7	Noodles	Stir-fried Indo-Chinese noodle dish
8	Poori	Deep-fried puffed unleavened wheat bread
9	Upma	Thick savory porridge made from dry-roasted semolina
10	Ven Pongal	Savory South Indian rice and yellow moong dal porridge seasoned with ghee, cumin, and pepper
🧠 Model Architecture & Training
Architecture Specifications
Backbone: torchvision.models.efficientnet_b0 (pretrained ImageNet feature extractor).
Classification Head:
python
nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features=1280, out_features=10)
)
Parameters: ~4.02 Million trainable/fine-tuned parameters.
Two-Phase Training Strategy
Phase 1 (Classifier Head Warmup): Backbone frozen, classifier trained with AdamW (
LR
=
10
−
3
LR=10
−3
) for 5 epochs.
Phase 2 (End-to-End Fine-Tuning): Backbone unfrozen, trained with reduced learning rate (
LR
=
10
−
4
LR=10
−4
) and Cosine Annealing scheduler for 15 epochs.
Loss Function: Cross-Entropy Loss with Label Smoothing (
0.1
0.1) to avoid overconfident predictions and improve generalization.
📐 Live Mathematical Calculations & Preprocessing Pipeline

The web application exposes a detailed expander: "🔍 Show image processing steps & calculations", computing all values dynamically in real time from the active image:

1️⃣ Step 1: Input Acquisition & Dimension Preparation
EXIF Transposition: Auto-corrects camera orientation metadata.
Thumbnail Proportional Downscaling: Restricts images to a max dimension of 
1024
×
1024
1024×1024 px while preserving natural aspect ratio.
Live Calculations: 
Aspect Ratio (AR)
=
𝑊
orig
𝐻
orig
(
Landscape 
>
1
,
 Portrait 
<
1
,
 Square 
=
1
)
Aspect Ratio (AR)=
H
orig
	​

W
orig
	​

	​

(Landscape >1, Portrait <1, Square =1) 
𝑁
orig
=
𝑊
orig
×
𝐻
orig
 pixels
N
orig
	​

=W
orig
	​

×H
orig
	​

 pixels 
Raw Memory (uint8 RGB)
=
𝑁
orig
×
3
 bytes
1
,
048
,
576
 MB
Raw Memory (uint8 RGB)=
1,048,576
N
orig
	​

×3 bytes
	​

 MB
2️⃣ Step 2: Spatial Resizing to (224 × 224)
Bilinear Interpolation: Resizes image to EfficientNet-B0's required input resolution.
Live Calculations: 
Target Aspect Ratio
=
224
224
=
1.000
:
1
(
1
:
1
 Square
)
Target Aspect Ratio=
224
224
	​

=1.000:1(1:1 Square) 
AR Distortion Factor
=
AR
target
AR
orig
=
1.000
AR
orig
AR Distortion Factor=
AR
orig
	​

AR
target
	​

	​

=
AR
orig
	​

1.000
	​

 
𝑆
𝑥
=
224
𝑊
orig
,
𝑆
𝑦
=
224
𝐻
orig
S
x
	​

=
W
orig
	​

224
	​

,S
y
	​

=
H
orig
	​

224
	​

 
Δ
 Pixel Count
=
50
,
176
−
𝑁
orig
𝑁
orig
×
100
%
Δ Pixel Count=
N
orig
	​

50,176−N
orig
	​

	​

×100%
3️⃣ Step 3: Tensor Conversion & [0.0, 1.0] Range Scaling
Layout Permutation: Converts PIL uint8 
[
0
,
255
]
[0,255] format 
(
𝐻
,
𝑊
,
𝐶
)
(H,W,C) to PyTorch float32 
[
0.0
,
1.0
]
[0.0,1.0] format 
(
𝐶
,
𝐻
,
𝑊
)
(C,H,W).
Live Calculations: 
Spatial Tensor Aspect Ratio
=
224
224
=
1.000
:
1
Spatial Tensor Aspect Ratio=
224
224
	​

=1.000:1 
𝑥
tensor
[
𝑐
,
𝑦
,
𝑥
]
=
𝑥
uint8
[
𝑦
,
𝑥
,
𝑐
]
255.0
x
tensor
	​

[c,y,x]=
255.0
x
uint8
	​

[y,x,c]
	​

 
Total Elements
=
3
×
224
×
224
=
150
,
528
 values
Total Elements=3×224×224=150,528 values 
Memory
=
150
,
528
×
4
 bytes
1
,
024
=
588.0
 KB
Memory=
1,024
150,528×4 bytes
	​

=588.0 KB
Exact center pixel 
(
𝑥
=
112
,
𝑦
=
112
)
(x=112,y=112) float conversion for Red, Green, and Blue channels.
Real-time channel summary statistics table: Min, Max, Mean (
𝜇
μ), and Standard Deviation (
𝜎
σ).
4️⃣ Step 4: ImageNet Z-Score Normalization
Standardization: Centers and scales channels using canonical ImageNet statistics: 
𝜇
=
[
0.485
,
0.456
,
0.406
]
,
𝜎
=
[
0.229
,
0.224
,
0.225
]
μ=[0.485,0.456,0.406],σ=[0.229,0.224,0.225]
Live Calculations: 
Spatial Aspect Ratio
=
224
224
=
1.000
:
1
Spatial Aspect Ratio=
224
224
	​

=1.000:1 
𝑧
𝑅
=
𝑥
𝑅
−
0.485
0.229
,
𝑧
𝐺
=
𝑥
𝐺
−
0.456
0.224
,
𝑧
𝐵
=
𝑥
𝐵
−
0.406
0.225
z
R
	​

=
0.229
x
R
	​

−0.485
	​

,z
G
	​

=
0.224
x
G
	​

−0.456
	​

,z
B
	​

=
0.225
x
B
	​

−0.406
	​

Step-by-step arithmetic substitution for center pixel 
(
112
,
112
)
(112,112).
Real-time post-normalization statistics table (
𝑧
z-score min, max, mean, std).
5️⃣ Step 5: Color Channel Decompositions & Intensity Share
Channel Planes: Displays individual Red, Green, and Blue intensity grayscale heatmaps (
224
×
224
224×224, AR 
1
:
1
1:1).
Live Calculations: 
𝐸
𝑐
=
∑
𝑦
=
1
224
∑
𝑥
=
1
224
𝐼
𝑐
[
𝑦
,
𝑥
]
(
Channel Energy
)
E
c
	​

=∑
y=1
224
	​

∑
x=1
224
	​

I
c
	​

[y,x](Channel Energy) 
𝐸
total
=
𝐸
𝑅
+
𝐸
𝐺
+
𝐸
𝐵
E
total
	​

=E
R
	​

+E
G
	​

+E
B
	​

 
Intensity Share
𝑐
=
𝐸
𝑐
𝐸
total
×
100
%
Intensity Share
c
	​

=
E
total
	​

E
c
	​

	​

×100%
6️⃣ Step 6: Numerical Inversion & Reconstruction Verification
Reversibility Proof: Inverts normalized values back to raw 
[
0.0
,
1.0
]
[0.0,1.0] space: 
𝑥
recon
=
clamp
(
𝑧
⋅
𝜎
+
𝜇
,
0.0
,
1.0
)
x
recon
	​

=clamp(z⋅σ+μ,0.0,1.0)
Live Calculations: 
Reconstructed Aspect Ratio
=
224
224
=
1.000
:
1
Reconstructed Aspect Ratio=
224
224
	​

=1.000:1 
Mean Absolute Error (MAE)
=
1
150
,
528
∑
∣
𝑥
raw
−
𝑥
recon
∣
≈
0.000000
(
100% Lossless
)
Mean Absolute Error (MAE)=
150,528
1
	​

∑∣x
raw
	​

−x
recon
	​

∣≈0.000000(100% Lossless)
7️⃣ Step 7: Training Augmentation Pipeline
Augmentation Previews: Synthesizes 5 live sample augmentations dynamically from the current image.
Live Calculations: 
Augmented Output Aspect Ratio
=
224
224
=
1.000
:
1
Augmented Output Aspect Ratio=
224
224
	​

=1.000:1 
Crop Aspect Ratio Range: 
𝑟
crop
∈
[
3
4
,
4
3
]
=
[
0.750
:
1
,
1.333
:
1
]
Crop Aspect Ratio Range: r
crop
	​

∈[
4
3
	​

,
3
4
	​

]=[0.750:1,1.333:1] 
𝐴
crop
∈
[
0.75
×
50
,
176
,
 
1.0
×
50
,
176
]
=
[
37
,
632
,
 
50
,
176
]
 pixels
A
crop
	​

∈[0.75×50,176, 1.0×50,176]=[37,632, 50,176] pixels
📊 Model Evaluation & Benchmark Results

Evaluated on the held-out test_set (300 images across 10 classes, 30 images per class):

Metric	Score
Overall Accuracy	92.67%
Macro Precision	92.94%
Macro Recall	92.67%
Macro F1-Score	92.70%
Per-Class Performance Breakdown
Food Category	Precision	Recall	F1-Score	Test Samples
🍛 Bisi Bele Bath	100.0%	90.0%	94.7%	30
🧽 Dhokla	100.0%	93.3%	96.6%	30
🥞 Dosa	93.3%	93.3%	93.3%	30
⚪ Idly	90.0%	90.0%	90.0%	30
🌯 Kathi Roll	96.8%	100.0%	98.4%	30
🍩 Medu Vadai	100.0%	93.3%	96.6%	30
🍜 Noodles	90.6%	96.7%	93.5%	30
🥟 Poori	93.5%	96.7%	95.1%	30
🥣 Upma	82.8%	80.0%	81.4%	30
🍲 Ven Pongal	82.4%	93.3%	87.5%	30
📁 Project Structure
text
restaurant/
├── EfficientNet_B0_best.pt   # Trained model checkpoint weights and class metadata
├── app.py                     # Streamlit multi-page web application
├── model.py                   # EfficientNet-B0 architecture, loader, and Grad-CAM implementation
├── preprocess.py              # Transforms, EXIF handler, and live calculations extractor
├── train.py                   # Two-phase training pipeline and test set evaluator
├── check_model.py             # Checkpoint verification script
├── test_app_flow.py           # End-to-end unit and verification test suite
├── test_results.json          # Cached quantitative benchmark evaluation metrics
├── requirements.txt           # Python dependency requirements
├── README.md                  # Comprehensive project documentation
└── food20dataset/             # Dataset directory
    ├── train_set/             # Training images organized by class folder
    └── test_set/              # Held-out testing images (30 per class)
💻 Installation & Quick Start
1. Prerequisites
Python 3.9+ or Python 3.10+
CUDA-enabled GPU (optional; CPU inference is fast and supported)
2. Clone / Open Workspace
bash
cd restaurant
3. Create & Activate a Virtual Environment
bash
# Windows
python -m venv venv
.\venv\Scripts\activate
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
bash
pip install -r requirements.txt
🌐 Running the Streamlit Application

Start the web dashboard with:

bash
streamlit run app.py

Once started, open your browser at http://localhost:8501.

App Pages:
Food Classifier:
Upload photos or pick random test images.
View live classification and top-3 confidence bars.
Toggle 🔥 Show Grad-CAM Attention Heatmap.
Expand 🔍 Show image processing steps & calculations for the full mathematical breakdown.
Model Results:
View the full confusion matrix heatmap and classification report across all 300 test images.
🧪 Testing & Verification

Run the automated test suite to verify checkpoint integrity, all 10 classes, image transformations, and Grad-CAM generation:

bash
python test_app_flow.py

Expected output:

text
Checkpoint loaded successfully. Best val loss: 0.7466
Class names from checkpoint: ['bisibelebath', 'dhokla', 'dosa', 'idly', 'kathi roll', 'meduvadai', 'noodles', 'poori', 'upma', 'ven pongal']
--- Testing Sample from Each Class ---
Sample Accuracy: 9/10 (90.0%)
--- Testing EXIF transpose and large image scaling ---
Original size: (3000, 2000) -> Prepared size: (1024, 683) (Max dim <= 1024)
--- Testing Grad-CAM Heatmap Generation ---
Grad-CAM shape: (7, 7), Overlay dimensions: (500, 441)
ALL VERIFICATION TESTS COMPLETED AND PASSED!
