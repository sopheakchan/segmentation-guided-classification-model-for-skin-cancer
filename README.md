# Skin Cancer Classification using Segmented Lesion Images (HAM10000)

A deep learning project for skin cancer classification using advanced preprocessing techniques, U-Net segmentation, and DenseNet201 transfer learning on the HAM10000 dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

## Overview

This project implements a comprehensive pipeline for automated skin cancer classification. The approach combines sophisticated preprocessing techniques with deep learning models to accurately classify seven types of skin lesions. The key innovation lies in using U-Net segmentation to isolate the lesion area before classification, which helps the classifier focus on the most relevant regions.

## Dataset

**HAM10000 (Human Against Machine with 10000 training images)**

The dataset contains 10,015 dermatoscopic images from diverse populations, primarily collected from:
- Cliff Rosendahl's skin cancer practice in Queensland, Australia
- Dermatology Department of the Medical University of Vienna, Austria

### Class Distribution (7 classes):
1. **MEL** - Melanoma: 1,113 images
2. **NV** - Melanocytic nevus: 6,705 images
3. **BCC** - Basal cell carcinoma: 514 images
4. **AKIEC** - Actinic keratosis / Intraepithelial carcinoma: 327 images
5. **BKL** - Benign keratosis: 1,099 images
6. **DF** - Dermatofibroma: 115 images
7. **VASC** - Vascular lesion: 142 images

** Note:** The original image dataset from Kaggle is **not included** in this repository due to its large size (~2GB). You can download it from [ISIC Archive](https://www.isic-archive.com/) or [Kaggle HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or (https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification).

## Project Structure

```
segmented-classification-model/
│
├── preprocessing-technique/
│   ├── mynotebook.ipynb                    # Data sorting and Dull Razor preprocessing
│   └── unet_model_for-segmentation.ipynb   # U-Net inference and segmentation application
│
├── unet-model/
│   └── unet-trained-model.ipynb            # U-Net training for lesion segmentation
│
├── classification/
│   └── segmented-classification-model.ipynb # DenseNet201 classification model
│
├── GroundTruth.csv                          # One-hot encoded labels
├── HAM10000_metadata.csv                    # Metadata information
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

## Methodology

### 1. Data Preparation and Preprocessing

#### **Step 1.1: Data Sorting**
- Organized 10,015 images into class-specific folders based on labels from `GroundTruth.csv`
- Created directory structure for organized data management

#### **Step 1.2: Dull Razor Algorithm**
Applied the Dull Razor algorithm to remove hair artifacts from dermoscopic images:
<img width="1052" height="288" alt="Screenshot 2025-11-18 155440" src="https://github.com/user-attachments/assets/3b196e7d-95bc-44d7-beef-70ca14a0606f" />


**Algorithm Steps:**
1. **Grayscale Conversion**: Convert RGB image to grayscale
2. **Blackhat Filtering**: Use morphological blackhat operation (11×11 kernel) to detect dark linear structures (hair)
3. **Gaussian Blur**: Apply 3×3 Gaussian blur to smooth the detected hair mask
4. **Binary Thresholding**: Create binary mask with threshold value of 10
5. **Dilation**: Dilate the mask using 9×9 kernel to ensure complete hair coverage
6. **Inpainting**: Use TELEA algorithm to fill in the hair regions with surrounding skin texture

**Benefits:**
- Removes hair occlusions that can interfere with lesion analysis
- Preserves lesion boundaries and texture information
- Improves overall image quality for downstream tasks

#### **Step 1.3: Dataset Splitting**
- **Training Set**: 70% (with augmentation to balance classes)
- **Validation Set**: 15%
- **Test Set**: 15%
- Used stratified splitting to maintain class distribution across all sets
<img width="1489" height="390" alt="image (3)" src="https://github.com/user-attachments/assets/c969a2f2-447e-4540-b1bc-b5ed94f82df2" />


### 2. U-Net Segmentation Model

#### **Model Architecture**
Built a custom U-Net architecture for precise lesion segmentation:

**Encoder Path:**
- Conv Block 1: 16 filters + MaxPooling + Dropout(0.1)
- Conv Block 2: 32 filters + MaxPooling + Dropout(0.1)
- Conv Block 3: 64 filters + MaxPooling + Dropout(0.2)
- Conv Block 4: 128 filters + MaxPooling + Dropout(0.2)

**Bottleneck:**
- Conv Block 5: 256 filters + Dropout(0.3)

**Decoder Path:**
- Upsampling blocks with skip connections from encoder
- Conv Blocks: 128 → 64 → 32 → 16 filters
- Final 1×1 convolution with sigmoid activation for binary mask output

**Each Conv Block contains:**
- Conv2D (3×3) → BatchNormalization → ReLU
- Conv2D (3×3) → BatchNormalization → ReLU
- Optional Dropout for regularization

#### **Training Configuration**
- **Input Size**: 224×224×3
- **Batch Size**: 16
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy + Dice Loss (combined)
- **Metrics**: Dice Coefficient, IoU (Intersection over Union), Accuracy
- **Callbacks**:
  - ModelCheckpoint (save best model based on validation Dice)
  - ReduceLROnPlateau (factor=0.5, patience=5)
- **Data Augmentation**:
  - Random horizontal/vertical flips
  - Random rotations (±20°)
  - Random brightness adjustments
  - Random zoom (0.9-1.1×)

#### **U-Net Performance**
- **Test Accuracy**: ~91%+
- **Test Dice Coefficient**: ~0.91+
- **Test IoU**: ~0.83+
  <img width="1280" height="448" alt="image" src="https://github.com/user-attachments/assets/6e59934c-8676-47ad-9869-e79881ecad1f" />


**Key Achievement:** The U-Net model successfully learned to segment skin lesions with high precision, enabling accurate extraction of lesion regions from dermoscopic images.

#### **Step 2.1: Segmentation Application**
After training, the U-Net model was used to:
1. Generate binary masks for all images in the dataset
2. Apply masks to original images to isolate lesion regions
3. Create a new segmented dataset with only the important lesion areas
4. This focused approach helps the classification model concentrate on relevant features
<img width="766" height="145" alt="image_2025-11-19_14-41-20" src="https://github.com/user-attachments/assets/c78e32c1-332a-4d12-9ef3-785bb648f102" />
<img width="397" height="150" alt="image_2025-11-19_14-42-43" src="https://github.com/user-attachments/assets/8ab22cce-e20f-42f9-9fc1-10e94e56a615" />



### 3. DenseNet201 Classification Model

#### **Model Architecture**
Utilized transfer learning with DenseNet201:

**Base Model:**
- DenseNet201 pre-trained on ImageNet
- Input shape: 224×224×3
- Initially frozen for feature extraction

**Custom Classification Head:**
- GlobalAveragePooling2D
- Dense(256, activation='relu')
- Dense(128, activation='relu')
- Dropout(0.5)
- Dense(7, activation='softmax') - 7 skin lesion classes

#### **Training Strategy**

**Phase 1: Ground Truth Dataset (Feature Extraction)**
- **Dataset**: Manually annotated ground truth segmentation masks
- Base model frozen, only train custom top layers
- Optimizer: Adam (lr=1e-4)
- Epochs: 10
- Preprocessing: DenseNet-specific preprocessing (ImageNet mean-std normalization)
- Batch Size: 32
- Callbacks: EarlyStopping (patience=7), ReduceLROnPlateau (factor=0.5, patience=3)

**Phase 2: Ground Truth Dataset (Fine-tuning)**
- Unfreeze all DenseNet201 layers
- Lower learning rate: Adam (lr=1e-5)
- Epochs: 10
- Continue training with same callbacks

**Phase 3: Transfer Learning on U-Net Segmented Dataset**
- Transfer learned weights from ground truth training
- Train on U-Net segmented images (automatically generated masks)
- Fine-tuning: Adam (lr=5e-6) - even lower learning rate
- Epochs: 30
- Same preprocessing and augmentation pipeline

#### **Classification Results**

**Ground Truth Dataset Performance:**
- **Test Accuracy**: ~85%
- **Macro Precision**: ~0.85
- **Macro Recall**: ~0.85
- **Macro F1-Score**: ~0.85
- **Macro AUC**: ~0.92+

**U-Net Segmented Dataset Performance:**
- **Test Accuracy**: ~85%
- **Macro Precision**: ~0.85
- **Macro Recall**: ~0.85
- **Macro F1-Score**: ~0.85

**Per-Class Performance (Representative):**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| AKIEC | 0.70-0.75 | 0.65-0.70 | 0.67-0.72 | ~50 |
| BCC | 0.80-0.85 | 0.75-0.80 | 0.77-0.82 | ~75 |
| BKL | 0.80-0.85 | 0.80-0.85 | 0.80-0.85 | ~165 |
| DF | 0.75-0.80 | 0.70-0.75 | 0.72-0.77 | ~18 |
| MEL | 0.75-0.80 | 0.75-0.80 | 0.75-0.80 | ~167 |
| NV | 0.90-0.95 | 0.92-0.97 | 0.91-0.96 | ~1006 |
| VASC | 0.85-0.90 | 0.80-0.85 | 0.82-0.87 | ~21 |

## Results Summary
<img width="1189" height="390" alt="image (4)" src="https://github.com/user-attachments/assets/a97830a1-c078-47c3-b58d-6c6e2d07f87f" />

### Achievements

1. **Successful Preprocessing Pipeline**
   - Dull Razor algorithm effectively removed hair artifacts
   - Improved image quality for both segmentation and classification

2. **High-Performance Segmentation**
   - U-Net achieved **91%+ accuracy** and **0.91+ Dice coefficient**
   - Generated high-quality segmentation masks automatically
   - Enabled focus on lesion-specific regions

3. **Robust Classification**
   - DenseNet201 achieved **~85% overall accuracy**
   - Strong performance on majority class (NV: 90-95% metrics)
   - Reasonable performance on minority classes despite imbalance
   - Transfer learning from ground truth to U-Net segments was successful

4. **End-to-End Automated Pipeline**
   - Complete workflow from raw images to classification predictions
   - Reproducible results with documented methodology

### Visualizations

The project includes comprehensive visualizations:
- **Training curves**: Loss and accuracy over epochs for all training phases
- **Confusion matrices**: Both raw counts and normalized
- **ROC curves**: Per-class and micro/macro averaged AUC
- **Precision-Recall curves**: Per-class average precision scores
- **Segmentation examples**: Original images, masks, and segmented outputs
- **Before/After preprocessing**: Dull Razor effect visualization

## Limitations and Future Work

### Current Limitations

1. **Classification Stability Issues** 
   - The training curves show some instability/fluctuation, especially in the later epochs
   - Model hasn't fully converged to a stable optimum
   - Validation loss shows some volatility indicating potential overfitting

2. **Class Imbalance Challenges**
   - Despite augmentation, minority classes (DF: 115, VASC: 142, AKIEC: 327) remain underrepresented
   - Lower performance on minority classes compared to NV (6,705 images)

3. **Generalization Concerns**
   - Dataset primarily from two geographic locations (Australia and Austria)
   - May not generalize well to different skin types, camera equipment, or imaging conditions

### Proposed Improvements

1. **Model Stability**
   - Implement more aggressive regularization (higher dropout rates, L2 regularization)
   - Use label smoothing to prevent overconfident predictions
   - Experiment with different learning rate schedules (cosine annealing, warm restarts)
   - Increase early stopping patience or use gradient clipping
   - Try ensemble methods (multiple models with different initializations)

2. **Architecture Improvements**
   - Experiment with other architectures (EfficientNet, Vision Transformers, ResNet)
   - Implement attention mechanisms to focus on critical lesion features
   - Use multi-scale feature fusion

3. **Training Strategy**
   - Implement focal loss to handle class imbalance better
   - Use curriculum learning (train on easy samples first)
   - Experiment with longer training schedules with careful monitoring

4. **Validation Strategy**
   - Use k-fold cross-validation for more robust performance estimates
   - Implement external validation on different datasets (ISIC 2019, 2020)
   - Analyze per-class performance in detail to identify specific weaknesses

## Installation

### Environment
- Python 3.8+
- Google Colab, Kaggle Notebook
- 8GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/skin-cancer-classification.git
cd skin-cancer-classification/segmented-classification-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the HAM10000 dataset:
   - Visit [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - Visit [Kaggle HAM10000](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
   - Download and extract to a `data/` folder (not included in repo)

## Usage

### 1. Preprocessing (Dull Razor)
```bash
jupyter notebook preprocessing-technique/mynotebook.ipynb
```
- Run cells sequentially to apply Dull Razor algorithm
- Images will be saved to `preprocessed_images/` folder

### 2. U-Net Training
```bash
jupyter notebook unet-model/unet-trained-model.ipynb
```
- Trains U-Net segmentation model
- Saves best model to `best_model.h5`
- Expected training time: 3-6 hours on GPU

### 3. Apply Segmentation
```bash
jupyter notebook preprocessing-technique/unet_model_for-segmentation.ipynb
```
- Loads trained U-Net model
- Generates segmentation masks for all images
- Creates segmented dataset

### 4. Classification Training
```bash
jupyter notebook classification/segmented-classification-model.ipynb
```
- Trains DenseNet201 classifier
- Evaluates on test set
- Generates all performance metrics and visualizations

## References

1. **Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).

2. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

3. **DenseNet**: Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. CVPR 2017.

4. **Dull Razor**: Lee, T., Ng, V., Gallagher, R., Coldman, A., & McLean, D. (1997). Dullrazor: A software approach to hair removal from images. Computers in Biology and Medicine, 27(6), 533-543.

## License

This project is for educational and research purposes. Please refer to the HAM10000 dataset license for data usage restrictions.

## Author

Sopheak Chan  
[GitHub](https://github.com/sopheakchan)

## Acknowledgments

- ISIC Archive for providing the HAM10000 dataset
- Kaggle community for computational resources and support
- TensorFlow and Keras teams for excellent deep learning frameworks

---

**Note**: This project is part of ongoing research in automated skin cancer detection. The models should not be used for actual medical diagnosis without proper validation and clinical testing.
