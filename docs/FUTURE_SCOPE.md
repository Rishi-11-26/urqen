# URQEN — Future Scope & Multimodal Architecture Roadmap

> **Status:** Post-v1.0.0 Roadmap Document  
> **Scope for v1.0.0:** Tabular Machine Learning & Data Intelligence Only.

This document preserves the architectural designs, prototypes, and specifications for expanding URQEN beyond structured tabular data in subsequent releases (v1.1+).

---

## 1. Computer Vision & Visual Intelligence (v1.1+)

### 1.1 Objective
Enable automated classification and continuous regression on image datasets without requiring complex GPU clusters for baseline exploration.

### 1.2 Featurization Architecture
- **Deep Feature Representation:** Transfer learning using pre-trained convolutional backbones (e.g. ResNet-18 / EfficientNet-B0 with ImageNet weights). The classification head is truncated, and 512-dimensional continuous embeddings are extracted for downstream classical algorithms.
- **Perceptual Fallback:** When deep learning dependencies are absent, a multi-scale spatial color quadrant histogram and gradient orientation distribution extractor (64-D) provides deterministic baseline representations.

### 1.3 Supported Algorithmic Suite
- **Classifiers:** Support Vector Machine (RBF), Random Forest, XGBoost, K-Nearest Neighbors, Logistic Regression, Extra Trees, Gradient Boosting, Multi-Layer Perceptron (MLP).
- **Regressors:** Support Vector Regressor (SVR), Random Forest Regressor, XGBoost Regressor, Ridge Regression, KNN Regressor, Extra Trees Regressor, Gradient Boosting Regressor, MLP Regressor.

### 1.4 Dataset Ingestion Specifications
- Upload format: ZIP archive containing subfolders (folder name = class label) or flat archive with an accompanying `labels.csv` for continuous regression targets.

---

## 2. Audio & Acoustic Intelligence (v1.2+)

### 2.1 Objective
Provide automated acoustic signal classification and regression for voice, environmental sound, and audio signals.

### 2.2 Featurization Architecture
- **Acoustic Signal Processing:** Extraction of Mel-Frequency Cepstral Coefficients (20 MFCC means + 20 MFCC standard deviations), Spectral Centroid, Spectral Rolloff, and Zero-Crossing Rate (ZCR) via `librosa`.
- **Native Spectral Fallback:** Dependency-free FFT spectrum featurizer computing RMS energy, spectral centroid, spectral spread, crest factor, and 16 log-filterbank subband energies (32-D).

### 2.3 Supported Algorithmic Suite
- **Classifiers:** SVM (RBF), Random Forest, XGBoost, Naive Bayes (GaussianNB), KNN, Logistic Regression, Gradient Boosting, MLP.
- **Regressors:** SVR, Random Forest, XGBoost, Ridge Regression, KNN, Gradient Boosting, MLP.

---

## 3. Video & Multimodal Fusion (v2.0+)

### 3.1 Objective
Multimodal fusion combining tabular clinical/demographic metadata with visual (scans/photos) and temporal/audio inputs.

### 3.2 Fusion Strategy
- Early fusion: Concatenation of normalized tabular feature vectors and deep modality embeddings into a unified ColumnTransformer pipeline.
- Late fusion: Decision-level ensembling across modality-specific models.
