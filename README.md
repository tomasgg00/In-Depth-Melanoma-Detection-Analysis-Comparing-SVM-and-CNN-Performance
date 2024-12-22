# Skin Cancer Detection: A Comparative Analysis of SVM and CNN on Melanoma Image Classification

## Project Description
This project implements a comparative study of **Support Vector Machines (SVM)** and **Convolutional Neural Networks (CNN)** for the classification of melanoma (benign vs malignant) using image datasets. Developed as part of a university Machine Learning and Deep Learning course, this project was conducted in collaboration with three colleagues.

---

## Overview
With melanoma skin cancer on the rise, early and accurate diagnosis is crucial. This project develops and compares the performance of:
1. **Naive Bayes** (Baseline Model)
2. **Support Vector Machines (SVM)**
3. **Convolutional Neural Networks (CNN)**

The goal is to assist dermatologists in detecting melanoma effectively through machine learning models.

---

## Dataset
The dataset comprises images categorized into:
- **Benign**
- **Malignant**
- **Undetected**

### Data Preprocessing
The preprocessing pipeline includes:
- **Filtering**: Removal of duplicates using image hashing (aHash, dHash, wHash).
- **Aspect Ratio Filtering**: Removal of images outside valid size ranges.
- **RGB Filtering**: Ensuring proper color balance.
- **Image Resizing**: Standardized to 128x128 pixels.

---

## Methods and Models

### 1. **Naive Bayes (Baseline Model)**
- Simple probabilistic classifier.
- Achieved an accuracy of **58%**.

### 2. **Support Vector Machines (SVM)**
- **Hyperparameter Optimization**: Conducted using GridSearchCV.
- **Feature Selection**: Implemented with Random Forest to reduce feature dimensions.
- **Dimensionality Reduction**: Applied Principal Component Analysis (PCA).
- Optimized accuracy: **87%**.

### 3. **Convolutional Neural Networks (CNN)**
- **Custom Architecture**:
  - 3 Convolutional Layers with ReLU activation and MaxPooling.
  - Batch Normalization for stable training.
  - Fully Connected Dense Layers with Dropout for regularization.
- **Optimizer**: Adam | **Loss Function**: Categorical Crossentropy
- Test accuracy: **88%**.

---

## Results
| Model        | Accuracy |
|--------------|----------|
| Naive Bayes  | 58%      |
| SVM          | 87%      |
| CNN          | **88%**  |

### Key Insight
The CNN model outperformed traditional SVM, highlighting the advantages of deep learning in image classification tasks. However, feature selection and PCA significantly boosted SVM's performance.


