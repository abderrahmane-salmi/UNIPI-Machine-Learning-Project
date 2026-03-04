# 🧠 Machine Learning Project – ML-CUP 2025 (Type B)

**University of Pisa – Department of Computer Science**  
**Academic Year:** 2025/2026  

## 👥 Authors

- [Abderrahmane Salmi]([url](https://github.com/abderrahmane-salmi))  
- [Aymene Chafai]([url](https://github.com/aymene-chafai))
- [Omrane Yakoubi]([url](https://github.com/omraneyakoubi))

---

# 📌 Project Overview

This project was developed for the *Machine Learning* course and follows the **Type B (Model Comparison)** guidelines.

The goal is to compare different machine learning models on:

- 🧩 **MONK Datasets** (Binary Classification)
- 📊 **ML-CUP 2025 Dataset** (Multi-Target Regression)

The main objective is **not** achieving state-of-the-art performance, but demonstrating:

- Correct validation methodology
- Proper model selection
- Rigorous performance evaluation
- Critical analysis of inductive bias

---

# 📂 Repository Structure

```text
.
├── src/
│   ├── knn/          # Jupyter notebooks for KNN
│   ├── linear/       # Jupyter notebooks for Linear Regression
│   ├── nn/           # Jupyter notebooks for Neural Networks
│   ├── svm/          # Jupyter notebooks for Support Vector Machines
│   ├── common/       # Utility scripts for data loading and preprocessing
│   └── data/         # Datasets (ML-CUP and MONK)
├── SYC-ML-CUP25-TS.csv # Final Blind Test Predictions
├── Report_Salmi_Yakoubi_Chafai.pdf # Detailed technical report
└── SYC_abstract.txt  # Project abstract
```

# 🗂 Datasets

## 1️⃣ MONK Problems

- 3 binary classification tasks
- Used for:
  - Verifying model correctness
  - Studying non-linear separability
  - Understanding inductive bias

Metrics:
- Accuracy
- MSE (for monitoring training)

---

## 2️⃣ ML-CUP 2025 Dataset

- 500 training samples
- 12 input features
- 4 continuous targets
- 1000 blind test samples

Task:
> Multi-output regression

Evaluation Metric:
> **Mean Euclidean Error (MEE)** computed in the original (unscaled) output space.

---

# 🔬 Validation Strategy

To ensure unbiased performance estimation:

### Step 1 – Hold-Out Split
- 90% → Development Set
- 10% → Internal Test Set (used only once)

### Step 2 – Model Selection
- 5-Fold Cross-Validation on Development Set
- Grid Search over hyperparameters
- Selection based on average Validation MEE

### Step 3 – Final Assessment
- Retrain best model on full Development Set
- Evaluate once on Internal Test Set

⚠️ The blind test set was **never used** for model selection.

---

# ⚙️ Models Compared

- Ridge Regression (Linear Baseline)
- Neural Network (MLP)
- Support Vector Machine (SVR)
- K-Nearest Neighbors (KNN)

---

# 🛠 Requirements

Python 3.10+ recommended.

Main dependencies:

```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```
