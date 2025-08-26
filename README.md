---

# ​ Credit Card Fraud Detection

---

##  Project Overview
This repository contains my credit card fraud detection project, developed by **kashakmodii**. It leverages machine learning techniques to classify credit card transactions as legitimate or fraudulent using a highly imbalanced dataset.


##  Problem Statement
Credit card fraud is a critical issue, with very few transactions being fraudulent. This project aims to build effective classifiers to detect such rare anomalies while minimizing false positives.


##  Dataset Details
- **Download link**: [dataset (ZIP)](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/creditcard.zip)
- Contains **284,807 transactions** collected over two days in September 2013 by European cardholders. Out of those, only **492 are fraud** (~0.17%) .
- Features:
  - **V1–V28**: PCA-transformed anonymized features
  - **Time**: Seconds elapsed from the first transaction
  - **Amount**: Transaction amount
  - **Class**: Target label (0 = Not fraud, 1 = Fraud) 


##  Exploratory Data Analysis (EDA)
- The dataset exhibits extreme class imbalance (~99.83% non-fraud, ~0.17% fraud), making metrics like **recall**, **precision**, **F1-score**, and **ROC-AUC** more meaningful than accuracy.
- Visual analyses (e.g., class distribution pie charts, amount box plots) reveal distinct patterns and justify scaling or log-transforming skewed numerical features.


### Preprocessing
- Applied **scaling** (e.g., StandardScaler or RobustScaler) to `Amount` and possibly `Time`
- Addressed class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique)

### Machine Learning Models Explored
- Logistic Regression
- LDA (Linear Discriminant Analysis)
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Model Evaluation
- Evaluated using **confusion matrix**, **accuracy**, **precision**, **recall**, **F1-score**, and visual plots
- Notable outcomes:
  - **Random Forest** and **SVM** delivered the highest accuracy
  - **Logistic Regression** and **LDA** offered better explainability while performing robustly
  - Emphasis placed on **recall**, ensuring high detection of fraudulent transactions


##  Final Insights
- All major steps—from data loading to modeling—were executed successfully
- Top-performing models: **Random Forest** and **SVM**
- Key insight: For fraud detection, a high **recall** with reasonable **precision** is more impactful than mere accuracy


##  Technology Stack
- **Python**
- **Pandas**, **NumPy**
- **scikit-learn**
- **Matplotlib**, **Seaborn**
- **imbalanced-learn** (for SMOTE and imbalance handling)

---
