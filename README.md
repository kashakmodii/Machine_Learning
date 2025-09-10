---

# Machine Learning Projects Collection

This repository contains multiple machine learning projects, each addressing a real-world problem with structured datasets, preprocessing, modeling, and evaluation.

---

## 1. Credit Card Fraud Detection
- **Goal:** Classify transactions as fraudulent or legitimate.  
- **Dataset:** [Credit Card Dataset](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/creditcard.zip) (284,807 transactions, 492 frauds).  
- **Techniques:** Scaling, SMOTE for imbalance handling.  
- **Models:** Logistic Regression, Random Forest, SVM, KNN, etc.  
- **Key Insight:** Random Forest and SVM achieved strong performance; recall was prioritized to catch fraud cases.  

---

## 2. Online Shoppers Purchasing Intention
- **Goal:** Predict whether a user session leads to a purchase.  
- **Dataset:** [Online Shoppers Dataset](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/online_shoppers_intention.csv).  
- **Techniques:** PCA for dimensionality reduction, SMOTE for balancing.  
- **Models:** KNN (core), Logistic Regression, Decision Trees, Random Forest.  
- **Key Insight:** PCA improved efficiency; balancing enhanced rare purchase detection.  

---

## 3. Advertisement Budget Prediction
- **Goal:** Predict sales revenue from advertising budgets (TV, Radio, Newspaper).  
- **Dataset:** Classic Advertising dataset (200 entries).  
- **Techniques:** Exploratory analysis, feature scaling.  
- **Models:** Linear Regression, Polynomial Regression, Decision Tree, Random Forest.  
- **Key Insight:** TV and Radio strongly drive sales; Newspaper has minimal impact. Random Forest performed best.  

---

## 4. Diabetes Prediction
- **Goal:** Classify patients as diabetic or non-diabetic.  
- **Dataset:** [Diabetes Dataset](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/diabetes-data.csv) (768 records).  
- **Techniques:** Missing value imputation, scaling, SMOTE for imbalance.  
- **Models:** KNN (primary), Logistic Regression, Random Forest, SVM.  
- **Key Insight:** Glucose, BMI, and Age are strong predictors. Recall prioritized to avoid missing diabetic cases.  

---

## 5. Bike Sharing Demand Prediction
- **Goal:** Predict daily bike rentals using weather and seasonal data.  
- **Dataset:** [Bike Sharing Dataset](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/Data-Bike-Share.zip), file: `day.csv` (731 records).  
- **Techniques:** Label encoding, scaling, train-test split.  
- **Model:** Linear Regression.  
- **Key Insight:** Demand is higher in summer/fall, influenced positively by temperature and negatively by humidity/windspeed.  

---

## Technology Stack
- **Python**  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, imbalanced-learn  

---
