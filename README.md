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

## 6. Food Delivery – Customer Segmentation
- **Goal:** Segment food delivery app users into meaningful groups for personalized marketing.  
- **Dataset:** [Food Delivery Dataset](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/food_delivery.csv) (user demographics, spending habits, app usage, and order patterns).  
- **Techniques:** Data preprocessing, scaling, PCA for dimensionality reduction, clustering (K-Means, Agglomerative, DBSCAN).  
- **Models:** K-Means, Hierarchical Clustering, DBSCAN.  
- **Key Insight:** PCA helped visualize distinct customer segments. Clusters revealed groups such as *young frequent users*, *older high spenders*, and *occasional low spenders*. These insights can drive personalized offers, loyalty programs, and better customer engagement.  

---

## 7. Health Insurance Cost Prediction
- **Goal:** Predict individual medical charges using demographic and lifestyle data.  
- **Dataset:** [Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (`insurance.csv`, 1,338 records).  
- **Techniques:** Feature engineering (encoding categorical variables, correlation analysis), multicollinearity check with **VIF**, regression analysis.  
- **Model:** **Ordinary Least Squares (OLS) Regression**.  
- **Key Insight:** Smoking, BMI, and Age are the strongest predictors of medical costs. Children and region have relatively minor effects. Multicollinearity was handled with VIF and correlation filtering.  

---

## 8. Iris Flower Classification
- **Goal:** Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal dimensions.  
- **Dataset:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) (150 records, 4 features, 3 classes).  
- **Techniques:** Scaling, train-test split, adding random noise to data, exploratory visualization, cross-validation (CV).  
- **Models:** Logistic Regression, Decision Tree.  
- **Key Insight:** Petal length and width are the most discriminative features. Logistic Regression and Decision Tree with cross-validation achieved high accuracy (>95%). Adding random noise helped evaluate model robustness.  

---

## 9. House Price Prediction
- **Goal:** Predict the selling price of houses based on key features such as area, number of rooms, age, and distance from the city center.  
- **Dataset:** Synthetic dataset of 10,000 records containing attributes — `square_feet`, `num_rooms`, `age`, `distance_to_city(km)`, and `price`.  
- **Techniques:** Data cleaning (removing invalid/negative prices), exploratory data analysis (correlation heatmap, scatterplots), and feature scaling.  
- **Models:** Linear Regression (baseline), Random Forest Regressor for improved accuracy.  
- **Key Insight:**  
  - **Square footage** and **number of rooms** were the most influential factors increasing house prices.  
  - **Age** and **distance from city** negatively impacted price.  
  - Random Forest outperformed Linear Regression with higher **R²** and lower **RMSE**, indicating better predictive power.  

---

## 10. Ridge & Lasso Regression
- **Goal:** Understand the effect of regularization in regression and control overfitting by tuning the hyperparameter **λ (alpha)**.  
- **Dataset:** Numerical regression dataset used to demonstrate the impact of regularization on model performance and coefficients.  
- **Techniques:** Train-test split, feature scaling, and cross-validation for selecting the optimal alpha value.  
- **Models:** Linear Regression (baseline), Ridge Regression, Lasso Regression.  
- **Key Insight:**  
  - Ridge Regression reduces overfitting by shrinking coefficients while keeping all features.  
  - Lasso Regression performs feature selection by shrinking some coefficients to zero.  
  - Cross-validation helps choose the best alpha value, improving model generalization and stability.  

---
