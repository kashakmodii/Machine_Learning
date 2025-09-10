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

# Online Shoppers Purchasing Intention Prediction

---

This project applies **machine learning techniques** to predict whether users will complete a purchase during an online shopping session. By analyzing behavioral features such as time spent on product pages and traffic sources, the model aims to identify **high-purchase-intent sessions** that can help e-commerce platforms improve targeting and conversions.

---

## Dataset
- **Name:** Online Shoppers Purchasing Intention Dataset  
- **Download link**: [dataset (ZIP)](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/online_shoppers_intention.csv)
- **Description:** Contains session-based features like:
  - Administrative, Informational, and Product-related pages visited
  - Bounce rate, Exit rate, Page values
  - Special day indicators
  - Traffic sources (Direct, Search, Referral, etc.)
  - Target variable: **Revenue (purchase vs. no purchase)**

---

## Techniques Used
- **Preprocessing**
  - Feature Scaling → Standardization  
  - Dimensionality Reduction → PCA  
  - Class Balancing → SMOTE  

- **Models Implemented**
  - K-Nearest Neighbors (KNN) (Core Model)  
  - Logistic Regression  
  - Decision Trees  
  - Random Forests  

- **Concepts Explored**
  - Bias-Variance Tradeoff  
  - Curse of Dimensionality  
  - Impact of Dimensionality Reduction  
  - Effect of Data Balancing on Rare Events  

---

## Results & Insights
- KNN was optimized and compared against other models for classification performance.  
- Dimensionality reduction (PCA) improved training efficiency and mitigated the curse of dimensionality.  
- SMOTE balancing significantly improved accuracy for predicting **rare purchase events**.  
- The project demonstrated how machine learning can provide insights into:
  - Customer engagement patterns  
  - Personalized targeting strategies  
  - Increasing conversion rates  

---

##  Conclusion
This project highlights the power of machine learning in **digital commerce decision-making**. By leveraging session-based data and advanced ML techniques, e-commerce platforms can better understand customer behavior, identify high-purchase-intent sessions, and optimize marketing strategies to boost conversions.

---

# Advertisement Budget Prediction

---

##  Project Overview
It applies **machine learning regression techniques** to predict **sales revenue** based on advertising budget spent on different media platforms such as **TV, Radio, and Newspaper**.

---

##  Problem Statement
Businesses often allocate advertising budgets across multiple channels but struggle to understand the **impact of each channel on sales**.  
The goal of this project is to build predictive models that:  
- Estimate sales revenue from given advertisement budgets  
- Identify which advertisement medium contributes the most to sales  
- Help businesses **optimize budget allocation** effectively  

---

##  Dataset Details
- **Source**: Advertising dataset (commonly used for regression tasks)  
- **Observations**: 200 entries  
- **Features**:
  - **TV**: Advertising budget spent on TV (in thousands of dollars)  
  - **Radio**: Advertising budget spent on Radio (in thousands of dollars)  
  - **Newspaper**: Advertising budget spent on Newspaper (in thousands of dollars)  
- **Target**:
  - **Sales**: Sales revenue generated (in thousands of units)  

---

##  Exploratory Data Analysis (EDA)
- **Scatter plots** between Sales and each feature show a **strong positive correlation** with TV and Radio, but weaker with Newspaper.  
- **Correlation heatmap** reveals:  
  - TV & Radio have strong influence on Sales  
  - Newspaper shows weak correlation  
- Visualizations suggest **diminishing returns** for higher ad spending.  

---

### Preprocessing
- Checked for **missing values** (none found)  
- Scaled features where needed for regression models  
- Split dataset into **training (80%)** and **testing (20%)**  

---

### Machine Learning Models Explored
- **Simple Linear Regression** (using TV only)  
- **Multiple Linear Regression** (TV, Radio, Newspaper)  
- **Polynomial Regression**  
- **Decision Tree Regressor**  
- **Random Forest Regressor**

---

### Model Evaluation
Models were evaluated using:  
- **R² Score**  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  

Key outcomes:  
- **Multiple Linear Regression** performed well with TV and Radio as strong predictors  
- **Random Forest Regressor** provided the highest accuracy and lowest error  
- **Newspaper feature** contributed little to prediction and could be dropped without significant loss  

---

##  Final Insights
- **TV and Radio** budgets have the most significant impact on sales  
- **Newspaper advertising** has minimal effect and can be deprioritized  
- Best performing model: **Random Forest Regressor**  
- Businesses should prioritize **TV and Radio investments** for maximum sales impact  

---

##  Technology Stack
- **Python**  
- **Pandas**, **NumPy**  
- **scikit-learn**  
- **Matplotlib**, **Seaborn**  

---


#  Diabetes Prediction Project

---

##  Project Overview
 It leverages machine learning techniques—especially **K-Nearest Neighbors (KNN)**—to classify patients as diabetic or non-diabetic using real-world health data.  
The project covers the full pipeline: data cleaning, preprocessing, handling imbalance, modeling, and evaluation.

---

##  Problem Statement
Diabetes is a major global health concern, and early detection is vital for timely intervention.  
The aim of this project is to:
- Predict whether a person is diabetic based on available health indicators
- Handle class imbalance (fewer diabetic cases)
- Prioritize **high recall** to minimize false negatives and avoid missing at-risk individuals

---

##  Dataset Details
- **Download link**: [diabetes data (CSV)](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/diabetes-data.csv)  
- **Observations**: 768 patient records  
- **Features**:
  - **Pregnancies**: Number of times pregnant  
  - **Glucose**: Plasma glucose concentration  
  - **BloodPressure**: Diastolic blood pressure (mm Hg)  
  - **SkinThickness**: Triceps skinfold thickness (mm)  
  - **Insulin**: 2-Hour serum insulin (mu U/ml)  
  - **BMI**: Body mass index (weight in kg/(height in m)²)  
  - **DiabetesPedigreeFunction**: Family history function score  
  - **Age**: Patient’s age  
- **Target**:
  - **Outcome**: (0 = Non-diabetic, 1 = Diabetic)

---

##  Exploratory Data Analysis (EDA)
- The dataset displayed **imbalanced class distribution** (more non-diabetic than diabetic patients).  
- Several features—**Glucose**, **BloodPressure**, **SkinThickness**, **Insulin**, and **BMI**—had zero or implausible values.  
- Correlation analysis identified **Glucose**, **BMI**, and **Age** as strong predictors of diabetes.  
- Visualizations included:
  - **Histograms** for understanding feature distributions  
  - **Heatmaps** for correlation assessment  
  - **Pair plots** to observe feature interactions

---

### Preprocessing
- Replaced invalid zeros in medical attributes using **mean or median imputation**  
- Applied **StandardScaler** for normalization of feature scales  
- Split data into **80% training** and **20% testing** subsets  
- Handled class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique)

---

### Machine Learning Models Explored
- **K-Nearest Neighbors (KNN)** (primary model)
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Naive Bayes

---

### Model Evaluation
Evaluation metrics used:
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **Confusion Matrix**

Highlights:
- **KNN** performed well when used with scaling and SMOTE  
- **Random Forest** and **SVM** delivered strong predictive performance  
- **Logistic Regression** offered interpretability of feature effects  
- Emphasis was placed on **Recall** to ensure that diabetic cases are identified

---

##  Final Insights
- Key predictors: **Glucose level**, **BMI**, and **Age**  
- Best performing models: **Random Forest**, **SVM**, and **KNN** (with proper preprocessing)  
- Main takeaway: In medical diagnosis tasks, **maximizing recall** is more critical than optimizing for accuracy to reduce the risk of missed diagnoses

---

##  Technology Stack
- **Python**
- **Pandas**, **NumPy**  
- **scikit-learn**  
- **Matplotlib**, **Seaborn**  
- **imbalanced-learn** (SMOTE)

---

---

#  Bike Sharing Demand Prediction

---

##  Project Overview 
It leverages **Linear Regression** to predict the demand for shared bikes based on historical data, applying data visualization, feature scaling, and model evaluation techniques.

---

##  Problem Statement
The rapid growth of bike-sharing systems in urban areas creates a need for accurate **demand forecasting**.  
The objective of this project is to analyze the dataset and build a regression model that predicts the number of bikes required, enabling better resource management and customer satisfaction.

---

##  Dataset Details
- **Download link**: [Bike Sharing Dataset (ZIP)](https://samatrix-data.s3.ap-south-1.amazonaws.com/ML/Data-Bike-Share.zip)  
- File used: **`day.csv`**  
- The dataset contains **daily records** of bike rentals, weather, and seasonal information.  
- Number of records: **731 days** (2011–2012)  
- Features:
  - **dteday**: Date
  - **season**: 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall
  - **yr**: Year (0 = 2011, 1 = 2012)
  - **mnth**: Month (1–12)
  - **holiday**: Whether the day is a holiday
  - **weekday**: Day of the week
  - **workingday**: If day is neither weekend nor holiday
  - **weathersit**: Weather condition (1–4 categories)
  - **temp**: Normalized temperature (0–1 scale)
  - **atemp**: Normalized “feeling” temperature
  - **hum**: Normalized humidity
  - **windspeed**: Normalized windspeed
  - **casual**: Count of casual users
  - **registered**: Count of registered users
  - **cnt**: Total rentals (**target variable**)

---

##  Exploratory Data Analysis (EDA)
- Explored correlations between bike demand and weather/seasonal variables.  
- Visualizations such as **heatmaps, bar charts, and trend lines** were used.  
- Key insights:
  - Demand is higher in **summer and fall**, lower in winter.  
  - Rentals increase on **working days** with good weather.  
  - Temperature has a strong positive effect on demand.  
  - Humidity and windspeed negatively impact rentals.

---

### Preprocessing
- Converted categorical variables (e.g., season, weather) using **Label Encoding**.  
- Applied **StandardScaler** for continuous features like `temp`, `hum`, `windspeed`.  
- Split data into **training (70%)** and **testing (30%)** sets.

---

### Machine Learning Model
- Applied **Linear Regression** to predict `cnt` (total rentals).  

---

### Model Evaluation
- Evaluated model performance with:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score**
- The regression model showed:
  - **High R² score**, capturing strong relationship between predictors and demand.  
  - **Low MAE and RMSE**, confirming reliable forecasting ability.  

---

##  Final Insights
- Bike demand depends heavily on **season, weather, and temperature**.  
- Linear Regression works effectively for this dataset and provides interpretable coefficients.  
- This model can help bike-sharing companies **plan fleet sizes and optimize resources**.

---

##  Technology Stack
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **scikit-learn** (Linear Regression, preprocessing, metrics)

---

