# Employee Attrition Prediction System

## 📌 Project Overview
This project aims to predict **employee attrition (whether an employee will leave the company or not)** using Machine Learning techniques.  
The system analyzes employee demographic, job-related, and performance data to identify employees with **high attrition risk**.

The project includes:
- Data exploration & visualization
- Data preprocessing
- Handling class imbalance
- Training multiple ML models
- Model evaluation & comparison
- Ensemble learning
- Saving the final trained model

---

## 🧠 Problem Statement
Employee attrition is a major challenge for companies as it increases recruitment costs and reduces productivity.  
This project helps HR departments:
- Identify employees at risk of leaving
- Understand key factors influencing attrition
- Take proactive retention actions

---

## 📊 Dataset
- **Dataset Name:** IBM HR Analytics Employee Attrition Dataset  
- **Source:** Kaggle / IBM Sample Dataset  
- **Target Variable:** `Attrition`
  - `1` → Employee left
  - `0` → Employee stayed

---

## 🔍 Project Workflow

### 1️⃣ Data Exploration
- Dataset shape & info
- Statistical summary
- Categorical & numerical analysis
- Automated profiling using **ydata-profiling**

### 2️⃣ Exploratory Data Analysis (EDA)
- Univariate analysis
- Bivariate analysis
- Attrition distribution
- Insights based on:
  - Age
  - Gender
  - Job Role
  - Work-life balance
  - Monthly income
  - Business travel

---

## 🛠 Data Preprocessing
- Dropping useless columns
- Feature scaling using **StandardScaler**
- Encoding categorical variables:
  - Binary Encoding
  - One-Hot Encoding
- Handling class imbalance using **SMOTE**
- Removing multicollinearity

---

## 🤖 Machine Learning Models
The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- Ensemble Model (Soft Voting):
  - Logistic Regression
  - XGBoost

---

## 📈 Model Evaluation
Models were evaluated using:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Score
- Threshold tuning for better performance on imbalanced data

---

## 🏆 Best Model
- **Final Model:** Ensemble (Logistic Regression + XGBoost)
- **Technique:** Soft Voting
- **Saved As:**  
  ```bash
  ensemble_attrition_model.pkl
