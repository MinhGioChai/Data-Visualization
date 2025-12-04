<<<<<<< HEAD
# 📘 Credit Default Risk Prediction #
### The Power of Data Preparation – A Data Storytelling Approach

## 📍 Introduction

This project was conducted as part of the course Data Preparation and Visualization.

Its primary goal is to demonstrate that data quality matters more than model complexity in predicting credit default risk, especially in the context of microfinance, where clients often lack traditional credit histories.

🔑 Core Idea: “Garbage In, Garbage Out (GIGO)” — clean, well-prepared data determines the performance and reliability of any machine learning model.

## 🎯 Project Objectives

- Build a complete data preparation pipeline for credit default prediction.

- Identify and address issues in raw data such as missing values, outliers, high cardinality, skewness, and inconsistent scales.

- Compare the performance of models before and after data preparation.

Develop a prediction model that supports microfinance institutions in making safer and more responsible lending decisions.

## 📂 Dataset Overview

The dataset contains detailed information about borrowers, loan applications, repayment behavior, and external credit indicators.

##### Key feature groups include:

- Client Profile: gender, age, education, marital status, occupation, income, number of children, housing status, etc.

- Loan Details: contract type, credit amount, annuity, goods price, application time info.

- Behavioral Indicators: 30/60-day past-due history, social-circle default observations.

- External Sources: EXT_SOURCE_1/2/3 — normalized credit scores.

##### Target variable:

- 1 → client had payment difficulties (late by ≥ X days in the first Y installments)

- 0 → otherwise

## ⚠️ Challenges in Raw Data

Exploration of the unprocessed dataset reveals significant issues:

- Numerous missing values, especially in occupation and external score variables

- Strong right-skewness in income and credit-related fields

- High-cardinality categorical variables (50+ levels)

- Outliers affecting distribution and model stability

- Scale inconsistency across numerical features

#### **Baseline Models on Raw Data**

Three baseline models were trained without any cleaning or preprocessing:

| Model | ROC–AUC | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.6203 | 0.6097 | 0.0946 | 0.5700 | 0.1622 |

#### Key observations:

- Extremely low precision → high number of false alarms

- Model is unreliable for real-world lending decisions

## 🔧 Data Preparation Workflow
#### 1. Data Understanding

- Check distributions, correlations, missing patterns

- Identify outliers and skewness

#### 2. Data Cleaning

- Missing value imputation (mean/median/advanced methods)

- Outlier smoothing or capping

- Consistency checks for date and address variables

#### 3. Data Transformation

- Feature scaling (standardization/normalization)

- Log transformation for heavily skewed variables

#### 4. Feature Engineering

Examples:

- `CREDIT_INCOME_RATIO` = `AMT_CREDIT / AMT_INCOME_TOTAL`

- `ANNUITY_INCOME_RATIO` = `AMT_ANNUITY / AMT_INCOME_TOTAL`

- Age and employment duration converted from days to years

- One-hot or target encoding for categorical variables

#### 5. Dimensionality Reduction

- Remove redundant variables

- Reduce categorical complexity

## 📊 Results & Impact

After proper data preparation:

- Model accuracy and robustness significantly improved

- False positives reduced notably

- Predictive insights became much clearer

    - Microfinance institutions can:

    - Reduce lending risks

    - Avoid over-indebting vulnerable customers

    - Approve more suitable borrowers who lack traditional credit history

## 🚀 How to Run (Sample Workflow)

## 🧠 Key Takeaways

- Data preparation is more important than model sophistication

- Clean data → better predictions → safer lending decisions

- Good storytelling enhances the interpretability and credibility of data insights

- Thorough evaluation is essential before deploying credit risk models

## 👥 Project Members

- Nguyen Khanh Huyen

- Nguyen Thi Huong Giang

- Nguyen Dang Minh

- Nguyen Thi Ha Phuong

- Duong Thi Huyen Trang

Supervisor: Dr. Nguyen Tuan Long
=======

>>>>>>> 40d4c40552d401d62299bb30798501795149289a
