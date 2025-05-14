# Insurance Benefit Analysis Project

## Table of Contents
1. [Introduction](#introduction)  
2. [Data Preprocessing & Exploration](#data-preprocessing--exploration)  
   1. [Loading and Cleaning Data](#loading-and-cleaning-data)  
   2. [Descriptive Statistics](#descriptive-statistics)  
   3. [Exploratory Data Analysis](#exploratory-data-analysis)  
3. [Task 1: Similar Customers (k‑Nearest Neighbors)](#task-1-similar-customers-k‑nearest-neighbors)  
   1. [Unscaled vs. Scaled Data](#unscaled-vs-scaled-data)  
   2. [Euclidean vs. Manhattan Distance](#euclidean-vs-manhattan-distance)  
4. [Task 2: Insurance Benefit Classification](#task-2-insurance-benefit-classification)  
   1. [Data Split & Baseline (Random) Model](#data-split--baseline-random-model)  
   2. [kNN Classifier Evaluation](#knn-classifier-evaluation)  
5. [Task 3: Regression (Linear Regression)](#task-3-regression-linear-regression)  
   1. [Analytical Solution & Implementation](#analytical-solution--implementation)  
   2. [Original vs. Scaled Data Comparison](#original-vs-scaled-data-comparison)  
6. [Task 4: Data Obfuscation](#task-4-data-obfuscation)  
   1. [Invertible Transformation](#invertible-transformation)  
   2. [Analytical Proof](#analytical-proof)  
   3. [Computational Verification](#computational-verification)  
7. [Conclusions](#conclusions)  
8. [Appendices](#appendices)  

---

## Introduction
This project demonstrates practical applications of linear algebra and machine learning on an insurance dataset. We perform similarity search, binary classification, regression, and data obfuscation to explore model behavior and data privacy.

## Data Preprocessing & Exploration

### Loading and Cleaning Data
- Loaded `insurance_us.csv`  
- Renamed columns to lowercase and corrected types  
- Ensured no missing values  
- Converted `age` to integer  

### Descriptive Statistics
- Used `df.describe()` to confirm reasonable ranges:  
  - Age: 18–65  
  - Income: 5,300–79,000  
  - Family members: 0–6  
  - Insurance benefits: 0–5 (median 0)

### Exploratory Data Analysis
- Pair plots showed no obvious clusters  
- Confirmed balanced gender and zero-inflated target  

## Task 1: Similar Customers (k‑Nearest Neighbors)

### Unscaled vs. Scaled Data
- **Unscaled**: Income dominated distances, poor neighbor selection  
- **Scaled**: Features on comparable scales → meaningful distances  

### Euclidean vs. Manhattan Distance
- Ordering of neighbors nearly identical after scaling  
- Absolute distance values differ but neighbor sets remain consistent  

## Task 2: Insurance Benefit Classification

### Data Split & Baseline (Random) Model
- Created binary target `benefit_received = (insurance_benefits > 0)`  
- 70 % train / 30 % test  
- Random baseline F1 scores:  
  - P=0 → F1=0.00  
  - P≈0.11 → F1≈0.12  
  - P=0.5 → F1≈0.20  
  - P=1 → F1≈0.20  

### kNN Classifier Evaluation
- **Unscaled Data**: best F1≈0.65 (k=1), then significant drop  
- **Scaled Data**: F1 consistently ≈0.92–0.95 for k 1…10  
- **Conclusion**: Scaling crucial for kNN; distance metric choice less critical once scaled  

## Task 3: Regression (Linear Regression)

### Analytical Solution & Implementation
- Derived closed‑form solution \( w = (X^T X)^{-1}X^T y \)  
- Implemented custom `MyLinearRegression` with intercept  

### Original vs. Scaled Data Comparison
- **Original**: RMSE = 0.34, R² = 0.66  
- **Scaled**: RMSE = 0.34, R² = 0.66  
- **Conclusion**: Scaling does not change LR quality when intercept handled properly  

## Task 4: Data Obfuscation

### Invertible Transformation
- Selected feature matrix \(X\) (gender, age, income, family_members)  
- Generated random invertible matrix \(P\) (checked \(\det(P)\neq0\))  
- Obfuscated via \(X' = X P\)

### Analytical Proof
- Showed \(w_P = P^{-1} w\) and \(\hat y_P = Xw\)  
- Concluded RMSE and predictions remain identical  

### Computational Verification
- Trained LR on original vs. obfuscated data  
- Both cases yielded RMSE = 0.36, R² = 0.65  
- Max prediction difference ≈ 4.4e−08  

## Conclusions
- **kNN**: Requires feature scaling; robust to distance metric after scaling  
- **Classification**: kNN outperforms random baseline significantly when scaled  
- **Regression**: Closed‑form LR invariant to scaling and invertible obfuscation  
- **Data Obfuscation**: Preserves model performance while protecting raw values  

## Appendices
- **Appendix A**: Glossary of terms (Euclidean & Manhattan distance, RMSE, R²)  
- **Appendix B**: Matrix identities and properties used in analytical proofs  
