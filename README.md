# Heart Disease Prediction — End-to-End ML Pipeline

I built an end-to-end, production-minded ML workflow to predict **heart disease (binary: yes/no)**.  
This project goes from **data preparation** → **feature engineering/selection** → **model training & evaluation** → **experiment tracking** → **(optional) feature store** → **monitoring & drift detection** → **testing** → **containerization & CI/CD deployment**.

---

## Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-1.26.4-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-2.2.2-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-3.8.4-11557c?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-1.11.4-8CAAE6?logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/joblib-1.4.2-005571" />
  <img src="https://img.shields.io/badge/MLflow-2.13.2-0194E2?logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Feast-Feature%20Store-8A2BE2" />
  <img src="https://img.shields.io/badge/pytest-8.2.1-0A9EDC?logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS%20Elastic%20Beanstalk-Deploy-FF9900?logo=amazonaws&logoColor=white" />
</p>

---

## What I built

- **Supervised binary classifier** on a heart-disease dataset (with a `target` column).
- **Data cleaning**: drop duplicates & empty columns, impute missing values (mean / KNN).
- **Feature engineering**: scale `age` **only on train** (no leakage), optional normalization.
- **Feature selection**: `SelectFromModel` using a `RandomForestClassifier`.
- **Training & evaluation**: Logistic Regression (main) + SVC baseline; 5-fold **balanced accuracy**, confusion matrix, per-fold scores.
- **Experiment tracking**: log LR coefficients & intercept with **MLflow**.
- **(Optional) Feature Store**: minimal **Feast** setup (Entity, FileSource, FeatureView).
- **Monitoring**: monthly prediction distributions (January vs. February).
- **Drift detection**: Kolmogorov–Smirnov (KS) test on sample distributions.
- **Feedback loop**: compare month-over-month balanced accuracy to flag declines.
- **Testing**: `unittest` to ensure **binary predictions** only.
- **Packaging & Deploy**: Docker image, local compose profiles, GitHub Actions CI/CD, and two realistic **AWS Elastic Beanstalk** paths.

---

## How I handle data leakage

- I **fit** the scaler for `age` **only on the training set** and then **transform** the test set with the **same** scaler.  
- **Feature selection** is fit on **training data** and consistently **applied to the test data**.

---

## Modeling details

- **Feature selection:** `RandomForestClassifier` + `SelectFromModel` to keep salient features.  
- **Main model:** `LogisticRegression(max_iter=1000, class_weight="balanced")`.  
- **Evaluation:** 5-fold **balanced accuracy** (robust to class imbalance) and a **holdout confusion matrix**.

---

## Results

- **5-fold balanced accuracy:** ~**0.80–0.85** (depends on seed/data splits).  
- **Confusion matrix** shows a reasonable precision/recall balance.  
- **KS test** quickly flags distribution shifts between months, and the **feedback loop** compared month-over-month balanced accuracy and caught performance decline.

