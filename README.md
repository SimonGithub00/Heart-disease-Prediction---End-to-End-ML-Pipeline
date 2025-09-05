# Heart Disease – End-to-End ML Pipeline

This repo contains an end-to-end machine-learning workflow for heart-disease prediction, from data prep all the way to monitoring and drift checks. It’s deliberately “production-minded”: tests, MLflow logging, (optional) Feast feature store definitions, containerization, and deployment steps are included.

## Scenario

You’re a data scientist building a binary classifier (heart disease: yes/no).  
The goal is to:
- clean and prepare tabular patient data,
- engineer/select useful features,
- train and evaluate robust models,
- package and (optionally) deploy them,
- monitor outputs over time and detect data drift.

## Pipeline (what we implemented)

1. **Problem & data** – Supervised binary classification on the UCI-style heart dataset (`target` column).
2. **Cleaning** – remove duplicates, drop empty columns, and impute missing values (mean and/or KNN).
3. **EDA** – quick histograms (e.g., `chol`, `age`).
4. **Feature engineering** – standardize/normalize selected columns without data leakage.
5. **Feature selection** – `SelectFromModel` with a trained `RandomForestClassifier`.
6. **Training & evaluation** – Logistic Regression (main) and SVC as baselines; 5-fold cross-validation with **balanced accuracy**, confusion matrix, and per-fold scores.
7. **Tracking** – log model coefficients/intercept with **MLflow**.
8. **Feature store** – example Feast `Entity`, `Field`, `FileSource`, and `FeatureView`.
9. **Monitoring** – compare distributions of monthly predictions (January vs February).
10. **Drift detection** – Kolmogorov–Smirnov (KS) test between monthly samples.
11. **Feedback loop** – compare month-over-month balanced accuracy; flag declines.
12. **Testing** – `unittest` to ensure binary predictions only.
13. **Packaging & deploy** – Dockerfile and Elastic Beanstalk command order; sample CI.

## Results (example outcomes)

- 5-fold **balanced accuracy** typically around the low-to-mid 0.80s (varies by seed/data).
- Stable per-fold variance; confusion matrix confirms sensible precision/recall trade-offs.
- Monitoring + KS test helps catch input drift before performance slides.
- Simple unit tests guard against non-binary outputs after model or pipeline changes.



