# Heart Disease – End-to-End ML Pipeline

Production-minded binary classification pipeline:
- cleaning, imputation, standardization (no leakage)
- feature selection (RandomForest + SelectFromModel)
- model training (Logistic Regression; SVC baseline)
- evaluation (5-fold balanced accuracy, confusion matrix)
- MLflow logging of coefficients
- monitoring & drift detection (KS test)
- unit tests for inference
- containerization + EB deployment order

See `src/` for classes and `deployment/` for CI/EB notes.
