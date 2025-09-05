from __future__ import annotations
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from data_prep import DataPreprocessor, split_xy
from features import FeatureSelector

DATA_PATH = os.getenv("HEART_DATA", "data/heart.csv")
ART_DIR = "artifacts"

class ModelTrainer:
    def __init__(self):
        self.scaler: StandardScaler | None = None
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.selected_cols: list[str] | None = None

    def _scale_age(self, X, fit: bool):
        if "age" in X.columns:
            if self.scaler is None:
                self.scaler = StandardScaler()
            if fit:
                X["age"] = self.scaler.fit_transform(X["age"].values.reshape(-1,1))
            else:
                X["age"] = self.scaler.transform(X["age"].values.reshape(-1,1))
        return X

    def fit(self, X_train, y_train):
        X_train = self._scale_age(X_train.copy(), fit=True)
        self.model.fit(X_train, y_train)
        self.selected_cols = X_train.columns.tolist()
        return self

    def predict(self, X):
        X = X[self.selected_cols].copy()
        X = self._scale_age(X, fit=False)
        return self.model.predict(X)

    def save(self, out_dir=ART_DIR):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.model, f"{out_dir}/model.joblib")
        joblib.dump(self.selected_cols, f"{out_dir}/feature_names.joblib")
        if self.scaler: joblib.dump(self.scaler, f"{out_dir}/age_scaler.joblib")

def main():
    df = pd.read_csv(DATA_PATH)

    # preprocessing
    prep = DataPreprocessor(drop_cols=["oldpeak"], knn_neighbors=2, impute_restecg=True)
    df = prep.fit_transform(df)

    X, y = split_xy(df, target="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # feature selection
    fs = FeatureSelector(max_depth=5).fit(X_train, y_train)
    X_train_sel = fs.transform(X_train); X_test_sel = X_test[fs.selected_features]

    # model
    trainer = ModelTrainer().fit(X_train_sel, y_train)
    y_pred = trainer.predict(X_test_sel)
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100
    print(f"Balanced accuracy (test): {bal_acc:.2f}%")
    trainer.save()

if __name__ == "__main__":
    main()
