from __future__ import annotations
import os, pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from data_prep import DataPreprocessor, split_xy

DATA_PATH = os.getenv("HEART_DATA", "data/heart.csv")

class Evaluator:
    def cv_scores(self, X, y, n_splits=5):
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cross_val_score(model, X, y, scoring="balanced_accuracy", cv=kf)

    def confusion(self, model, X_test, y_test):
        y_pred = model.predict(X_test); return confusion_matrix(y_test, y_pred)

def main():
    df = pd.read_csv(DATA_PATH)
    df = DataPreprocessor(drop_cols=["oldpeak"]).fit_transform(df)
    X, y = split_xy(df)
    ev = Evaluator()
    scores = ev.cv_scores(X, y, n_splits=5)
    print("Per-fold balanced accuracy:", scores, "\nMean:", scores.mean(), "Std:", scores.std())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    cm = ev.confusion(model, Xte, yte)
    print("Confusion matrix:\n", cm)
    print("Balanced accuracy (holdout):", balanced_accuracy_score(yte, model.predict(Xte)))

if __name__ == "__main__":
    main()
