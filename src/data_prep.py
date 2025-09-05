from __future__ import annotations
import pandas as pd
from sklearn.impute import KNNImputer
from typing import Iterable, List

class DataPreprocessor:
    """Cleans, imputes, and returns a numeric-safe DataFrame."""
    def __init__(self, drop_cols: Iterable[str] | None = None, knn_neighbors: int = 2, impute_restecg: bool = True):
        self.drop_cols = list(drop_cols or [])
        self.knn_neighbors = knn_neighbors
        self.impute_restecg = impute_restecg
        self.imputer: KNNImputer | None = None
        self.numeric_cols: List[str] | None = None

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        if self.drop_cols:
            df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")
        if self.impute_restecg and "restecg" in df.columns:
            df["restecg"] = df["restecg"].fillna(df["restecg"].mean())
        return df

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        df = self._basic_clean(df)
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.imputer = KNNImputer(n_neighbors=self.knn_neighbors, weights="uniform")
        if self.numeric_cols:
            self.imputer.fit(df[self.numeric_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._basic_clean(df.copy())
        if self.numeric_cols:
            df.loc[:, self.numeric_cols] = self.imputer.transform(df[self.numeric_cols])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

def split_xy(df: pd.DataFrame, target: str = "target"):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
