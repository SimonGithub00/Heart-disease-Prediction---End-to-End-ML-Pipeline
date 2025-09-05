from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

class FeatureSelector:
    """RandomForest + SelectFromModel feature selection."""
    def __init__(self, max_depth: int = 5, random_state: int = 42):
        self.rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=max_depth, random_state=random_state)
        self.selector: SelectFromModel | None = None
        self.selected_features: list[str] | None = None

    def fit(self, X: pd.DataFrame, y) -> "FeatureSelector":
        self.rf.fit(X, y)
        self.selector = SelectFromModel(self.rf, prefit=True)
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selector is None:
            raise RuntimeError("Call fit first.")
        Xt = self.selector.transform(X)
        return pd.DataFrame(Xt, columns=self.selected_features, index=X.index)
