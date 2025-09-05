import unittest, numpy as np, joblib, pandas as pd

class TestModelInference(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load("artifacts/model.joblib")
        self.features = joblib.load("artifacts/feature_names.joblib")
        df = pd.read_csv("data/heart.csv").drop(columns=["target"])
        self.X_test = df[self.features].head(32)

    def test_prediction_output_values(self):
        y_pred = self.model.predict(self.X_test.values)
        for v in np.unique(y_pred): self.assertIn(v, [0,1])

if __name__ == "__main__": unittest.main()
