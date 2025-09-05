import joblib, mlflow
class MLflowLogger:
    def log_logreg(self, model):
        mlflow.set_experiment("Logistic Regression Heart Disease Prediction")
        with mlflow.start_run():
            for i, c in enumerate(model.coef_[0]): mlflow.log_param(f"coef_{i}", float(c))
            mlflow.log_param("intercept", float(model.intercept_[0]))
            return mlflow.active_run().info.run_id

def main():
    model = joblib.load("artifacts/model.joblib")
    run_id = MLflowLogger().log_logreg(model)
    print("MLflow run id:", run_id)

if __name__ == "__main__": main()
