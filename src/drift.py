import os, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score

JAN_SAMPLES = os.getenv("JAN_SAMPLES", "data/jan_samples.csv")
FEB_SAMPLES = os.getenv("FEB_SAMPLES", "data/feb_samples.csv")
FEB_LABELS = os.getenv("FEB_LABELS", "data/feb_labels.csv")

class DriftDetector:
    @staticmethod
    def ks(jan, feb):
        return ks_2samp(jan, feb)  # (stat, pval)

def main():
    jan = pd.read_csv(JAN_SAMPLES).squeeze("columns")
    feb = pd.read_csv(FEB_SAMPLES).squeeze("columns")
    stat, p = DriftDetector.ks(jan, feb)
    print(f"KS stat: {stat:.4f}  p: {p:.4f}  Significant? {'Yes' if p<0.05 else 'No'}")

    if os.path.exists(FEB_LABELS):
        lab = pd.read_csv(FEB_LABELS)
        acc = balanced_accuracy_score(lab["y_true"], lab["y_pred"])*100
        jan_ref = float(os.getenv("BAL_ACC_JAN", "90"))
        print(f"Balanced accuracy Feb: {acc:.2f}%. Decline vs Jan ({jan_ref:.1f}%): {'Yes' if acc<jan_ref else 'No'}")

if __name__ == "__main__": main()
