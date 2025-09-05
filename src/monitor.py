import os, pandas as pd, matplotlib.pyplot as plt
JAN = os.getenv("JAN_LOGS", "data/logs_january.csv")
FEB = os.getenv("FEB_LOGS", "data/logs_february.csv")

class OutputMonitor:
    def plot_months(self, jan: pd.DataFrame, feb: pd.DataFrame):
        fig, ax = plt.subplots(1,2, figsize=(15,6))
        jan["target"].value_counts().plot(kind="bar", ax=ax[0]); ax[0].set_title("January"); ax[0].set_xlabel("Class"); ax[0].set_ylabel("Frequency")
        feb["target"].value_counts().plot(kind="bar", ax=ax[1]); ax[1].set_title("February"); ax[1].set_xlabel("Class"); ax[1].set_ylabel("Frequency")
        plt.tight_layout(); plt.show()

def main():
    OutputMonitor().plot_months(pd.read_csv(JAN), pd.read_csv(FEB))

if __name__ == "__main__": main()
