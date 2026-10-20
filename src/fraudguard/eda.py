import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import load_data

def analyze(data_path: str, output: str = "reports") -> dict:
    df = load_data(data_path)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    summary = {"rows": len(df), "fraud": int(counts[1]), "legitimate": int(counts[0]), "fraud_rate": float(counts[1] / len(df)), "duplicates": int(df.duplicated().sum()), "amount_by_class": {str(k): float(v) for k, v in df.groupby("Class")["Amount"].median().items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Legitimate", "Fraud"], counts.values)
    ax.set_yscale("log"); ax.set_ylabel("Transactions (log scale)"); ax.set_title("Class imbalance — ULB real transactions")
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=160); plt.close(fig)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    print(json.dumps(analyze(args.data, args.output), indent=2))
