import argparse
import json
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES, load_data

def train(data_path: str, output: str = "artifacts", split: str = "temporal") -> dict:
    df = load_data(data_path).drop_duplicates().reset_index(drop=True)
    if split == "temporal":
        df = df.sort_values("Time", kind="stable")
        cutoff = int(len(df) * 0.8)
        tr, te = df.iloc[:cutoff], df.iloc[cutoff:]
    elif split == "stratified":
        tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Class"])
    else:
        raise ValueError("split must be temporal or stratified")
    if tr["Class"].nunique() < 2 or te["Class"].nunique() < 2:
        raise ValueError("Both partitions must contain fraud and legitimate samples")
    X_train, y_train = tr[FEATURES], tr["Class"]
    X_test, y_test = te[FEATURES], te["Class"]
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        predictions = (scores >= 0.5).astype(int)
        metrics[name] = {
            "pr_auc": float(average_precision_score(y_test, scores)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
        joblib.dump(model, out / f"{name}.joblib")
    result = {"source": str(data_path), "split": split, "train_rows": len(tr), "test_rows": len(te), "train_frauds": int(y_train.sum()), "test_frauds": int(y_test.sum()), "models": metrics}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser(description="Train FraudGuard on REAL ULB credit-card data")
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--split", choices=["temporal", "stratified"], default="temporal")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output, args.split), indent=2))

if __name__ == "__main__":
    main()
