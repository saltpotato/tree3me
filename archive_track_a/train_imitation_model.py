import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

CSV_PATH = "training_data.csv"
MODEL_PATH = "imitation_model.joblib"


FEATURE_COLUMNS = [
    "history_len",
    "history_min_size",
    "history_max_size",
    "history_sum_size",
    "history_max_height",
    "history_label1_count",

    "candidate_size",
    "candidate_height",
    "candidate_leaf_count",
    "candidate_max_branching",
    "candidate_total_branching",
    "candidate_root_label",
    "candidate_label1_count",
    "candidate_label2_count",
    "candidate_label3_count",
    "candidate_score",
]


def main():
    df = pd.read_csv(CSV_PATH)

    # For clean imitation, only imitate the pure heuristic policy.
    df = df[df["agent"] == "heuristic"].copy()

    if df.empty:
        raise RuntimeError("No rows found for agent == 'heuristic'.")

    X = df[FEATURE_COLUMNS]
    y = df["chosen"].astype(int)

    print("rows:", len(df))
    print("chosen rows:", int(y.sum()))
    print("not chosen rows:", int((1 - y).sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print()
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    print()
    print("feature importances:")
    for name, importance in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{name:30s} {importance:.4f}")

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )

    print()
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()