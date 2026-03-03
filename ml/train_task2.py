from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ml.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_RISK_THRESHOLDS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from app.ml.feature_engineering import (
    CustomerObservation,
    TicketObservation,
    build_feature_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 2 churn model.")
    parser.add_argument(
        "--data-path",
        default=r"C:\Users\vvkou\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to Telco Customer Churn CSV file.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/churn_model.joblib",
        help="Path to save trained model artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/training_metrics.json",
        help="Path to save training metrics JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_telco_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def simulate_previous_charges(
    monthly_charges: pd.Series,
    churn_target: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    drift = rng.normal(loc=np.where(churn_target == 1, 6.0, 1.5), scale=3.5)
    previous = np.clip(monthly_charges.to_numpy(dtype=float) - drift, a_min=0.0, a_max=None)
    return previous


def simulate_tickets(
    churn_label: int,
    rng: np.random.Generator,
    reference_time: datetime,
) -> list[TicketObservation]:
    avg_tickets = 9 if churn_label == 1 else 3
    ticket_count = int(min(30, rng.poisson(avg_tickets)))

    if ticket_count == 0:
        return []

    if churn_label == 1:
        category_probs = [0.45, 0.20, 0.20, 0.15]
        sentiment_mean = -0.35
        recency_scale = 18
    else:
        category_probs = [0.10, 0.35, 0.25, 0.30]
        sentiment_mean = 0.25
        recency_scale = 40

    categories = ["complaint", "billing", "technical", "general"]
    tickets: list[TicketObservation] = []

    for _ in range(ticket_count):
        days_ago = int(min(180, rng.exponential(scale=recency_scale)))
        category = str(rng.choice(categories, p=category_probs))
        is_complaint = category == "complaint" or bool(
            rng.random() < (0.18 if churn_label == 1 else 0.03)
        )
        sentiment = float(np.clip(rng.normal(sentiment_mean, 0.35), -1.0, 1.0))

        tickets.append(
            TicketObservation(
                opened_at=reference_time - timedelta(days=days_ago),
                category=category,
                is_complaint=is_complaint,
                sentiment_score=sentiment,
            )
        )
    return tickets


def build_training_frame(
    df: pd.DataFrame,
    previous_charges: np.ndarray,
    churn_target: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    reference_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for i, row in df.iterrows():
        customer = CustomerObservation(
            customer_id=str(row["customerID"]),
            contract_type=str(row["Contract"]),
            monthly_charges_current=float(row["MonthlyCharges"]),
            monthly_charges_previous=float(previous_charges[i]),
            tenure_months=int(row["tenure"]),
            total_charges=float(row["TotalCharges"]),
            internet_service=str(row["InternetService"]),
            payment_method=str(row["PaymentMethod"]),
        )
        tickets = simulate_tickets(int(churn_target.iloc[i]), rng, reference_time)
        feature_row = build_feature_vector(customer, tickets, reference_time=reference_time)
        rows.append(feature_row)

    feature_df = pd.DataFrame(rows, columns=MODEL_FEATURE_COLUMNS)
    return feature_df


def train_model(
    x: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[Pipeline, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        random_state=seed,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(auc(recall, precision)),
    }
    return pipeline, metrics


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_telco_dataframe(data_path)
    target = (df["Churn"].str.strip().str.lower() == "yes").astype(int)
    prev_charges = simulate_previous_charges(df["MonthlyCharges"], target, rng)
    features = build_training_frame(df, prev_charges, target, rng)

    model, metrics = train_model(features, target, args.seed)

    artifact = {
        "pipeline": model,
        "metadata": {
            "model_version": "task2-logreg-v1",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "risk_thresholds": DEFAULT_RISK_THRESHOLDS,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "dataset_rows": int(len(df)),
            "positive_class_rate": float(target.mean()),
        },
    }
    joblib.dump(artifact, model_path)

    report = {
        "model_path": str(model_path),
        "metrics": metrics,
        "seed": args.seed,
        "data_path": str(data_path),
        "dataset_rows": int(len(df)),
        "positive_class_rate": float(target.mean()),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training completed.")
    print(f"Model artifact: {model_path}")
    print(f"Metrics report: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
