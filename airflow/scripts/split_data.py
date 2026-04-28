"""
airflow/scripts/split_data.py
──────────────────────────────
Stage 4 of the DDD data pipeline.

Reads data/features/features.csv
Splits into stratified train/test sets (default 80/20).
Saves: data/processed/train.csv and data/processed/test.csv

Stratification is done on the label column to ensure both
drowsy and alert classes are represented in both splits.

Can be run:
  • Standalone:  python airflow/scripts/split_data.py
  • By Airflow:  PythonOperator in dag_data_pipeline.py
  • By DVC:      dvc repro split_data
"""

import json
import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("split_data")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "/app/data" if os.path.exists("/app/data") else "data"

FEATURES_CSV = os.getenv("FEATURES_CSV", os.path.join(DEFAULT_DATA_DIR, "features/features.csv"))
PROCESSED_DIR = os.getenv("PROCESSED_DIR", os.path.join(DEFAULT_DATA_DIR, "processed"))
TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def run_split(
    features_csv: str = FEATURES_CSV,
    processed_dir: str = PROCESSED_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Stratified train/test split.
    Returns summary dict for Airflow XCom.
    """
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.isfile(features_csv):
        logger.error(f"Features CSV not found: {features_csv}")
        sys.exit(1)

    df = pd.read_csv(features_csv)
    logger.info(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} cols")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Stratified split on label
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    summary = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_size": test_size,
        "random_state": random_state,
        "train_label_dist": train_df["label"].value_counts().to_dict(),
        "test_label_dist": test_df["label"].value_counts().to_dict(),
    }

    logger.info(
        f"\n{'='*50}\n"
        f"SPLIT COMPLETE\n"
        f"  Total    : {summary['total_samples']}\n"
        f"  Train    : {summary['train_samples']} — {summary['train_label_dist']}\n"
        f"  Test     : {summary['test_samples']} — {summary['test_label_dist']}\n"
        f"{'='*50}"
    )
    return summary


if __name__ == "__main__":
    summary = run_split()
    os.makedirs("reports", exist_ok=True)
    with open("reports/split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    sys.exit(0)
