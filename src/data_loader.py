"""
data_loader.py
==============
Handles dataset ingestion, schema validation, and ground-truth risk labeling
for the Bank Transaction Fraud Detection pipeline.
"""

import os
import shutil
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_DATASET_HANDLE = "valakhorasani/bank-transaction-dataset-for-fraud-detection"
RAW_FILENAME = "bank_transactions_data_2.csv"
EXPECTED_COLUMNS = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
    "TransactionType", "Location", "DeviceID", "IP Address", "MerchantID",
    "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance", "PreviousTransactionDate"
]


def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def download_or_locate_dataset(dest_dir: Optional[str] = None) -> str:
    """
    Downloads the dataset from Kaggle via kagglehub if not present in dest_dir,
    or copies it to the destination directory.
    """
    if dest_dir is None:
        dest_dir = os.path.join(get_project_root(), "data", "raw")
    os.makedirs(dest_dir, exist_ok=True)

    dest_file = os.path.join(dest_dir, RAW_FILENAME)
    if os.path.exists(dest_file):
        logger.info(f"Raw dataset already exists at: {dest_file}")
        return dest_file

    logger.info(f"Downloading dataset from Kaggle: {KAGGLE_DATASET_HANDLE}...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        logger.info(f"Kaggle download completed to cache: {download_path}")

        source_file = os.path.join(download_path, RAW_FILENAME)
        if not os.path.exists(source_file):
            # Search for any csv if exact filename changed
            csv_candidates = [f for f in os.listdir(download_path) if f.endswith(".csv")]
            if csv_candidates:
                source_file = os.path.join(download_path, csv_candidates[0])
            else:
                raise FileNotFoundError(f"No CSV file found in download path: {download_path}")

        shutil.copy2(source_file, dest_file)
        logger.info(f"Copied dataset to project raw directory: {dest_file}")
        return dest_file
    except Exception as e:
        logger.error(f"Failed to download or locate dataset: {e}")
        raise


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads the raw dataset into a pandas DataFrame."""
    if file_path is None:
        file_path = download_or_locate_dataset()
    logger.info(f"Loading raw transaction data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validates that all expected columns are present and checks missing values."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Detected null values in dataset:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("Schema validation passed: No missing values detected.")
    return True


def generate_fraud_labels(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Constructs an auditable, multi-vector fraud ground truth for bank transactions.
    
    Financial Risk Factors:
    1. Financial Exposure Rule: TransactionAmount > AccountBalance (overdraft/drain attempt).
    2. Account Security Rule: LoginAttempts >= 3 (credential stuffing / brute force).
    3. Behavioral Outlier: Multi-feature Isolation Forest (unusual combinations of amount, duration, and balance ratio).
    
    A transaction is flagged as fraud (is_fraud = 1) if it meets any of these risk indicators.
    """
    df = df.copy()

    # Vector 1: Overdraft / account drain
    rule_overdraft = (df["TransactionAmount"] > df["AccountBalance"]).astype(int)

    # Vector 2: High login failures
    rule_brute_force = (df["LoginAttempts"] >= 3).astype(int)

    # Vector 3: Multivariate Isolation Forest on core behavioral metrics
    iso_features = pd.DataFrame({
        "TransactionAmount": df["TransactionAmount"],
        "TransactionDuration": df["TransactionDuration"],
        "LoginAttempts": df["LoginAttempts"],
        "AccountBalance": df["AccountBalance"],
        "AmountToBalance": df["TransactionAmount"] / (df["AccountBalance"] + 1.0)
    })

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    iso_preds = iso.fit_predict(iso_features)
    anomaly_iso = (iso_preds == -1).astype(int)

    # Combined ground truth
    is_fraud = ((rule_overdraft == 1) | (rule_brute_force == 1) | (anomaly_iso == 1)).astype(int)

    df["rule_overdraft"] = rule_overdraft
    df["rule_brute_force"] = rule_brute_force
    df["anomaly_iso_forest"] = anomaly_iso
    df["is_fraud"] = is_fraud

    total = len(df)
    fraud_count = is_fraud.sum()
    fraud_rate = fraud_count / total * 100

    logger.info(
        f"Ground-truth generation complete:\n"
        f"  Total records: {total}\n"
        f"  Overdraft flags: {rule_overdraft.sum()}\n"
        f"  Brute force flags: {rule_brute_force.sum()}\n"
        f"  Isolation Forest anomalies: {anomaly_iso.sum()}\n"
        f"  Final Fraud count (Union): {fraud_count} ({fraud_rate:.2f}%)\n"
        f"  Legitimate count: {total - fraud_count}"
    )

    return df


def prepare_and_save_data(output_dir: Optional[str] = None) -> str:
    """Executes the full data loading and ground truth generation pipeline."""
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    df_raw = load_raw_data()
    validate_schema(df_raw)
    df_labeled = generate_fraud_labels(df_raw)

    output_path = os.path.join(output_dir, "bank_transactions_labeled.csv")
    df_labeled.to_csv(output_path, index=False)
    logger.info(f"Saved labeled dataset to: {output_path}")
    return output_path


if __name__ == "__main__":
    prepare_and_save_data()
