"""
feature_engineering.py
======================
Extracts domain-specific financial features, performs leak-free encoding,
and generates stratified train/test partitions for fraud modeling.
"""

import os
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FraudFeatureEngineer:
    """
    Feature engineering transformer designed for financial transaction telemetry.
    Ensures zero data leakage by computing all statistical baselines solely on train data.
    """

    def __init__(self):
        self.freq_encodings: Dict[str, pd.Series] = {}
        self.categorical_columns = ["TransactionType", "Channel", "CustomerOccupation"]
        self.one_hot_columns: List[str] = []
        self.fitted = False

    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts mathematical ratios, temporal metrics, and behavioral flags."""
        data = df.copy()

        # 1. Financial Liquidity Ratios
        data["amount_to_balance_ratio"] = data["TransactionAmount"] / (data["AccountBalance"] + 1.0)
        data["log_amount"] = np.log1p(np.maximum(0, data["TransactionAmount"]))
        data["log_balance"] = np.log1p(np.maximum(0, data["AccountBalance"]))
        data["balance_drain_delta"] = data["AccountBalance"] - data["TransactionAmount"]

        # 2. Transaction Duration Dynamics
        data["duration_per_dollar"] = data["TransactionDuration"] / (data["TransactionAmount"] + 1.0)
        data["is_rapid_transaction"] = (data["TransactionDuration"] < 30).astype(int)

        # 3. Temporal & Velocity Features
        t_current = pd.to_datetime(data["TransactionDate"], errors="coerce")
        t_prev = pd.to_datetime(data["PreviousTransactionDate"], errors="coerce")
        
        # Absolute hours elapsed between events
        time_diff_hours = (t_current - t_prev).abs().dt.total_seconds() / 3600.0
        data["time_since_prev_txn_hours"] = time_diff_hours.fillna(time_diff_hours.median())
        data["is_rapid_succession"] = (data["time_since_prev_txn_hours"] < 1.0).astype(int)

        data["txn_hour"] = t_current.dt.hour.fillna(12).astype(int)
        data["is_night_txn"] = ((data["txn_hour"] < 6) | (data["txn_hour"] >= 23)).astype(int)
        data["txn_dayofweek"] = t_current.dt.dayofweek.fillna(0).astype(int)
        data["is_weekend"] = (data["txn_dayofweek"] >= 5).astype(int)

        # 4. Identity & Risk Flags
        data["login_attempts_raw"] = data["LoginAttempts"]
        data["customer_age"] = data["CustomerAge"]
        data["is_high_risk_channel"] = (data["Channel"] == "Online").astype(int)

        return data

    def fit(self, X_train: pd.DataFrame) -> "FraudFeatureEngineer":
        """Computes reference frequency encodings and one-hot categories from training data."""
        data = self._extract_base_features(X_train)

        # Frequency encodings for high-cardinality categorical entities
        for col in ["Location", "MerchantID", "DeviceID", "IP Address"]:
            freq_series = data[col].value_counts(normalize=True)
            self.freq_encodings[col] = freq_series

        # Determine one-hot dummy columns from training categories
        dummies = pd.get_dummies(data[self.categorical_columns], drop_first=True, dtype=int)
        self.one_hot_columns = dummies.columns.tolist()

        self.fitted = True
        logger.info(f"Feature engineer fitted with {len(self.one_hot_columns)} dummy features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations using training statistics without leakage."""
        if not self.fitted:
            raise RuntimeError("FraudFeatureEngineer must be fitted on training data before transforming.")

        data = self._extract_base_features(X)

        # Apply frequency encodings (default 0.0 for unseen categories)
        for col, freq_map in self.freq_encodings.items():
            data[f"{col.lower().replace(' ', '_')}_freq"] = data[col].map(freq_map).fillna(0.0)

        # Apply One-Hot Encoding aligned to training columns
        dummies = pd.get_dummies(data[self.categorical_columns], drop_first=True, dtype=int)
        for col in self.one_hot_columns:
            if col in dummies.columns:
                data[col] = dummies[col]
            else:
                data[col] = 0

        # Select final numerical and encoded features for machine learning
        feature_cols = [
            "TransactionAmount", "AccountBalance", "TransactionDuration", "CustomerAge",
            "amount_to_balance_ratio", "log_amount", "log_balance", "balance_drain_delta",
            "duration_per_dollar", "is_rapid_transaction", "time_since_prev_txn_hours",
            "is_rapid_succession", "txn_hour", "is_night_txn", "txn_dayofweek", "is_weekend",
            "login_attempts_raw", "is_high_risk_channel",
            "location_freq", "merchantid_freq", "deviceid_freq", "ip_address_freq"
        ] + self.one_hot_columns

        return data[feature_cols]

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fits on training data and transforms it in one step."""
        return self.fit(X_train).transform(X_train)


def prepare_train_test_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FraudFeatureEngineer]:
    """
    Executes stratified split and leak-free feature engineering.
    Returns:
        X_train, X_test, y_train, y_test, fitted_engineer
    """
    if "is_fraud" not in df.columns:
        raise ValueError("DataFrame must contain target column 'is_fraud'.")

    # Exclude internal rule audit columns from feature training
    drop_cols = ["is_fraud", "rule_overdraft", "rule_brute_force", "anomaly_iso_forest"]
    feature_source = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["is_fraud"]

    logger.info(f"Performing Stratified {int((1-test_size)*100)}/{int(test_size*100)} train/test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        feature_source,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    engineer = FraudFeatureEngineer()
    X_train = engineer.fit_transform(X_train_raw)
    X_test = engineer.transform(X_test_raw)

    logger.info(
        f"Pipeline preparation complete:\n"
        f"  X_train shape: {X_train.shape} (Frauds: {y_train.sum()} / {y_train.mean()*100:.2f}%)\n"
        f"  X_test shape: {X_test.shape} (Frauds: {y_test.sum()} / {y_test.mean()*100:.2f}%)\n"
        f"  Feature count: {X_train.shape[1]}"
    )

    return X_train, X_test, y_train, y_test, engineer


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_raw_data, generate_fraud_labels
    df = load_raw_data()
    df_labeled = generate_fraud_labels(df)
    X_train, X_test, y_train, y_test, engineer = prepare_train_test_pipeline(df_labeled)
    print("Sample feature columns:", X_train.columns.tolist()[:10])
