"""
test_pipeline.py
================
Unit and integration tests for the Bank Transaction Fraud Detection pipeline.
Validates data ingestion, feature generation, leak prevention, and model inference.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_data, validate_schema, generate_fraud_labels
from src.feature_engineering import FraudFeatureEngineer, prepare_train_test_pipeline


@pytest.fixture(scope="module")
def raw_dataframe():
    """Loads raw dataframe for testing."""
    df = load_raw_data()
    return df


@pytest.fixture(scope="module")
def labeled_dataframe(raw_dataframe):
    """Generates labeled dataset for testing."""
    df_labeled = generate_fraud_labels(raw_dataframe)
    return df_labeled


def test_raw_data_schema(raw_dataframe):
    """Validates raw data schema and column counts."""
    assert isinstance(raw_dataframe, pd.DataFrame)
    assert len(raw_dataframe) > 0
    assert validate_schema(raw_dataframe) is True


def test_fraud_label_properties(labeled_dataframe):
    """Validates properties of generated ground truth target labels."""
    assert "is_fraud" in labeled_dataframe.columns
    unique_vals = set(labeled_dataframe["is_fraud"].unique())
    assert unique_vals.issubset({0, 1})
    
    fraud_rate = labeled_dataframe["is_fraud"].mean()
    assert 0.01 <= fraud_rate <= 0.20, f"Unexpected fraud rate: {fraud_rate:.4f}"


def test_feature_engineering_pipeline(labeled_dataframe):
    """Ensures transformer correctly handles train/test partitions without NaNs."""
    X_train, X_test, y_train, y_test, engineer = prepare_train_test_pipeline(
        labeled_dataframe, test_size=0.2, random_state=42
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
    
    # Assert zero missing values after transformation
    assert X_train.isnull().sum().sum() == 0, "X_train contains NaN values!"
    assert X_test.isnull().sum().sum() == 0, "X_test contains NaN values!"

    # Key engineered features exist
    expected_engineered = ["amount_to_balance_ratio", "balance_drain_delta", "time_since_prev_txn_hours"]
    for col in expected_engineered:
        assert col in X_train.columns, f"Missing engineered feature: {col}"


def test_model_artifact_inference(labeled_dataframe):
    """Verifies that the serialized champion model loads and produces valid probabilities."""
    model_path = os.path.join(PROJECT_ROOT, "models", "champion_fraud_model.joblib")
    assert os.path.exists(model_path), "Model artifact does not exist!"

    bundle = joblib.load(model_path)
    assert "model" in bundle
    assert "feature_engineer" in bundle

    model = bundle["model"]
    engineer = bundle["feature_engineer"]

    # Transform a batch of test transactions
    drop_cols = ["is_fraud", "rule_overdraft", "rule_brute_force", "anomaly_iso_forest"]
    sample_df = labeled_dataframe.drop(columns=[c for c in drop_cols if c in labeled_dataframe.columns]).head(10)
    X_features = engineer.transform(sample_df)

    probs = model.predict_proba(X_features)[:, 1]
    assert len(probs) == 10
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities out of bounds [0, 1]!"
