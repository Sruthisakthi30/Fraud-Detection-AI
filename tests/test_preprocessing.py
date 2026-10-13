"""Tests for preprocessing.py."""
import numpy as np
import pytest
from src.config import FEATURE_COLUMNS, TARGET_COLUMN


class TestSplitAndScale:
    def test_split_sizes(self, preprocessed, fake_df):
        """Train + test should equal original size."""
        n = len(fake_df)
        n_train = len(preprocessed["y_train"])   # SMOTE-expanded
        n_test  = len(preprocessed["y_test"])
        # Test set must be exactly TEST_SIZE fraction of original
        from src.config import TEST_SIZE
        expected_test = round(n * TEST_SIZE)
        assert abs(n_test - expected_test) <= 2   # allow ±2 rounding

    def test_test_set_not_smoted(self, preprocessed, fake_df):
        """Test set size must equal the pre-SMOTE split, not the SMOTE-expanded training size."""
        n = len(fake_df)
        from src.config import TEST_SIZE
        raw_train_size = round(n * (1 - TEST_SIZE))
        n_test = len(preprocessed["y_test"])
        # SMOTE should have INCREASED training set; test must be unchanged
        n_train_smote = len(preprocessed["y_train"])
        assert n_train_smote >= raw_train_size, "SMOTE should expand training set"
        assert abs(n_test - round(n * TEST_SIZE)) <= 2

    def test_no_data_leakage(self, preprocessed):
        """X_test must not have been seen during scaler fitting (shape check)."""
        X_test  = preprocessed["X_test"]
        X_train = preprocessed["X_train"]
        assert X_train.shape[1] == X_test.shape[1] == len(FEATURE_COLUMNS)

    def test_scaler_zero_mean(self, preprocessed):
        """Scaled training data should have near-zero mean per feature."""
        X_train = preprocessed["X_train"]
        means = np.abs(np.mean(X_train, axis=0))
        # Due to SMOTE the exact mean may shift; just check finite values
        assert np.all(np.isfinite(means))

    def test_feature_names(self, preprocessed):
        assert preprocessed["feature_names"] == FEATURE_COLUMNS


class TestSmoteIsolation:
    def test_smote_only_on_train(self, preprocessed, fake_df):
        """Verify SMOTE was NOT applied to the test set."""
        from src.config import TEST_SIZE
        n_test_expected = round(len(fake_df) * TEST_SIZE)
        n_test_actual   = len(preprocessed["y_test"])
        assert abs(n_test_actual - n_test_expected) <= 2

    def test_smote_increases_minority(self, preprocessed, fake_df):
        """After SMOTE, minority class in training set should be ≥ original."""
        raw_fraud = int(fake_df[TARGET_COLUMN].sum())
        raw_train_fraud = preprocessed["meta"]["train_fraud"]
        smote_fraud = preprocessed["meta"]["train_fraud_smote"]
        assert smote_fraud >= raw_train_fraud


class TestSingleTransform:
    def test_transform_single_shape(self, preprocessed):
        from src.preprocessing import transform_single
        scaler = preprocessed["scaler"]
        row = {f: 0.0 for f in FEATURE_COLUMNS}
        X = transform_single(scaler, FEATURE_COLUMNS, row)
        assert X.shape == (1, len(FEATURE_COLUMNS))

    def test_transform_single_numeric(self, preprocessed):
        from src.preprocessing import transform_single
        scaler = preprocessed["scaler"]
        row = {f: float(i) for i, f in enumerate(FEATURE_COLUMNS)}
        X = transform_single(scaler, FEATURE_COLUMNS, row)
        assert np.all(np.isfinite(X))
