"""Tests for data_loader.py."""
import pytest
import pandas as pd
from pathlib import Path
from src.config import FEATURE_COLUMNS, TARGET_COLUMN, ALL_COLUMNS


class TestDatasetAvailability:
    def test_missing_dataset_raises(self, tmp_path, monkeypatch):
        """Loading a non-existent CSV must raise FileNotFoundError."""
        import src.data_loader as dl
        monkeypatch.setattr(dl, "DATASET_PATH", tmp_path / "nope.csv")
        monkeypatch.setattr(dl, "dataset_available",
                            lambda: False)
        with pytest.raises(FileNotFoundError, match="creditcard"):
            dl.load_dataset()

    def test_dataset_available_false_when_missing(self, tmp_path, monkeypatch):
        import src.data_loader as dl
        monkeypatch.setattr(dl, "DATASET_PATH", tmp_path / "nope.csv")
        monkeypatch.setattr(dl, "dataset_available",
                            lambda: (tmp_path / "nope.csv").exists())
        assert not dl.dataset_available()


class TestSchemaValidation:
    def test_missing_column_raises(self, fake_df, tmp_path, monkeypatch):
        """Drop a required column — loader must raise ValueError."""
        import src.data_loader as dl

        bad_df = fake_df.drop(columns=[TARGET_COLUMN])
        csv_path = tmp_path / "bad.csv"
        bad_df.to_csv(csv_path, index=False)

        monkeypatch.setattr(dl, "DATASET_PATH", csv_path)
        monkeypatch.setattr(dl, "dataset_available", lambda: True)

        with pytest.raises(ValueError, match="missing required columns"):
            dl.load_dataset()

    def test_invalid_target_values_raises(self, fake_df, tmp_path, monkeypatch):
        """Non-binary target values must raise ValueError."""
        import src.data_loader as dl

        bad_df = fake_df.copy()
        bad_df[TARGET_COLUMN] = 99   # invalid
        csv_path = tmp_path / "bad2.csv"
        bad_df.to_csv(csv_path, index=False)

        monkeypatch.setattr(dl, "DATASET_PATH", csv_path)
        monkeypatch.setattr(dl, "dataset_available", lambda: True)

        with pytest.raises(ValueError, match="unexpected values"):
            dl.load_dataset()

    def test_valid_fixture_columns(self, fake_df):
        """Synthetic fixture must have all required columns."""
        assert TARGET_COLUMN in fake_df.columns
        for col in FEATURE_COLUMNS:
            assert col in fake_df.columns


class TestDatasetSummary:
    def test_summary_keys(self, fake_df):
        from src.data_loader import get_dataset_summary
        s = get_dataset_summary(fake_df)
        assert s["total_transactions"] == len(fake_df)
        assert s["fraud_count"] == int(fake_df[TARGET_COLUMN].sum())
        assert s["legitimate_count"] == int((fake_df[TARGET_COLUMN] == 0).sum())
        assert 0 <= s["fraud_percentage"] <= 100
        assert "amount_stats" in s

    def test_summary_no_negative_counts(self, fake_df):
        from src.data_loader import get_dataset_summary
        s = get_dataset_summary(fake_df)
        assert s["fraud_count"] >= 0
        assert s["legitimate_count"] >= 0

    def test_sampled_flag(self, fake_df):
        from src.data_loader import get_dataset_summary
        s = get_dataset_summary(fake_df, sampled=True)
        assert s["is_sampled"] is True
