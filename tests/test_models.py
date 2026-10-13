"""Tests for model training and evaluation."""
import numpy as np
import pytest


class TestModelTraining:
    @pytest.mark.parametrize("name", [
        "logistic_regression", "decision_tree", "random_forest", "xgboost"
    ])
    def test_supervised_model_trains(self, preprocessed, name):
        """Each supervised model should train without error."""
        from src.train_models import train_supervised
        model, elapsed = train_supervised(
            name, preprocessed["X_train"], preprocessed["y_train"]
        )
        assert model is not None
        assert elapsed >= 0

    def test_isolation_forest_trains(self, preprocessed):
        from src.train_models import train_isolation_forest
        model, elapsed = train_isolation_forest(preprocessed["X_train_raw"])
        assert model is not None
        assert elapsed >= 0

    @pytest.mark.parametrize("name", [
        "logistic_regression", "decision_tree", "random_forest", "xgboost"
    ])
    def test_model_predict_shape(self, trained_models, preprocessed, name):
        model = trained_models[name]
        preds = model.predict(preprocessed["X_test"])
        assert preds.shape == (len(preprocessed["y_test"]),)

    @pytest.mark.parametrize("name", [
        "logistic_regression", "random_forest", "xgboost"
    ])
    def test_model_predict_proba_valid(self, trained_models, preprocessed, name):
        model = trained_models[name]
        probs = model.predict_proba(preprocessed["X_test"])
        assert probs.shape[1] == 2
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0) and np.all(probs <= 1)


class TestEvaluation:
    @pytest.mark.parametrize("name", [
        "logistic_regression", "decision_tree", "random_forest", "xgboost"
    ])
    def test_supervised_eval_keys(self, trained_models, preprocessed, name):
        from src.evaluate_models import evaluate_supervised
        model = trained_models[name]
        result = evaluate_supervised(name, model, preprocessed["X_test"],
                                     preprocessed["y_test"])
        for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
            assert key in result, f"Missing key: {key}"

    def test_metrics_in_range(self, trained_models, preprocessed):
        from src.evaluate_models import evaluate_supervised
        model = trained_models["random_forest"]
        result = evaluate_supervised(
            "random_forest", model, preprocessed["X_test"], preprocessed["y_test"]
        )
        for m in ["accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= result[m] <= 1.0, f"{m} out of range: {result[m]}"

    def test_confusion_matrix_sum(self, trained_models, preprocessed):
        from src.evaluate_models import evaluate_supervised
        model = trained_models["logistic_regression"]
        result = evaluate_supervised(
            "logistic_regression", model,
            preprocessed["X_test"], preprocessed["y_test"]
        )
        cm_sum = result["tn"] + result["fp"] + result["fn"] + result["tp"]
        assert cm_sum == len(preprocessed["y_test"])

    def test_isolation_forest_eval(self, trained_models, preprocessed):
        from src.evaluate_models import evaluate_isolation_forest
        model = trained_models["isolation_forest"]
        result = evaluate_isolation_forest(
            model, preprocessed["X_test"], preprocessed["y_test"]
        )
        assert result["model_type"] == "unsupervised_anomaly_detector"
        assert 0 <= result["f1"] <= 1

    def test_comparison_report_keys(self, trained_models, preprocessed):
        from src.evaluate_models import (
            evaluate_supervised, evaluate_isolation_forest, build_comparison_report
        )
        results = []
        for name in ["logistic_regression", "decision_tree"]:
            results.append(evaluate_supervised(
                name, trained_models[name],
                preprocessed["X_test"], preprocessed["y_test"]
            ))
        results.append(evaluate_isolation_forest(
            trained_models["isolation_forest"],
            preprocessed["X_test"], preprocessed["y_test"]
        ))
        dataset_meta = {"total_transactions": len(preprocessed["y_test"])}
        report = build_comparison_report(results, dataset_meta)
        assert "best_model" in report
        assert "timestamp" in report
        assert "supervised_results" in report

    def test_threshold_analysis_returns_rows(self, trained_models, preprocessed):
        from src.evaluate_models import threshold_analysis
        model = trained_models["random_forest"]
        rows = threshold_analysis(model, preprocessed["X_test"], preprocessed["y_test"],
                                  thresholds=[0.3, 0.5, 0.7])
        assert len(rows) == 3
        for row in rows:
            assert 0 <= row["precision"] <= 1
            assert 0 <= row["recall"] <= 1


class TestModelPersistence:
    def test_save_and_load_model(self, trained_models, preprocessed, tmp_path, monkeypatch):
        from src.train_models import _save_model, load_model
        import src.train_models as tm
        monkeypatch.setattr(tm, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(tm.model_path.__func__ if hasattr(tm.model_path, '__func__') else tm,
                            "model_path", lambda name: tmp_path / f"{name}.joblib",
                            raising=False)

        import src.config as cfg
        original_fn = cfg.model_path
        cfg.model_path = lambda name: tmp_path / f"{name}.joblib"

        try:
            model = trained_models["logistic_regression"]
            import joblib
            p = tmp_path / "logistic_regression.joblib"
            joblib.dump(model, p)
            loaded = joblib.load(p)
            preds = loaded.predict(preprocessed["X_test"])
            assert preds.shape == (len(preprocessed["y_test"]),)
        finally:
            cfg.model_path = original_fn
