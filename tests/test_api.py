"""Tests for Flask API endpoints."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client(trained_models, preprocessed):
    """
    Provide a Flask test client with all models and preprocessor saved
    to the default artifacts directory.
    """
    import numpy as np
    from src.config import MODELS_DIR, MODEL_COMPARISON_PATH, BEST_MODEL_META_PATH
    from src.preprocessing import save_preprocessor
    from src.evaluate_models import (
        evaluate_supervised, evaluate_isolation_forest, build_comparison_report
    )
    from src.train_models import (
        save_comparison_report, save_best_model_meta, save_dataset_meta,
        MODEL_DISPLAY_NAMES,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessor
    save_preprocessor(preprocessed["scaler"], preprocessed["feature_names"])

    # Save test arrays
    np.save(MODELS_DIR / "X_test.npy", preprocessed["X_test"])
    np.save(MODELS_DIR / "y_test.npy", preprocessed["y_test"].values
            if hasattr(preprocessed["y_test"], "values") else preprocessed["y_test"])

    # Evaluate and save reports
    results = []
    for name in ["logistic_regression", "decision_tree", "random_forest", "xgboost"]:
        r = evaluate_supervised(name, trained_models[name],
                                preprocessed["X_test"], preprocessed["y_test"])
        results.append(r)
    if_result = evaluate_isolation_forest(trained_models["isolation_forest"],
                                          preprocessed["X_test"], preprocessed["y_test"])
    results.append(if_result)

    dataset_meta = {
        "total_transactions": int(len(preprocessed["y_test"]) + len(preprocessed["y_train"])),
        "fraud_count": int(preprocessed["y_test"].sum()),
        "legitimate_count": int((preprocessed["y_test"] == 0).sum()),
        "fraud_percentage": 2.0,
    }
    report = build_comparison_report(results, dataset_meta)
    save_comparison_report(report)
    save_dataset_meta({**dataset_meta, "timestamp": "2026-01-01T00:00:00+00:00"})
    save_best_model_meta({
        "model_name":    report["best_model"],
        "display_name":  report["best_display"],
        "selection_metric": "f1",
        "metrics":       report["best_metrics"],
        "timestamp":     report["timestamp"],
        "is_sampled":    True,
        "sample_size":   1020,
    })

    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, client):
        res = client.get("/api/health")
        data = json.loads(res.data)
        assert "timestamp" in data

    def test_health_has_dataset_info(self, client):
        res = client.get("/api/health")
        data = json.loads(res.data)
        assert "dataset" in data


class TestDatasetSummaryEndpoint:
    def test_summary_200_if_artifact(self, client):
        res = client.get("/api/dataset/summary")
        # Will be 404 if real dataset missing, 200 if artifact exists
        assert res.status_code in (200, 404)
        data = json.loads(res.data)
        assert "available" in data


class TestModelsEndpoints:
    def test_comparison_available(self, client):
        res = client.get("/api/models/comparison")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["available"] is True
        assert "best_model" in data
        assert "supervised_results" in data

    def test_models_status(self, client):
        res = client.get("/api/models/status")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "models" in data


class TestPredictEndpoint:
    def _sample_body(self):
        from src.config import FEATURE_COLUMNS
        return {f: 0.0 for f in FEATURE_COLUMNS}

    def test_predict_valid(self, client):
        body = self._sample_body()
        res  = client.post("/api/predict",
                           data=json.dumps(body),
                           content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "prediction" in data
        assert data["prediction"] in (0, 1)

    def test_predict_missing_feature(self, client):
        body = {"Time": 1.0}   # missing most features
        res  = client.post("/api/predict",
                           data=json.dumps(body),
                           content_type="application/json")
        assert res.status_code == 400

    def test_predict_non_numeric(self, client):
        from src.config import FEATURE_COLUMNS
        body = {f: "abc" for f in FEATURE_COLUMNS}
        res  = client.post("/api/predict",
                           data=json.dumps(body),
                           content_type="application/json")
        assert res.status_code == 400

    def test_predict_returns_disclaimer(self, client):
        body = self._sample_body()
        res  = client.post("/api/predict",
                           data=json.dumps(body),
                           content_type="application/json")
        data = json.loads(res.data)
        assert "disclaimer" in data

    def test_predict_has_risk_category(self, client):
        body = self._sample_body()
        res  = client.post("/api/predict",
                           data=json.dumps(body),
                           content_type="application/json")
        data = json.loads(res.data)
        assert "risk_category" in data


class TestModelsComparison:
    def test_comparison_has_all_models(self, client):
        res  = client.get("/api/models/comparison")
        data = json.loads(res.data)
        names = {r["model_name"] for r in data.get("supervised_results", [])}
        expected = {"logistic_regression", "decision_tree", "random_forest", "xgboost"}
        assert expected.issubset(names)

    def test_if_in_unsupervised(self, client):
        res  = client.get("/api/models/comparison")
        data = json.loads(res.data)
        names = {r["model_name"] for r in data.get("unsupervised_results", [])}
        assert "isolation_forest" in names


class TestMissingArtifact:
    def test_no_json_body_predict_400(self, client):
        res = client.post("/api/predict",
                          data="not json",
                          content_type="text/plain")
        assert res.status_code == 400

    def test_business_insights_requires_report(self, client):
        res = client.get("/api/business-insights")
        assert res.status_code in (200, 404)


class TestCasesAPI:
    def test_case_lifecycle(self, client):
        # 1. Create case
        payload = {
            "title": "Suspicious $4,200 transfer",
            "amount": 4200.0,
            "time": 10523.0,
            "risk_level": "High",
            "notes": "Spike in transaction volume flagged by XGBoost"
        }
        create_res = client.post("/api/cases",
                                 data=json.dumps(payload),
                                 content_type="application/json")
        assert create_res.status_code == 201
        created = json.loads(create_res.data)
        assert created["risk_level"] == "High"
        case_id = created["id"]

        # 2. List cases
        list_res = client.get("/api/cases")
        assert list_res.status_code == 200
        cases_data = json.loads(list_res.data)
        assert cases_data["total"] >= 1

        # 3. Update case
        patch_res = client.patch(f"/api/cases/{case_id}",
                                 data=json.dumps({"status": "Under Review", "notes": "Investigating customer IP"}),
                                 content_type="application/json")
        assert patch_res.status_code == 200
        updated = json.loads(patch_res.data)
        assert updated["status"] == "Under Review"

        # 4. Delete case
        del_res = client.delete(f"/api/cases/{case_id}")
        assert del_res.status_code == 200

