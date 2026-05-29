"""
SkyPrice API — pytest test suite.

Usage:
    # Make sure the API is running first:
    #   uvicorn api:app --port 8000
    pytest api_test.py -v
"""
import pytest
import requests
import time

BASE_URL = "http://127.0.0.1:8000"

# ── Shared valid payload ──────────────────────────────────────────────────
VALID_PAYLOAD = {
    "airline": "Vistara",
    "source_city": "Delhi",
    "destination_city": "Mumbai",
    "departure_time": "Morning",
    "arrival_time": "Afternoon",
    "duration": 2.5,
    "days_left": 15,
    "stops": "zero",
    "flight_class": "Economy",
}


# ═══════════════════════════════════════════════════════════════════════════
# Health & Monitoring
# ═══════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_200(self):
        res = requests.get(f"{BASE_URL}/health")
        assert res.status_code == 200

    def test_health_schema(self):
        res = requests.get(f"{BASE_URL}/health")
        body = res.json()
        assert "status" in body
        assert "model_loaded" in body
        assert body["status"] in ("healthy", "degraded")

    def test_health_model_loaded(self):
        """API should report model as loaded if train_model.py has been run."""
        res = requests.get(f"{BASE_URL}/health")
        body = res.json()
        assert body["model_loaded"] is True, (
            "Model not loaded — run train_model.py first."
        )

    def test_model_info_returns_200(self):
        res = requests.get(f"{BASE_URL}/model-info")
        assert res.status_code == 200

    def test_model_info_has_metrics(self):
        res = requests.get(f"{BASE_URL}/model-info")
        body = res.json()
        assert "metrics" in body
        assert "r2_score" in body["metrics"]
        assert "mae" in body["metrics"]
        assert body["metrics"]["r2_score"] > 0.5, "R² seems too low — check training."

    def test_feature_importance_returns_200(self):
        res = requests.get(f"{BASE_URL}/feature-importance")
        assert res.status_code == 200

    def test_feature_importance_schema(self):
        res = requests.get(f"{BASE_URL}/feature-importance")
        body = res.json()
        assert "feature_importance" in body
        assert "features" in body
        assert len(body["features"]) == 9


# ═══════════════════════════════════════════════════════════════════════════
# Single Prediction — Happy Path
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictSingle:
    def test_predict_returns_200(self):
        res = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        assert res.status_code == 200

    def test_predict_schema(self):
        res = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        body = res.json()
        assert "predicted_price" in body
        assert "latency_seconds" in body

    def test_predict_price_is_positive(self):
        res = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        assert res.json()["predicted_price"] > 0

    def test_predict_price_is_plausible(self):
        """Indian domestic fares are typically between Rs. 1,000 and Rs. 1,00,000."""
        res = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        price = res.json()["predicted_price"]
        assert 1_000 <= price <= 1_00_000, f"Price {price} seems implausible."

    def test_predict_latency_under_100ms(self):
        """Inference should complete in under 100 ms."""
        res = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        assert res.json()["latency_seconds"] < 0.1, "Inference latency exceeded 100ms."

    def test_predict_business_higher_than_economy(self):
        eco = requests.post(f"{BASE_URL}/predict", json={**VALID_PAYLOAD, "flight_class": "Economy"})
        biz = requests.post(f"{BASE_URL}/predict", json={**VALID_PAYLOAD, "flight_class": "Business"})
        assert biz.json()["predicted_price"] > eco.json()["predicted_price"], (
            "Business class fare should be higher than Economy."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Prediction with Confidence Range
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictRange:
    def test_predict_range_returns_200(self):
        res = requests.post(f"{BASE_URL}/predict-range", json=VALID_PAYLOAD)
        assert res.status_code == 200

    def test_predict_range_schema(self):
        body = requests.post(f"{BASE_URL}/predict-range", json=VALID_PAYLOAD).json()
        for key in ("predicted_price", "p10", "p90", "std_dev", "latency_seconds"):
            assert key in body, f"Missing key: {key}"

    def test_predict_range_ordering(self):
        """p10 must be ≤ predicted_price ≤ p90."""
        body = requests.post(f"{BASE_URL}/predict-range", json=VALID_PAYLOAD).json()
        assert body["p10"] <= body["predicted_price"] <= body["p90"], (
            f"Interval ordering violated: {body}"
        )

    def test_predict_range_std_dev_positive(self):
        body = requests.post(f"{BASE_URL}/predict-range", json=VALID_PAYLOAD).json()
        assert body["std_dev"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Batch Simulation
# ═══════════════════════════════════════════════════════════════════════════

class TestSimulate:
    def _make_batch(self, n: int):
        return {"flights": [VALID_PAYLOAD] * n}

    def test_simulate_single_item(self):
        res = requests.post(f"{BASE_URL}/simulate", json=self._make_batch(1))
        assert res.status_code == 200
        assert len(res.json()["predicted_prices"]) == 1

    def test_simulate_batch_10(self):
        res = requests.post(f"{BASE_URL}/simulate", json=self._make_batch(10))
        assert res.status_code == 200
        body = res.json()
        assert body["total_scenarios"] == 10
        assert len(body["predicted_prices"]) == 10

    def test_simulate_all_prices_positive(self):
        res = requests.post(f"{BASE_URL}/simulate", json=self._make_batch(5))
        for price in res.json()["predicted_prices"]:
            assert price > 0

    def test_simulate_has_latency(self):
        res = requests.post(f"{BASE_URL}/simulate", json=self._make_batch(5))
        assert "latency_seconds" in res.json()


# ═══════════════════════════════════════════════════════════════════════════
# Input Validation — Invalid Inputs Must Return 422
# ═══════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    def _post(self, overrides: dict):
        return requests.post(f"{BASE_URL}/predict", json={**VALID_PAYLOAD, **overrides})

    def test_invalid_airline_returns_422(self):
        res = self._post({"airline": "FakeAir"})
        assert res.status_code == 422, f"Expected 422, got {res.status_code}"

    def test_invalid_source_city_returns_422(self):
        res = self._post({"source_city": "London"})
        assert res.status_code == 422

    def test_invalid_destination_city_returns_422(self):
        res = self._post({"destination_city": "NewYork"})
        assert res.status_code == 422

    def test_invalid_departure_time_returns_422(self):
        res = self._post({"departure_time": "Midnight"})
        assert res.status_code == 422

    def test_invalid_stops_returns_422(self):
        res = self._post({"stops": "three"})
        assert res.status_code == 422

    def test_invalid_class_returns_422(self):
        res = self._post({"flight_class": "First"})
        assert res.status_code == 422

    def test_days_left_zero_returns_422(self):
        res = self._post({"days_left": 0})
        assert res.status_code == 422

    def test_days_left_negative_returns_422(self):
        res = self._post({"days_left": -5})
        assert res.status_code == 422

    def test_days_left_too_large_returns_422(self):
        res = self._post({"days_left": 999})
        assert res.status_code == 422

    def test_duration_zero_returns_422(self):
        res = self._post({"duration": 0})
        assert res.status_code == 422

    def test_duration_negative_returns_422(self):
        res = self._post({"duration": -2.5})
        assert res.status_code == 422

    def test_duration_too_large_returns_422(self):
        res = self._post({"duration": 25.0})
        assert res.status_code == 422

    def test_missing_required_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "airline"}
        res = requests.post(f"{BASE_URL}/predict", json=payload)
        assert res.status_code == 422

    def test_empty_body_returns_422(self):
        res = requests.post(f"{BASE_URL}/predict", json={})
        assert res.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Latency Benchmark
# ═══════════════════════════════════════════════════════════════════════════

class TestLatency:
    def test_10_sequential_predictions_average_under_50ms(self):
        latencies = []
        for _ in range(10):
            t0 = time.time()
            requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
            latencies.append(time.time() - t0)
        avg = sum(latencies) / len(latencies)
        assert avg < 0.05, f"Average end-to-end latency {avg*1000:.1f}ms exceeds 50ms target."
