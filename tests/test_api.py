from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "medallion_status" in data
    assert "bronze_files" in data["medallion_status"]
    assert "silver_files" in data["medallion_status"]
    assert "total_fraud_alerts" in data["medallion_status"]
    assert "total_user_profiles" in data["medallion_status"]
    assert "ml_model_active" in data["medallion_status"]


def test_api_alerts_unfiltered():
    response = client.get("/alerts?limit=10")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    if alerts:
        first = alerts[0]
        assert "alert_id" in first
        assert "risk_score" in first
        assert "risk_level" in first
        assert "rules_triggered" in first
        assert "ml_probability" in first


def test_api_alerts_filtered_by_risk_level():
    response = client.get("/alerts?risk_level=HIGH&limit=5")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    for alert in alerts:
        assert alert["risk_level"] == "HIGH"


def test_api_alerts_filtered_by_rule():
    response = client.get("/alerts?rule=balance&limit=5")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    for alert in alerts:
        assert "BALANCE" in alert["rules_triggered"].upper()


def test_api_alerts_filtered_by_ml_probability():
    response = client.get("/alerts?min_ml_prob=0.0&limit=5")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    for alert in alerts:
        assert alert["ml_probability"] >= 0.0


def test_api_user_risk_profile_existing():
    # C1057507014 es una de las cuentas procesadas en la Capa Gold
    response = client.get("/users/C1057507014/risk-profile")
    assert response.status_code == 200
    profile = response.json()
    assert profile["nameOrig"] == "C1057507014"
    assert "total_transactions" in profile
    assert "total_volume" in profile
    assert profile["user_risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_api_user_risk_profile_not_found():
    response = client.get("/users/UNKNOWN_USER_99999/risk-profile")
    assert response.status_code == 404
    error = response.json()
    assert "detail" in error
    assert "no encontrado" in error["detail"]


def test_api_evaluate_high_risk_transaction():
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 350000.0,
        "nameOrig": "C_FRAUDSTER",
        "oldbalanceOrg": 350000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "M_MULE",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] >= 60
    assert data["risk_level"] == "HIGH"
    assert "BLOQUEO" in data["recommendation"]
    assert "LARGE_TRANSFER_ZERO_DEST" in data["rules_triggered"]
    assert data["decision_color"] == "#EF553B"


def test_api_evaluate_low_risk_transaction():
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 50.0,
        "nameOrig": "C_CLIENT",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 450.0,
        "nameDest": "M_STORE",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] == 0
    assert data["risk_level"] == "NONE"
    assert "APROBADA" in data["recommendation"]
    assert data["decision_color"] == "#00CC96"

