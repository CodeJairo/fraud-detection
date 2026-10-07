from datetime import datetime, timezone
from fastapi.testclient import TestClient
import polars as pl
import pytest

from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_gold_data_if_missing(monkeypatch):
    """
    Asegura que las pruebas de API sean herméticas e independientes del disco local.
    Si la Capa Gold no existe en disco (ej. entorno limpio de CI en GitHub Actions),
    proporciona datasets tipados en memoria para validar la lógica de los endpoints.
    """
    from src.api import main as api_main
    from src.config.settings import GOLD_ALERTS_PATH, GOLD_PROFILES_PATH

    orig_read = api_main.read_latest_parquet

    def mock_read(directory):
        real_df = orig_read(directory)
        if not real_df.is_empty():
            return real_df

        if directory == GOLD_PROFILES_PATH:
            return pl.DataFrame({
                "nameOrig": ["C1057507014", "C684230144"],
                "account_type": ["CLIENT", "CLIENT"],
                "total_transactions": [5, 2],
                "total_volume": [500000.0, 100000.0],
                "avg_transaction_amount": [100000.0, 50000.0],
                "max_transaction_amount": [300000.0, 80000.0],
                "transfer_count": [4, 1],
                "cash_out_count": [1, 1],
                "total_alerts": [3, 1],
                "high_risk_alerts": [2, 0],
                "avg_balance_error": [25000.0, 0.0],
                "user_risk_level": ["HIGH", "MEDIUM"],
                "last_activity_step": [1, 1],
                "profile_updated_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            })

        if directory == GOLD_ALERTS_PATH:
            return pl.DataFrame({
                "alert_id": ["alert-001", "alert-002"],
                "step": [1, 1],
                "type": ["TRANSFER", "CASH_OUT"],
                "amount": [250000.0, 300000.0],
                "nameOrig": ["C1057507014", "C684230144"],
                "oldbalanceOrg": [250000.0, 300000.0],
                "newbalanceOrig": [0.0, 0.0],
                "nameDest": ["M9999", "M8888"],
                "orig_account_type": ["CLIENT", "CLIENT"],
                "dest_account_type": ["MERCHANT", "MERCHANT"],
                "rules_triggered": ["BALANCE_INCONSISTENCY", "BURST_TRANSACTIONS"],
                "risk_score": [85, 45],
                "risk_level": ["HIGH", "MEDIUM"],
                "ml_probability": [0.95, 0.40],
                "isFraud": [1, 0],
                "kafka_offset": [100, 101],
                "alert_timestamp": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            })

        return real_df

    monkeypatch.setattr(api_main, "read_latest_parquet", mock_read)


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


def test_api_pipeline_simulate(monkeypatch):
    from unittest.mock import MagicMock
    mock_seed = MagicMock(return_value={
        "status": "success",
        "transactions_generated": 100,
        "silver_records": 100,
        "fraud_alerts_created": 15,
        "user_profiles_created": 40,
        "execution_time_seconds": 0.25,
    })
    monkeypatch.setattr("src.pipeline.seed.run_pipeline_seed", mock_seed)

    response = client.post("/pipeline/simulate?count=100")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["transactions_generated"] == 100
    assert data["fraud_alerts_created"] == 15


