from datetime import datetime, timezone
from pathlib import Path
import polars as pl
import pytest

from src.config.schemas import FraudAlertRecord, UserRiskProfileRecord
from src.pipeline.gold import GoldProcessor


def test_gold_rule_1_balance_inconsistency():
    df = pl.DataFrame({
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "PAYMENT"],
        "amount": [250000.0, 300000.0, 50000.0, 300000.0],
        "balance_error_orig": [-250000.0, 100.0, -50000.0, 500.0],
        "oldbalanceDest": [1000.0, 2000.0, 0.0, 0.0],
        "newbalanceDest": [1000.0, 2000.0, 0.0, 0.0],
        "nameOrig": ["C1", "C2", "C3", "C4"],
        "step": [1, 1, 1, 1],
    })

    processor = GoldProcessor(large_amount_threshold=200000.0, burst_threshold=5, min_alert_score=35)
    evaluated = processor.evaluate_rules(df.lazy()).collect()

    # Fila 0 (TRANSFER >= 200k con error): Dispara Regla 1 (50 pts)
    # Fila 1 (CASH_OUT >= 200k con error): Dispara Regla 1 (50 pts)
    # Fila 2 (TRANSFER < 200k): No dispara
    # Fila 3 (PAYMENT): No aplica Regla 1
    assert evaluated["rule_severe_balance_error"].to_list() == [True, True, False, False]
    assert evaluated["risk_score"].to_list() == [50, 50, 0, 0]
    assert evaluated["risk_level"].to_list() == ["MEDIUM", "MEDIUM", "NONE", "NONE"]


def test_gold_rule_2_large_transfer_zero_dest():
    df = pl.DataFrame({
        "type": ["TRANSFER", "TRANSFER", "CASH_OUT"],
        "amount": [500000.0, 100000.0, 500000.0],
        "balance_error_orig": [0.0, 0.0, 0.0],
        "oldbalanceDest": [0.0, 0.0, 0.0],
        "newbalanceDest": [0.0, 0.0, 0.0],
        "nameOrig": ["C1", "C2", "C3"],
        "step": [1, 1, 1],
    })

    processor = GoldProcessor(large_amount_threshold=200000.0, burst_threshold=5, min_alert_score=35)
    evaluated = processor.evaluate_rules(df.lazy()).collect()

    # Fila 0 (TRANSFER >= 200k a cuenta en 0): Dispara Regla 2 (35 pts)
    # Fila 1 (TRANSFER < 200k): No dispara
    # Fila 2 (CASH_OUT): Regla 2 solo aplica a TRANSFER
    assert evaluated["rule_large_zero_dest"].to_list() == [True, False, False]
    assert evaluated["risk_score"][0] == 35
    assert evaluated["risk_level"][0] == "MEDIUM"


def test_gold_rule_3_burst_frequency():
    df = pl.DataFrame({
        "type": ["TRANSFER", "PAYMENT", "TRANSFER"],
        "amount": [10.0, 20.0, 30.0],
        "balance_error_orig": [0.0, 0.0, 0.0],
        "oldbalanceDest": [100.0, 100.0, 100.0],
        "newbalanceDest": [100.0, 100.0, 100.0],
        "nameOrig": ["C_BURST", "C_BURST", "C_SINGLE"],  # C_BURST tiene 2 ops en step 1
        "step": [1, 1, 1],
    })

    processor = GoldProcessor(burst_threshold=2)
    evaluated = processor.evaluate_rules(df.lazy()).collect()

    assert evaluated["rule_burst_frequency"].to_list() == [True, True, False]
    assert evaluated["risk_score"].to_list() == [25, 25, 0]


def test_gold_combined_score_and_alert_extraction():
    df = pl.DataFrame({
        "type": ["TRANSFER", "TRANSFER"],
        "amount": [500000.0, 10.0],
        "balance_error_orig": [-500000.0, 0.0],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [0.0, 100.0],
        "nameOrig": ["C_HIGH", "C_LOW"],
        "nameDest": ["M_DEST", "C_DEST"],
        "orig_account_type": ["CLIENT", "CLIENT"],
        "dest_account_type": ["MERCHANT", "CLIENT"],
        "isFraud": [1, 0],
        "kafka_offset": [100, 101],
        "step": [1, 1],
        "ingestion_date": ["2026-10-07", "2026-10-07"],
    })

    # C_HIGH dispara Regla 1 (50) + Regla 2 (35) = 85 -> HIGH
    processor = GoldProcessor(min_alert_score=35)
    evaluated = processor.evaluate_rules(df.lazy()).collect()

    assert evaluated["risk_score"].to_list() == [85, 0]
    assert evaluated["risk_level"].to_list() == ["HIGH", "NONE"]
    assert "SEVERE_BALANCE_ERROR" in evaluated["rules_triggered"][0]
    assert "LARGE_TRANSFER_ZERO_DEST" in evaluated["rules_triggered"][0]

    alerts_df = processor.extract_fraud_alerts(evaluated)
    assert len(alerts_df) == 1
    assert alerts_df["nameOrig"][0] == "C_HIGH"
    assert alerts_df["risk_level"][0] == "HIGH"
    assert "alert_id" in alerts_df.columns


def test_gold_user_risk_profile_aggregation():
    df_evaluated = pl.DataFrame({
        "nameOrig": ["USER_A", "USER_A", "USER_B"],
        "orig_account_type": ["CLIENT", "CLIENT", "CLIENT"],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
        "amount": [300000.0, 100000.0, 50.0],
        "balance_error_orig": [100.0, 0.0, 0.0],
        "risk_score": [50, 0, 0],
        "risk_level": ["MEDIUM", "NONE", "NONE"],
        "step": [1, 2, 1],
        "ingestion_date": ["2026-10-07", "2026-10-07", "2026-10-07"],
    })

    processor = GoldProcessor(min_alert_score=35)
    profiles_df = processor.build_user_risk_profiles(df_evaluated)

    assert len(profiles_df) == 2
    user_a = profiles_df.filter(pl.col("nameOrig") == "USER_A").to_dicts()[0]
    assert user_a["total_transactions"] == 2
    assert user_a["total_volume"] == 400000.0
    assert user_a["transfer_count"] == 1
    assert user_a["cash_out_count"] == 1
    assert user_a["total_alerts"] == 1
    assert user_a["user_risk_level"] == "MEDIUM"

    user_b = profiles_df.filter(pl.col("nameOrig") == "USER_B").to_dicts()[0]
    assert user_b["user_risk_level"] == "LOW"


def test_gold_pipeline_end_to_end(tmp_path):
    silver_dir = tmp_path / "silver" / "transactions" / "ingestion_date=2026-10-07"
    silver_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir = tmp_path / "gold" / "fraud_alerts"
    profiles_dir = tmp_path / "gold" / "user_risk_profile"

    sample_silver = pl.DataFrame({
        "step": [1, 1],
        "type": ["TRANSFER", "PAYMENT"],
        "amount": [500000.0, 100.0],
        "nameOrig": ["C_FRAUD", "C_SAFE"],
        "nameDest": ["M_DEST", "M_SHOP"],
        "oldbalanceOrg": [500000.0, 500.0],
        "newbalanceOrig": [0.0, 400.0],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [0.0, 0.0],
        "orig_account_type": ["CLIENT", "CLIENT"],
        "dest_account_type": ["MERCHANT", "MERCHANT"],
        "balance_error_orig": [-500000.0, 0.0],
        "balance_error_dest": [500000.0, 100.0],
        "isFraud": [1, 0],
        "isFlaggedFraud": [0, 0],
        "ingested_at": [datetime.now(timezone.utc).isoformat()] * 2,
        "ingestion_date": ["2026-10-07"] * 2,
        "kafka_partition": [0, 1],
        "kafka_offset": [201, 202],
        "kafka_timestamp": [1791395400000] * 2,
        "silver_processed_at": [datetime.now(timezone.utc).isoformat()] * 2,
    })

    sample_silver.write_parquet(silver_dir / "clean_silver_test.parquet")

    processor = GoldProcessor(
        silver_path=tmp_path / "silver" / "transactions",
        alerts_path=alerts_dir,
        profiles_path=profiles_dir,
        min_alert_score=35,
    )

    alerts_count, profiles_count = processor.process(partition_date="2026-10-07")

    assert alerts_count == 1
    assert profiles_count == 2

    # Validar archivos en alerts
    alert_files = list(alerts_dir.glob("ingestion_date=2026-10-07/*.parquet"))
    assert len(alert_files) == 1
    df_alerts = pl.read_parquet(alert_files[0])
    assert len(df_alerts) == 1
    assert df_alerts["nameOrig"][0] == "C_FRAUD"

    # Validar schema Pydantic de Alerta
    alert_record = FraudAlertRecord(**df_alerts.to_dicts()[0])
    assert alert_record.risk_score >= 35

    # Validar archivos en profiles
    profile_files = list(profiles_dir.glob("ingestion_date=2026-10-07/*.parquet"))
    assert len(profile_files) == 1
    df_profiles = pl.read_parquet(profile_files[0])
    assert len(df_profiles) == 2

    # Validar schema Pydantic de Perfil
    for row in df_profiles.to_dicts():
        profile_record = UserRiskProfileRecord(**row)
        assert profile_record.user_risk_level in ("LOW", "MEDIUM", "HIGH")


def test_gold_fallback_without_model():
    df = pl.DataFrame({
        "type": ["TRANSFER"],
        "amount": [500000.0],
        "balance_error_orig": [-500000.0],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [0.0],
        "nameOrig": ["C_TEST"],
        "step": [1],
    })

    processor = GoldProcessor(model_path=None)
    evaluated = processor.evaluate_rules(df.lazy()).collect()
    df_final = processor.apply_ml_inference(evaluated)

    # Con fallback, risk_score es exactamente rules_score y ml_probability es 0.0
    assert df_final["risk_score"][0] == 85
    assert df_final["ml_probability"][0] == 0.0
    assert df_final["risk_level"][0] == "HIGH"


def test_gold_with_ml_model_scoring(monkeypatch):
    class MockModel:
        def predict_proba(self, X):
            # Retorna 90% de probabilidad de fraude para cada fila
            return np.array([[0.10, 0.90] for _ in range(len(X))])

    import numpy as np

    df = pl.DataFrame({
        "type": ["TRANSFER"],
        "amount": [500000.0],
        "oldbalanceOrg": [500000.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [0.0],
        "balance_error_orig": [-500000.0],
        "balance_error_dest": [0.0],
        "orig_account_type": ["CLIENT"],
        "dest_account_type": ["MERCHANT"],
        "nameOrig": ["C_TEST"],
        "step": [1],
    })

    processor = GoldProcessor(model_path=None, rules_weight=0.5, ml_weight=0.5)
    processor.model = MockModel()

    evaluated = processor.evaluate_rules(df.lazy()).collect()
    df_final = processor.apply_ml_inference(evaluated)

    # Reglas dan 85 puntos (50 de inconsistencia + 35 de destino cero)
    # ML da 90% -> 90 puntos
    # Score híbrido = round(0.5 * 85 + 0.5 * 90) = round(42.5 + 45.0) = 88
    assert df_final["rules_score"][0] == 85
    assert df_final["ml_probability"][0] == 0.90
    assert df_final["risk_score"][0] == 88
    assert df_final["risk_level"][0] == "HIGH"
