from pathlib import Path
import polars as pl
import pytest

from src.pipeline.seed import generate_synthetic_transactions, run_pipeline_seed


def test_generate_synthetic_transactions_structure():
    """Valida la generación de transacciones sintéticas y presencia de cuentas ancla."""
    df = generate_synthetic_transactions(n_transactions=50, seed=42)
    assert len(df) == 50
    assert "nameOrig" in df.columns
    assert "amount" in df.columns
    assert "balance_error_orig" not in df.columns  # Se calcula en Silver, no en Bronze

    users = df["nameOrig"].to_list()
    # Verifica que las 3 cuentas ancla de demostración estén presentes
    assert "C1057507014" in users
    assert "C684230144" in users
    assert "C1231006815" in users


def test_run_pipeline_seed_execution(tmp_path, monkeypatch):
    """Valida la ejecución del pipeline Medallion completo en un directorio temporal."""
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    alerts_dir = tmp_path / "gold" / "alerts"
    profiles_dir = tmp_path / "gold" / "profiles"

    monkeypatch.setattr("src.pipeline.seed.BRONZE_DATA_PATH", bronze_dir)
    monkeypatch.setattr("src.pipeline.seed.SILVER_DATA_PATH", silver_dir)
    monkeypatch.setattr("src.pipeline.seed.GOLD_ALERTS_PATH", alerts_dir)
    monkeypatch.setattr("src.pipeline.seed.GOLD_PROFILES_PATH", profiles_dir)

    result = run_pipeline_seed(n_transactions=60, seed=123)

    assert result["status"] == "success"
    assert result["transactions_generated"] == 60
    assert result["silver_records"] > 0
    assert result["user_profiles_created"] > 0
    assert result["execution_time_seconds"] >= 0.0

    # Verifica que se hayan creado los archivos Parquet en disco
    bronze_files = list(bronze_dir.glob("**/*.parquet"))
    silver_files = list(silver_dir.glob("**/*.parquet"))
    profiles_files = list(profiles_dir.glob("**/*.parquet"))

    assert len(bronze_files) >= 1
    assert len(silver_files) >= 1
    assert len(profiles_files) >= 1
