from datetime import datetime, timezone
from pathlib import Path
import polars as pl
import pytest

from src.config.schemas import SilverTransactionRecord
from src.pipeline.silver import SilverProcessor


def test_account_type_extraction():
    df = pl.DataFrame({
        "nameOrig": ["C12345", "M99999", "X00000"],
        "nameDest": ["M55555", "C66666", "Y77777"],
        "amount": [100.0, 200.0, 300.0],
        "step": [1, 1, 1],
        "oldbalanceOrg": [500.0, 500.0, 500.0],
        "newbalanceOrig": [400.0, 300.0, 200.0],
        "oldbalanceDest": [0.0, 0.0, 0.0],
        "newbalanceDest": [100.0, 200.0, 300.0],
        "isFraud": [0, 0, 0],
        "isFlaggedFraud": [0, 0, 0],
    })

    processor = SilverProcessor(bronze_path=Path("dummy"), silver_path=Path("dummy"))
    cleaned_lf = processor.clean_and_normalize(df.lazy())
    result = cleaned_lf.collect()

    assert result["orig_account_type"].to_list() == ["CLIENT", "MERCHANT", "UNKNOWN"]
    assert result["dest_account_type"].to_list() == ["MERCHANT", "CLIENT", "UNKNOWN"]


def test_balance_error_calculations():
    # Caso 1: Consistente (old 100 - amount 40 = new 60) -> error = 0.0
    # Caso 2: Inconsistencia emisor (old 100 - amount 40 = 60, pero new es 0) -> (100 - 40) - 0 = 60.0
    # Caso 3: Inconsistencia receptor (old 0 + amount 40 = 40, pero new es 0) -> (0 + 40) - 0 = 40.0
    df = pl.DataFrame({
        "oldbalanceOrg": [100.0, 100.0, 50.0],
        "amount": [40.0, 40.0, 20.0],
        "newbalanceOrig": [60.0, 0.0, 30.0],
        "oldbalanceDest": [0.0, 10.0, 0.0],
        "newbalanceDest": [40.0, 50.0, 0.0],
    })

    processor = SilverProcessor(bronze_path=Path("dummy"), silver_path=Path("dummy"))
    featured_lf = processor.add_balance_features(df.lazy())
    result = featured_lf.collect()

    assert result["balance_error_orig"].to_list() == [0.0, 60.0, 0.0]
    assert result["balance_error_dest"].to_list() == [0.0, 0.0, 20.0]


def test_deduplication():
    df = pl.DataFrame({
        "nameOrig": ["C1", "C2", "C1"],  # Filas 0 y 2 son duplicados de negocio
        "nameDest": ["M1", "C3", "M1"],
        "amount": [10.0, 20.0, 10.0],
        "step": [1, 2, 1],
        "kafka_offset": [101, 102, 101],
        "payload_extra": ["first", "other", "duplicate"],
    })

    processor = SilverProcessor(bronze_path=Path("dummy"), silver_path=Path("dummy"))
    dedup_lf = processor.deduplicate(df.lazy())
    result = dedup_lf.collect()

    assert len(result) == 2
    assert set(result["payload_extra"].to_list()) == {"first", "other"}
    assert "duplicate" not in result["payload_extra"].to_list()


def test_clean_and_filter_invalid_rows():
    df = pl.DataFrame({
        "nameOrig": ["C1", None, "C2"],
        "nameDest": ["M1", "M2", None],
        "amount": [100.0, 50.0, -10.0],  # Fila 1 y 2 con nulos/monto negativo
        "step": [1, 1, 1],
        "oldbalanceOrg": [100.0, 50.0, 50.0],
        "newbalanceOrig": [0.0, 0.0, 60.0],
        "oldbalanceDest": [0.0, 0.0, 0.0],
        "newbalanceDest": [100.0, 50.0, 0.0],
        "isFraud": [0, 0, 0],
        "isFlaggedFraud": [0, 0, 0],
    })

    processor = SilverProcessor(bronze_path=Path("dummy"), silver_path=Path("dummy"))
    cleaned_lf = processor.clean_and_normalize(df.lazy())
    result = cleaned_lf.collect()

    # Solo la fila 0 es válida
    assert len(result) == 1
    assert result["nameOrig"][0] == "C1"


def test_silver_pipeline_end_to_end(tmp_path):
    bronze_dir = tmp_path / "bronze" / "transactions" / "ingestion_date=2026-10-07"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir = tmp_path / "silver" / "transactions"

    # Crear dataset de prueba simulando Bronze
    sample_data = pl.DataFrame({
        "step": [1, 1, 1],
        "type": ["PAYMENT", "TRANSFER", "PAYMENT"],  # Tercera fila duplicada de la primera
        "amount": [150.0, 5000.0, 150.0],
        "nameOrig": ["C100", "C200", "C100"],
        "oldbalanceOrg": [300.0, 5000.0, 300.0],
        "newbalanceOrig": [150.0, 0.0, 150.0],
        "nameDest": ["M300", "C400", "M300"],
        "oldbalanceDest": [0.0, 1000.0, 0.0],
        "newbalanceDest": [0.0, 6000.0, 0.0],
        "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 0],
        "ingested_at": [datetime.now(timezone.utc).isoformat()] * 3,
        "ingestion_date": ["2026-10-07"] * 3,
        "kafka_partition": [0, 1, 0],
        "kafka_offset": [10, 11, 10],
        "kafka_timestamp": [1791395400000] * 3,
    })

    test_bronze_file = bronze_dir / "bronze_batch_test.parquet"
    sample_data.write_parquet(test_bronze_file)

    processor = SilverProcessor(
        bronze_path=tmp_path / "bronze" / "transactions",
        silver_path=silver_dir,
    )

    processed_count = processor.process(partition_date="2026-10-07")

    # Debe haber deduplicado la tercera fila (3 -> 2)
    assert processed_count == 2

    # Verificar existencia de archivos en Silver
    silver_files = list(silver_dir.glob("ingestion_date=2026-10-07/*.parquet"))
    assert len(silver_files) == 1

    df_silver = pl.read_parquet(silver_files[0])
    assert len(df_silver) == 2
    assert "orig_account_type" in df_silver.columns
    assert "dest_account_type" in df_silver.columns
    assert "balance_error_orig" in df_silver.columns
    assert "balance_error_dest" in df_silver.columns
    assert "silver_processed_at" in df_silver.columns

    # Validar que los registros cumplen el esquema Pydantic SilverTransactionRecord
    for row in df_silver.to_dicts():
        silver_record = SilverTransactionRecord(**row)
        assert silver_record.orig_account_type in ("CLIENT", "MERCHANT")
