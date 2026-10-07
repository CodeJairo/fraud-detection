from datetime import datetime, timezone
from unittest.mock import MagicMock
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.config.schemas import BronzeTransactionRecord, TransactionEvent
from src.pipeline.bronze import BronzeConsumer


class MockKafkaMessage:
    def __init__(self, partition=1, offset=42, timestamp=(1, 1696694400000)):
        self._partition = partition
        self._offset = offset
        self._timestamp = timestamp

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def timestamp(self):
        return self._timestamp


def test_transaction_event_schema_validation():
    valid_data = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.50,
        "nameOrig": "C12345",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 399.50,
        "nameDest": "M67890",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    event = TransactionEvent(**valid_data)
    assert event.amount == 100.50
    assert event.type == "PAYMENT"


def test_bronze_record_enrichment():
    mock_msg = MockKafkaMessage(partition=2, offset=105, timestamp=(1, 1700000000000))

    consumer = BronzeConsumer.__new__(BronzeConsumer)
    consumer.batch = []

    raw_payload = {
        "step": 2,
        "type": "TRANSFER",
        "amount": 50000.0,
        "nameOrig": "C999",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C888",
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 51000.0,
        "isFraud": 1,
        "isFlaggedFraud": 0,
    }

    enriched = consumer.enrich_record(raw_payload, mock_msg)

    assert "ingested_at" in enriched
    assert "ingestion_date" in enriched
    assert enriched["kafka_partition"] == 2
    assert enriched["kafka_offset"] == 105
    assert enriched["kafka_timestamp"] == 1700000000000
    assert enriched["isFraud"] == 1

    # Validar contra esquema Pydantic Bronze
    bronze_obj = BronzeTransactionRecord(**enriched)
    assert bronze_obj.nameOrig == "C999"


def test_bronze_flush_batch_to_parquet(tmp_path):
    consumer = BronzeConsumer.__new__(BronzeConsumer)
    consumer.output_dir = tmp_path / "bronze"
    consumer.batch_size = 10
    consumer.batch_timeout_sec = 5.0
    consumer.total_processed = 0
    consumer.last_flush_time = 0
    consumer.consumer = MagicMock()

    # Preparar lote de prueba
    sample_records = [
        {
            "step": i,
            "type": "PAYMENT",
            "amount": float(i * 10),
            "nameOrig": f"C{i}",
            "oldbalanceOrg": 100.0,
            "newbalanceOrig": 100.0 - (i * 10),
            "nameDest": f"M{i}",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "ingestion_date": "2026-10-07",
            "kafka_partition": 0,
            "kafka_offset": i,
            "kafka_timestamp": 1696694400000,
        }
        for i in range(5)
    ]
    consumer.batch = list(sample_records)

    # Ejecutar flush
    saved_file = consumer.flush_batch()

    assert saved_file is not None
    assert saved_file.exists()
    assert saved_file.suffix == ".parquet"
    assert consumer.total_processed == 5
    assert len(consumer.batch) == 0

    # Verificar que se realizó commit síncrono a Kafka
    consumer.consumer.commit.assert_called_once_with(asynchronous=False)

    # Leer el archivo Parquet y validar contenido
    df_read = pd.read_parquet(saved_file)
    assert len(df_read) == 5
    assert "ingested_at" in df_read.columns
    assert "kafka_offset" in df_read.columns
    assert df_read["amount"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
