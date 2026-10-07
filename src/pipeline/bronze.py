import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, List, Optional
import uuid

from confluent_kafka import Consumer, KafkaError, KafkaException
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.settings import (
    BRONZE_BATCH_SIZE,
    BRONZE_BATCH_TIMEOUT_SEC,
    BRONZE_DATA_PATH,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP_BRONZE,
    KAFKA_TOPIC_RAW,
)


class BronzeConsumer:
    """
    Consumidor de la Capa Bronze para ingesta cruda desde Redpanda/Kafka.
    Escribe eventos en micro-lotes hacia archivos Parquet optimizados con metadatos de auditoría.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topic: str = KAFKA_TOPIC_RAW,
        group_id: str = KAFKA_CONSUMER_GROUP_BRONZE,
        output_dir: Path = BRONZE_DATA_PATH,
        batch_size: int = BRONZE_BATCH_SIZE,
        batch_timeout_sec: float = BRONZE_BATCH_TIMEOUT_SEC,
        auto_offset_reset: str = "earliest",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.batch_timeout_sec = batch_timeout_sec

        self.batch: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()
        self.total_processed = 0
        self.running = True

        # Configuración del consumidor
        consumer_conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": False,  # Commit manual para garantizar at-least-once
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 300000,
        }
        self.consumer = Consumer(consumer_conf)

    def _setup_signals(self) -> None:
        """Configura captura de señales para apagado ordenado."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Señal {signum} recibida. Deteniendo consumidor Bronze de forma ordenada...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def flush_batch(self) -> Optional[Path]:
        """
        Vuelca el lote actual en memoria a un archivo Parquet en disco
        y realiza commit manual del offset en Kafka.
        """
        if not self.batch:
            self.last_flush_time = time.time()
            return None

        count = len(self.batch)
        now_utc = datetime.now(timezone.utc)
        date_partition = now_utc.strftime("%Y-%m-%d")
        partition_dir = self.output_dir / f"ingestion_date={date_partition}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        batch_id = uuid.uuid4().hex[:8]
        timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S_%f")
        file_path = partition_dir / f"bronze_batch_{timestamp_str}_{batch_id}.parquet"

        try:
            df = pd.DataFrame(self.batch)
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, file_path, compression="snappy")

            # Garantía at-least-once: el commit del offset ocurre ÚNICAMENTE tras persistir y cerrar el archivo Parquet
            self.consumer.commit(asynchronous=False)
            self.total_processed += count
            print(f"💾 [BRONZE] Batch persistido: {count} registros en {file_path.name} | Total acumulado: {self.total_processed}")

            self.batch.clear()
            self.last_flush_time = time.time()
            return file_path

        except Exception as e:
            print(f"❌ Error al escribir archivo Parquet o hacer commit: {e}", file=sys.stderr)
            raise

    def enrich_record(self, raw_data: Dict[str, Any], msg) -> Dict[str, Any]:
        """Agrega metadatos de auditoría al evento crudo."""
        now_utc = datetime.now(timezone.utc)
        enriched = dict(raw_data)
        enriched["ingested_at"] = now_utc.isoformat()
        enriched["ingestion_date"] = now_utc.strftime("%Y-%m-%d")
        enriched["kafka_partition"] = msg.partition()
        enriched["kafka_offset"] = msg.offset()

        ts_type, ts_value = msg.timestamp()
        enriched["kafka_timestamp"] = ts_value if ts_type != 0 else None
        return enriched

    def start(
        self,
        max_messages: Optional[int] = None,
        max_batches: Optional[int] = None,
        poll_timeout: float = 1.0,
    ) -> int:
        """Inicia el ciclo principal de consumo."""
        self._setup_signals()
        self.consumer.subscribe([self.topic])
        print(f"📥 Consumidor Bronze conectado a '{self.topic}' (grupo: '{self.group_id}').")
        print(f"📂 Destino de almacenamiento: {self.output_dir}")
        print(f"⚙️ Configuración de batch: {self.batch_size} registros o {self.batch_timeout_sec}s timeout.")

        batches_written = 0

        try:
            while self.running:
                msg = self.consumer.poll(poll_timeout)

                if msg is None:
                    # Timeout alcanzado, verificar si corresponde hacer flush por tiempo
                    if self.batch and (time.time() - self.last_flush_time >= self.batch_timeout_sec):
                        if self.flush_batch():
                            batches_written += 1
                            if max_batches is not None and batches_written >= max_batches:
                                break
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                # Decodificar y parsear mensaje
                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                    record = self.enrich_record(payload, msg)
                    self.batch.append(record)
                except Exception as e:
                    print(f"⚠️ Error deserializando mensaje offset={msg.offset()}: {e}", file=sys.stderr)
                    continue

                # Verificar si alcanzamos el tamaño del batch
                if len(self.batch) >= self.batch_size:
                    if self.flush_batch():
                        batches_written += 1
                        if max_batches is not None and batches_written >= max_batches:
                            break

                if max_messages is not None and (self.total_processed + len(self.batch)) >= max_messages:
                    if self.batch:
                        self.flush_batch()
                        batches_written += 1
                    break

        except KeyboardInterrupt:
            print("\nInterrupción detectada en loop principal.")
        finally:
            print("🛑 Finalizando consumidor Bronze...")
            if self.batch:
                print(f"📦 Vaciando último batch en memoria ({len(self.batch)} registros)...")
                self.flush_batch()
            self.consumer.close()
            print(f"🏁 Consumidor Bronze cerrado. Total procesado: {self.total_processed} registros.")

        return self.total_processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Consumidor Bronze para ingesta de transacciones crudas")
    parser.add_argument("--batch-size", type=int, default=BRONZE_BATCH_SIZE, help="Cantidad de eventos por lote")
    parser.add_argument("--batch-timeout", type=float, default=BRONZE_BATCH_TIMEOUT_SEC, help="Timeout en segundos para volcar lote")
    parser.add_argument("--max-messages", type=int, default=None, help="Límite de mensajes a procesar (para testing)")
    parser.add_argument("--max-batches", type=int, default=None, help="Límite de batches a persistir (para testing)")
    args = parser.parse_args()

    consumer = BronzeConsumer(
        batch_size=args.batch_size,
        batch_timeout_sec=args.batch_timeout,
    )
    consumer.start(
        max_messages=args.max_messages,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
