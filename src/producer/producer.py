import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from confluent_kafka import KafkaError, Producer

from src.config.schemas import TransactionEvent
from src.config.settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_RAW,
    PAYSIM_CSV_PATH,
    PRODUCER_CHUNK_SIZE,
    PRODUCER_DELAY_SECONDS,
)


def delivery_report(err: Optional[KafkaError], msg) -> None:
    """Callback invocado por librdkafka tras la entrega o fallo de un mensaje."""
    if err is not None:
        print(f"❌ Error entregando mensaje: {err}", file=sys.stderr)


def stream_transactions(
    file_path: Path = PAYSIM_CSV_PATH,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    topic: str = KAFKA_TOPIC_RAW,
    chunk_size: int = PRODUCER_CHUNK_SIZE,
    delay_seconds: float = PRODUCER_DELAY_SECONDS,
    limit: Optional[int] = None,
) -> int:
    """
    Lee transacciones desde el archivo CSV por chunks y las transmite al topic de Kafka.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")

    producer_conf = {
        "bootstrap.servers": bootstrap_servers,
        "client.id": "fraud-transactions-producer",
        "compression.type": "snappy",
        "queue.buffering.max.messages": 100000,
        "linger.ms": 10,
    }
    producer = Producer(producer_conf)

    print(f"🚀 Iniciando productor hacia Redpanda ({bootstrap_servers}) en topic '{topic}'...")
    print(f"📁 Leyendo {file_path.name} (chunk_size={chunk_size}, delay={delay_seconds}s, limit={limit or 'Sin límite'})")

    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
    total_events = 0

    try:
        for chunk_num, chunk in enumerate(chunk_iterator, start=1):
            records = chunk.to_dict(orient="records")

            for record in records:
                try:
                    event = TransactionEvent(**record)
                    # Serializar a JSON
                    payload = event.model_dump_json().encode("utf-8")
                    # Clave por cuenta de origen: garantiza orden cronológico estricto por usuario en la misma partición
                    key = event.nameOrig.encode("utf-8")

                    # Publicar asíncronamente con particionamiento por cuenta origen
                    producer.produce(
                        topic=topic,
                        key=key,
                        value=payload,
                        on_delivery=delivery_report,
                    )
                    producer.poll(0)
                    total_events += 1

                    if total_events <= 5 or total_events % 1000 == 0:
                        print(f"📤 EVT #{total_events} | Tipo: {event.type:<10} | Monto: ${event.amount:>12,.2f} | EsFraude: {event.isFraud}")

                    if limit is not None and total_events >= limit:
                        print(f"🛑 Límite de {limit} eventos alcanzado.")
                        break

                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

                except Exception as e:
                    print(f"⚠️ Error al procesar registro: {e}", file=sys.stderr)

            print(f"✅ Chunk #{chunk_num} ({len(records)} registros) enviado al buffer.")

            if limit is not None and total_events >= limit:
                break

    except KeyboardInterrupt:
        print("\nInterrupción por usuario recibida. Vaciando mensajes pendientes...")
    finally:
        print("⏳ Vaciando buffer del productor (flush)...")
        remaining = producer.flush(timeout=15)
        if remaining > 0:
            print(f"⚠️ {remaining} mensajes no se pudieron entregar a tiempo.", file=sys.stderr)
        else:
            print("✨ Todos los mensajes fueron entregados exitosamente.")

    print(f"🏁 Productor finalizado. Total de eventos transmitidos: {total_events}")
    return total_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Productor de transacciones a Redpanda")
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de eventos a emitir")
    parser.add_argument("--chunk-size", type=int, default=PRODUCER_CHUNK_SIZE, help="Tamaño del chunk de lectura")
    parser.add_argument("--delay", type=float, default=PRODUCER_DELAY_SECONDS, help="Delay en segundos entre eventos")
    args = parser.parse_args()

    stream_transactions(
        limit=args.limit,
        chunk_size=args.chunk_size,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()