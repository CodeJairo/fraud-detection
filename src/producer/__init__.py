"""Módulo productor de eventos de transacciones hacia Redpanda/Kafka."""

__all__ = ["stream_transactions"]


def __getattr__(name: str):
    if name == "stream_transactions":
        from src.producer.producer import stream_transactions
        return stream_transactions
    raise AttributeError(f"module {__name__!r} no tiene el atributo {name!r}")
