"""Módulo productor de eventos de transacciones hacia Redpanda/Kafka."""

from src.producer.producer import stream_transactions

__all__ = ["stream_transactions"]
