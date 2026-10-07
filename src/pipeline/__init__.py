"""Módulo de capas de procesamiento de datos (Bronze, Silver, Gold)."""

__all__ = ["BronzeConsumer", "SilverProcessor"]


def __getattr__(name: str):
    if name == "BronzeConsumer":
        from src.pipeline.bronze import BronzeConsumer
        return BronzeConsumer
    if name == "SilverProcessor":
        from src.pipeline.silver import SilverProcessor
        return SilverProcessor
    raise AttributeError(f"module {__name__!r} no tiene el atributo {name!r}")
