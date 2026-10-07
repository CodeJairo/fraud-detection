from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class TransactionEvent(BaseModel):
    """
    Representa el evento transaccional crudo emitido por la simulación PaySim.
    Todos los campos se validan en tiempo de ingesta y el modelo es inmutable (frozen).
    """
    model_config = ConfigDict(frozen=True)

    step: int = Field(..., description="Hora de la simulación (1 step = 1 hora)")
    type: str = Field(..., description="Tipo de transacción (PAYMENT, TRANSFER, CASH_OUT, etc.)")
    amount: float = Field(..., description="Monto gastado o transferido")
    nameOrig: str = Field(..., description="ID cuenta origen")
    oldbalanceOrg: float = Field(..., description="Saldo origen antes de la transacción")
    newbalanceOrig: float = Field(..., description="Saldo origen después de la transacción")
    nameDest: str = Field(..., description="ID cuenta destino")
    oldbalanceDest: float = Field(..., description="Saldo destino antes de la transacción")
    newbalanceDest: float = Field(..., description="Saldo destino después de la transacción")
    isFraud: int = Field(default=0, description="Etiqueta real de fraude (1 = Fraude, 0 = Legítimo)")
    isFlaggedFraud: int = Field(default=0, description="Alerta de regla de negocio (>200k en transferencia ilegal)")


class BronzeTransactionRecord(TransactionEvent):
    """
    Registro almacenado en la Capa Bronze.
    Extiende el evento original preservando fidelidad total y añadiendo metadatos de auditoría.
    """
    ingested_at: datetime = Field(..., description="Timestamp UTC exacto de recepción en la Capa Bronze")
    kafka_partition: int = Field(..., description="Partición del broker Kafka de donde provino el evento")
    kafka_offset: int = Field(..., description="Offset único dentro de la partición")
    kafka_timestamp: Optional[int] = Field(None, description="Timestamp de creación provisto por Kafka")