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


class SilverTransactionRecord(BronzeTransactionRecord):
    """
    Registro transformado en la Capa Silver.
    Normalizado, validado y enriquecido con variables contables y de auditoría.
    """
    orig_account_type: str = Field(..., description="Tipo de cuenta origen (CLIENT, MERCHANT o UNKNOWN)")
    dest_account_type: str = Field(..., description="Tipo de cuenta destino (CLIENT, MERCHANT o UNKNOWN)")
    balance_error_orig: float = Field(..., description="Inconsistencia de saldo origen: (oldbalanceOrg - amount) - newbalanceOrig")
    balance_error_dest: float = Field(..., description="Inconsistencia de saldo destino: (oldbalanceDest + amount) - newbalanceDest")
    silver_processed_at: datetime = Field(..., description="Timestamp UTC del procesamiento en la Capa Silver")


class FraudAlertRecord(BaseModel):
    """
    Representa una alerta de fraude emitida por el motor de reglas en la Capa Gold.
    """
    model_config = ConfigDict(frozen=True)

    alert_id: str = Field(..., description="Identificador único UUID de la alerta")
    step: int = Field(..., description="Paso de la simulación temporal")
    type: str = Field(..., description="Tipo de transacción")
    amount: float = Field(..., description="Monto involucrado")
    nameOrig: str = Field(..., description="ID cuenta origen")
    nameDest: str = Field(..., description="ID cuenta destino")
    orig_account_type: str = Field(..., description="Tipo cuenta origen (CLIENT / MERCHANT)")
    dest_account_type: str = Field(..., description="Tipo cuenta destino (CLIENT / MERCHANT)")
    rules_triggered: str = Field(..., description="Reglas violadas separadas por comas")
    risk_score: int = Field(..., description="Puntaje de riesgo calculado (0-100)")
    risk_level: str = Field(..., description="Nivel de severidad de riesgo (LOW, MEDIUM, HIGH)")
    isFraud: int = Field(default=0, description="Etiqueta real de fraude para contraste")
    kafka_offset: int = Field(..., description="Offset de trazabilidad en Kafka")
    alert_timestamp: datetime = Field(..., description="Timestamp UTC de generación de la alerta")


class UserRiskProfileRecord(BaseModel):
    """
    Perfil de riesgo agregado por usuario en la Capa Gold.
    """
    model_config = ConfigDict(frozen=True)

    nameOrig: str = Field(..., description="Identificador de la cuenta de usuario")
    account_type: str = Field(..., description="Tipo de cuenta (CLIENT / MERCHANT)")
    total_transactions: int = Field(..., description="Conteo total de transacciones realizadas")
    total_volume: float = Field(..., description="Monto acumulado total transaccionado")
    avg_transaction_amount: float = Field(..., description="Monto promedio por transacción")
    max_transaction_amount: float = Field(..., description="Monto máximo transaccionado")
    transfer_count: int = Field(..., description="Cantidad de operaciones tipo TRANSFER")
    cash_out_count: int = Field(..., description="Cantidad de operaciones tipo CASH_OUT")
    total_alerts: int = Field(..., description="Total de alertas de fraude generadas para esta cuenta")
    high_risk_alerts: int = Field(..., description="Alertas de severidad HIGH")
    avg_balance_error: float = Field(..., description="Promedio del valor absoluto del error de balance")
    user_risk_level: str = Field(..., description="Clasificación general de riesgo del cliente (LOW, MEDIUM, HIGH)")
    last_activity_step: int = Field(..., description="Último step de actividad registrado")
    profile_updated_at: datetime = Field(..., description="Timestamp UTC de actualización del perfil")