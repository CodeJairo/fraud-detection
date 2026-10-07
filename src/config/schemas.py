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
    ml_probability: float = Field(default=0.0, description="Probabilidad de fraude calculada por el modelo de ML (0.0 - 1.0)")
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


class MedallionStatus(BaseModel):
    """
    Métricas de volumen y archivos almacenados en la arquitectura Medallion.
    """
    bronze_files: int = Field(..., description="Archivos Parquet almacenados en la Capa Bronze")
    silver_files: int = Field(..., description="Archivos Parquet almacenados en la Capa Silver")
    total_fraud_alerts: int = Field(..., description="Total de alertas emitidas en la Capa Gold")
    total_user_profiles: int = Field(..., description="Total de perfiles de clientes en la Capa Gold")
    ml_model_active: bool = Field(default=False, description="Indica si el modelo de ML está entrenado y activo")


class HealthResponse(BaseModel):
    """
    Respuesta del health check del sistema y estado del pipeline.
    """
    status: str = Field("healthy", description="Estado de operatividad del servicio")
    medallion_status: MedallionStatus = Field(..., description="Métricas de las capas Medallion")


class EvaluateTransactionRequest(BaseModel):
    """
    Solicitud para evaluar una transacción en tiempo real a través del motor de reglas y ML.
    """
    step: int = Field(default=1, description="Paso o timestamp horario de la transacción (1-744)")
    type: str = Field(default="TRANSFER", description="Tipo de transacción (TRANSFER, CASH_OUT, PAYMENT, etc.)")
    amount: float = Field(default=250000.0, description="Monto involucrado en la transacción")
    nameOrig: str = Field(default="C_DEMO_USER", description="ID de cuenta origen")
    oldbalanceOrg: float = Field(default=250000.0, description="Saldo origen antes de la transacción")
    newbalanceOrig: float = Field(default=0.0, description="Saldo origen después de la transacción")
    nameDest: str = Field(default="M_DESTINATION", description="ID de cuenta destino")
    oldbalanceDest: float = Field(default=0.0, description="Saldo destino antes de la transacción")
    newbalanceDest: float = Field(default=0.0, description="Saldo destino después de la transacción")


class EvaluateTransactionResponse(BaseModel):
    """
    Respuesta de evaluación en tiempo real emitida por el motor híbrido de fraude.
    """
    rules_score: int = Field(..., description="Puntos asignados por las reglas deterministas de negocio (0-100)")
    ml_probability: float = Field(..., description="Probabilidad de fraude calculada por el modelo LightGBM (0.0-1.0)")
    risk_score: int = Field(..., description="Puntaje combinado final de riesgo (0-100)")
    risk_level: str = Field(..., description="Nivel de severidad de riesgo (LOW, MEDIUM, HIGH)")
    rules_triggered: str = Field(..., description="Lista de reglas de negocio violadas")
    recommendation: str = Field(..., description="Acción de mitigación o cumplimiento recomendada")
    decision_color: str = Field(..., description="Código de color hexadecimal para visualización en frontend")