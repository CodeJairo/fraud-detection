"""
Microservicio API REST con FastAPI para la Arquitectura Medallion de Detección de Fraude.
Expone endpoints de salud, consulta de alertas de fraude y perfiles de riesgo de clientes.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
import polars as pl

from src.config.schemas import FraudAlertRecord, HealthResponse, UserRiskProfileRecord
from src.config.settings import (
    BRONZE_DATA_PATH,
    GOLD_ALERTS_PATH,
    GOLD_PROFILES_PATH,
    SILVER_DATA_PATH,
)

app = FastAPI(
    title="Fintech Fraud Detection API",
    description="API de alta velocidad para consulta de estado del pipeline Medallion, alertas de fraude y perfiles de riesgo en tiempo real.",
    version="1.0.0",
)


def read_latest_parquet(directory: Path) -> pl.DataFrame:
    """
    Lee todos los archivos Parquet dentro de un directorio de forma segura.
    Retorna un DataFrame vacío si el directorio no existe o no contiene archivos.
    """
    if not directory.exists():
        return pl.DataFrame()
    files = list(directory.glob("**/*.parquet"))
    if not files:
        return pl.DataFrame()
    try:
        return pl.read_parquet(files)
    except Exception:
        return pl.DataFrame()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health_status():
    """
    Retorna el estado de operatividad del servicio y las métricas de volumen
    almacenado a lo largo de las capas Medallion (Bronze, Silver y Gold).
    """
    bronze_count = len(list(BRONZE_DATA_PATH.glob("**/*.parquet"))) if BRONZE_DATA_PATH.exists() else 0
    silver_count = len(list(SILVER_DATA_PATH.glob("**/*.parquet"))) if SILVER_DATA_PATH.exists() else 0
    alerts_df = read_latest_parquet(GOLD_ALERTS_PATH)
    profiles_df = read_latest_parquet(GOLD_PROFILES_PATH)

    return {
        "status": "healthy",
        "medallion_status": {
            "bronze_files": bronze_count,
            "silver_files": silver_count,
            "total_fraud_alerts": len(alerts_df),
            "total_user_profiles": len(profiles_df),
        },
    }


@app.get("/alerts", response_model=List[FraudAlertRecord], tags=["Fraud Analytics"])
def get_fraud_alerts(
    risk_level: Optional[str] = Query(None, description="Filtrar por nivel de riesgo: HIGH, MEDIUM, LOW"),
    rule: Optional[str] = Query(None, description="Filtrar por término contenido en la regla (ej. 'balance', 'burst')"),
    limit: int = Query(50, ge=1, le=500, description="Cantidad máxima de alertas a retornar"),
    offset: int = Query(0, ge=0, description="Desplazamiento para paginación"),
):
    """
    Lista las transacciones sospechosas detectadas por el motor de reglas de la Capa Gold.
    Permite filtrar por nivel de riesgo o coincidencia parcial con el nombre de la regla.
    """
    df = read_latest_parquet(GOLD_ALERTS_PATH)
    if df.is_empty():
        return []

    if risk_level:
        df = df.filter(pl.col("risk_level") == risk_level.upper())

    if rule:
        # Búsqueda insensible a mayúsculas que contenga el término solicitado
        pattern = f"(?i){rule.strip()}"
        df = df.filter(pl.col("rules_triggered").str.contains(pattern))

    return df.slice(offset, limit).to_dicts()


@app.get("/users/{name_orig}/risk-profile", response_model=UserRiskProfileRecord, tags=["User Profiling"])
def get_user_risk_profile(name_orig: str):
    """
    Consulta en tiempo real el perfil acumulado de riesgo de un cliente en la Capa Gold.
    Retorna métricas de volumen transaccional, conteo de alertas y nivel global de riesgo.
    """
    df = read_latest_parquet(GOLD_PROFILES_PATH)
    if df.is_empty():
        raise HTTPException(status_code=404, detail="No se encontraron perfiles de usuario en la Capa Gold.")

    user_data = df.filter(pl.col("nameOrig") == name_orig)
    if user_data.is_empty():
        raise HTTPException(status_code=404, detail=f"Usuario '{name_orig}' no encontrado en el sistema.")

    return user_data.to_dicts()[0]
