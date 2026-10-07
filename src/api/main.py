"""
Microservicio API REST con FastAPI para la Arquitectura Medallion de Detección de Fraude.
Expone endpoints de salud, consulta de alertas de fraude y perfiles de riesgo de clientes.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
import polars as pl

from src.config.schemas import (
    EvaluateTransactionRequest,
    EvaluateTransactionResponse,
    FraudAlertRecord,
    HealthResponse,
    UserRiskProfileRecord,
)
from src.config.settings import (
    BRONZE_DATA_PATH,
    GOLD_ALERTS_PATH,
    GOLD_LARGE_AMOUNT_THRESHOLD,
    GOLD_ML_WEIGHT,
    GOLD_PROFILES_PATH,
    GOLD_RULES_WEIGHT,
    MODEL_PATH,
    SILVER_DATA_PATH,
)
from src.ml import load_fraud_model
from src.ml.train import FEATURE_COLUMNS

app = FastAPI(
    title="Fintech Fraud Detection API",
    description="API de alta velocidad para consulta de estado del pipeline Medallion, alertas de fraude y perfiles de riesgo en tiempo real.",
    version="1.0.0",
)

# Carga en memoria del modelo para inferencia de baja latencia (<5ms)
ml_model = load_fraud_model(MODEL_PATH)



def read_latest_parquet(directory: Path) -> pl.DataFrame:
    """
    Lee todos los archivos Parquet dentro de un directorio de forma segura.
    Soporta evolución de esquemas entre lotes mediante concatenación diagonal relajada.
    """
    if not directory.exists():
        return pl.DataFrame()
    files = sorted(directory.glob("**/*.parquet"))
    if not files:
        return pl.DataFrame()
    try:
        df = pl.read_parquet(files)
    except Exception:
        try:
            dfs = [pl.read_parquet(f) for f in files]
            df = pl.concat(dfs, how="diagonal_relaxed")
        except Exception as e:
            print(f"⚠️ Error al leer archivos Parquet en {directory}: {e}")
            return pl.DataFrame()

    if "ml_probability" in df.columns:
        df = df.with_columns(pl.col("ml_probability").fill_null(0.0))
    if "rules_score" in df.columns and "risk_score" in df.columns:
        df = df.with_columns(pl.col("rules_score").fill_null(pl.col("risk_score")))
    return df


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health_status():
    """
    Retorna el estado de operatividad del servicio y las métricas de volumen
    almacenado a lo largo de las capas Medallion (Bronze, Silver y Gold),
    incluyendo el estado del modelo de Machine Learning.
    """
    bronze_count = len(list(BRONZE_DATA_PATH.glob("**/*.parquet"))) if BRONZE_DATA_PATH.exists() else 0
    silver_count = len(list(SILVER_DATA_PATH.glob("**/*.parquet"))) if SILVER_DATA_PATH.exists() else 0
    alerts_df = read_latest_parquet(GOLD_ALERTS_PATH)
    profiles_df = read_latest_parquet(GOLD_PROFILES_PATH)
    ml_active = MODEL_PATH.exists()

    return {
        "status": "healthy",
        "medallion_status": {
            "bronze_files": bronze_count,
            "silver_files": silver_count,
            "total_fraud_alerts": len(alerts_df),
            "total_user_profiles": len(profiles_df),
            "ml_model_active": ml_active,
        },
    }


@app.get("/alerts", response_model=List[FraudAlertRecord], tags=["Fraud Analytics"])
def get_fraud_alerts(
    risk_level: Optional[str] = Query(None, description="Filtrar por nivel de riesgo: HIGH, MEDIUM, LOW"),
    rule: Optional[str] = Query(None, description="Filtrar por término contenido en la regla (ej. 'balance', 'burst')"),
    min_ml_prob: Optional[float] = Query(None, ge=0.0, le=1.0, description="Filtrar por probabilidad mínima estimada por ML (0.0 - 1.0)"),
    limit: int = Query(50, ge=1, le=500, description="Cantidad máxima de alertas a retornar"),
    offset: int = Query(0, ge=0, description="Desplazamiento para paginación"),
):
    """
    Lista las transacciones sospechosas detectadas por el motor de reglas y ML de la Capa Gold.
    Permite filtrar por nivel de riesgo, nombre de la regla o umbral de probabilidad de Machine Learning.
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

    if min_ml_prob is not None and "ml_probability" in df.columns:
        df = df.filter(pl.col("ml_probability") >= min_ml_prob)

    return df.slice(offset, limit).to_dicts()


@app.get("/users/{name_orig}/risk-profile", response_model=UserRiskProfileRecord, tags=["User Profiling"])
def get_user_risk_profile(name_orig: str):
    """
    Consulta en tiempo real el perfil acumulado de riesgo de un cliente en la Capa Gold.
    Retorna métricas de volumen transaccional, conteo de alertas y nivel global de riesgo.
    """
    df = read_latest_parquet(GOLD_PROFILES_PATH)
    if df.is_empty():
        raise HTTPException(status_code=404, detail=f"Usuario '{name_orig}' no encontrado en la Capa Gold.")

    user_data = df.filter(pl.col("nameOrig") == name_orig)
    if user_data.is_empty():
        raise HTTPException(status_code=404, detail=f"Usuario '{name_orig}' no encontrado en el sistema.")

    return user_data.to_dicts()[0]


@app.post("/evaluate", response_model=EvaluateTransactionResponse, tags=["Fraud Engine"])
def evaluate_transaction(req: EvaluateTransactionRequest):
    """
    Evalúa una transacción individual en tiempo real combinando el motor de reglas
    deterministas de la Capa Gold y el modelo probabilístico LightGBM.
    Retorna un score combinado (0-100), nivel de severidad y recomendación de compliance.
    """
    # 1. Normalizar tipos de cuenta
    orig_type = "CLIENT" if req.nameOrig.startswith("C") else ("MERCHANT" if req.nameOrig.startswith("M") else "UNKNOWN")
    dest_type = "CLIENT" if req.nameDest.startswith("C") else ("MERCHANT" if req.nameDest.startswith("M") else "UNKNOWN")

    # 2. Calcular errores contables de balance
    bal_err_orig = round((req.oldbalanceOrg - req.amount) - req.newbalanceOrig, 4)
    bal_err_dest = round((req.oldbalanceDest + req.amount) - req.newbalanceDest, 4)

    # 3. Evaluar reglas deterministas
    r1 = (req.type in ["TRANSFER", "CASH_OUT"]) and (abs(bal_err_orig) > 0.01) and (req.amount >= GOLD_LARGE_AMOUNT_THRESHOLD)
    r2 = (req.type == "TRANSFER") and (req.amount >= GOLD_LARGE_AMOUNT_THRESHOLD) and (req.oldbalanceDest == 0.0) and (req.newbalanceDest == 0.0)

    rules_triggered_list = []
    if r1:
        rules_triggered_list.append("SEVERE_BALANCE_ERROR")
    if r2:
        rules_triggered_list.append("LARGE_TRANSFER_ZERO_DEST")

    rules_score = (50 if r1 else 0) + (35 if r2 else 0)

    # 4. Inferencia con LightGBM (si el modelo está cargado)
    ml_prob = 0.0
    global ml_model
    if ml_model is None and MODEL_PATH.exists():
        ml_model = load_fraud_model(MODEL_PATH)

    if ml_model is not None:
        try:
            import pandas as pd
            row_dict = {
                "step": [req.step],
                "amount": [req.amount],
                "oldbalanceOrg": [req.oldbalanceOrg],
                "newbalanceOrig": [req.newbalanceOrig],
                "oldbalanceDest": [req.oldbalanceDest],
                "newbalanceDest": [req.newbalanceDest],
                "balance_error_orig": [bal_err_orig],
                "balance_error_dest": [bal_err_dest],
                "type": [req.type],
                "orig_account_type": [orig_type],
                "dest_account_type": [dest_type],
            }
            pdf = pd.DataFrame(row_dict)[FEATURE_COLUMNS]
            ml_prob = round(float(ml_model.predict_proba(pdf)[0, 1]), 4)
        except Exception as e:
            print(f"⚠️ Error en inferencia en tiempo real: {e}")
            ml_prob = 0.0

    # 5. Score híbrido combinado
    risk_score = round(GOLD_RULES_WEIGHT * rules_score + GOLD_ML_WEIGHT * (ml_prob * 100.0))

    # 6. Severidad y Recomendación
    if risk_score >= 60:
        risk_level = "HIGH"
        recommendation = "BLOQUEO PREVENTIVO INMEDIATO: Alto riesgo de fraude financiero detectado."
        decision_color = "#EF553B"
    elif risk_score >= 35:
        risk_level = "MEDIUM"
        recommendation = "REVISIÓN MANUAL REQUERIDA: Solicitar autenticación reforzada o 2FA."
        decision_color = "#FFA15A"
    elif risk_score > 0:
        risk_level = "LOW"
        recommendation = "OPERACIÓN MONITOREADA: Riesgo leve pero dentro de parámetros tolerables."
        decision_color = "#636EFA"
    else:
        risk_level = "NONE"
        recommendation = "OPERACIÓN APROBADA: No se detectaron anomalías contables ni sospecha de ML."
        decision_color = "#00CC96"

    return {
        "rules_score": rules_score,
        "ml_probability": ml_prob,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "rules_triggered": ", ".join(rules_triggered_list) if rules_triggered_list else "NINGUNA",
        "recommendation": recommendation,
        "decision_color": decision_color,
    }

