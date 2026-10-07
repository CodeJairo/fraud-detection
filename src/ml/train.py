"""
Script de Entrenamiento del Modelo de Detección de Fraude con LightGBM.
Entrena un clasificador supervisado sobre la Capa Silver (o PaySim CSV),
evalúa métricas clave (PR-AUC, ROC-AUC) y serializa el artefacto en models/fraud_model.joblib.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config.settings import (
    MODEL_METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PAYSIM_CSV_PATH,
    SILVER_DATA_PATH,
)

NUMERICAL_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_error_orig",
    "balance_error_dest",
]

CATEGORICAL_FEATURES = [
    "type",
    "orig_account_type",
    "dest_account_type",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "isFraud"


def build_pipeline() -> Pipeline:
    """
    Construye el Pipeline de Scikit-Learn:
    - Preprocesamiento: Codificación One-Hot para categóricas y passthrough para numéricas.
    - Clasificador: LightGBM con balanceo automático de clases para datos desbalanceados.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    classifier = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def load_dataset_from_silver(silver_path: Path) -> pl.DataFrame:
    """Carga los archivos Parquet de la Capa Silver."""
    files = list(silver_path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos Parquet en {silver_path}")
    return pl.read_parquet(files)


def load_dataset_from_csv(csv_path: Path, sample_size: Optional[int] = 50000) -> pl.DataFrame:
    """
    Carga y enriquece una muestra del archivo crudo PaySim CSV
    generando las mismas variables que la Capa Silver.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en {csv_path}")

    lf = pl.scan_csv(str(csv_path))
    if sample_size:
        lf = lf.slice(0, sample_size)

    # Replicar transformaciones de Silver
    lf = lf.with_columns([
        pl.when(pl.col("nameOrig").str.starts_with("C")).then(pl.lit("CLIENT"))
          .when(pl.col("nameOrig").str.starts_with("M")).then(pl.lit("MERCHANT"))
          .otherwise(pl.lit("UNKNOWN")).alias("orig_account_type"),

        pl.when(pl.col("nameDest").str.starts_with("C")).then(pl.lit("CLIENT"))
          .when(pl.col("nameDest").str.starts_with("M")).then(pl.lit("MERCHANT"))
          .otherwise(pl.lit("UNKNOWN")).alias("dest_account_type"),

        ((pl.col("oldbalanceOrg") - pl.col("amount")) - pl.col("newbalanceOrig")).round(4).alias("balance_error_orig"),
        ((pl.col("oldbalanceDest") + pl.col("amount")) - pl.col("newbalanceDest")).round(4).alias("balance_error_dest"),
    ])
    return lf.collect()


def train_model(
    df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Ejecuta el split, entrenamiento y evaluación del pipeline.
    """
    # Verificar que existan las columnas necesarias
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    # Extraer features y etiquetas
    pdf = df.select(FEATURE_COLUMNS + [TARGET_COLUMN]).to_pandas()
    X = pdf[FEATURE_COLUMNS]
    y = pdf[TARGET_COLUMN].astype(int)

    positives = int(y.sum())
    total_samples = len(y)
    stratify = y if positives >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Inferencia probabilística en prueba
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Métricas
    has_both_classes = len(np.unique(y_test)) > 1
    roc_auc = float(roc_auc_score(y_test, y_pred_proba)) if has_both_classes else 1.0
    pr_auc = float(average_precision_score(y_test, y_pred_proba)) if has_both_classes else 1.0
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": total_samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_samples": positives,
        "fraud_rate": round(positives / total_samples, 6) if total_samples > 0 else 0.0,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "features": FEATURE_COLUMNS,
    }

    return pipeline, metrics


def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    model_path: Path = MODEL_PATH,
    metrics_path: Path = MODEL_METRICS_PATH,
) -> None:
    """Guarda el artefacto serializado del modelo y sus métricas asociadas."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"💾 Modelo guardado exitosamente en: {model_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Métricas guardadas en: {metrics_path}")


def run_training_pipeline(
    silver_path: Path = SILVER_DATA_PATH,
    csv_path: Optional[Path] = None,
    sample_size: Optional[int] = None,
    output_model: Path = MODEL_PATH,
    output_metrics: Path = MODEL_METRICS_PATH,
) -> Dict[str, Any]:
    """Flujo principal de orquestación de entrenamiento."""
    print("🤖 [ML] Iniciando pipeline de entrenamiento de detección de fraude...")

    # Cargar datos desde Silver o fallback a CSV
    df = None
    if csv_path and csv_path.exists():
        print(f"📂 Cargando datos desde CSV: {csv_path} (muestra: {sample_size or 'todas'})")
        df = load_dataset_from_csv(csv_path, sample_size=sample_size)
    else:
        try:
            print(f"📂 Cargando datos desde Capa Silver: {silver_path}")
            df = load_dataset_from_silver(silver_path)
        except FileNotFoundError:
            if PAYSIM_CSV_PATH.exists():
                print(f"⚠️ Capa Silver no encontrada. Fallback a PaySim CSV: {PAYSIM_CSV_PATH}")
                df = load_dataset_from_csv(PAYSIM_CSV_PATH, sample_size=sample_size or 50000)
            else:
                raise FileNotFoundError("No se encontraron fuentes de datos (ni Silver Parquet ni PaySim CSV).")

    print(f"📦 Registros cargados: {len(df):,} | Fraudes etiquetados: {int(df['isFraud'].sum())}")

    # Entrenar y evaluar
    pipeline, metrics = train_model(df)

    print("\n📈 [ML] Resultados de Evaluación en Test Set:")
    print(f"   - ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"   - PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"   - F1-Score: {metrics['f1_score']:.4f}")
    print(f"   - Matriz de Confusión: {metrics['confusion_matrix']}")

    # Guardar
    save_artifacts(pipeline, metrics, model_path=output_model, metrics_path=output_metrics)
    print("✨ [ML] Proceso de entrenamiento finalizado con éxito.\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenador de Modelo de Fraude con LightGBM")
    parser.add_argument("--silver-path", type=str, default=str(SILVER_DATA_PATH), help="Ruta de archivos Silver Parquet")
    parser.add_argument("--csv-path", type=str, default=None, help="Ruta opcional a CSV PaySim para muestras mayores")
    parser.add_argument("--sample-size", type=int, default=None, help="Tamaño de muestra a cargar si se usa CSV")
    parser.add_argument("--output-model", type=str, default=str(MODEL_PATH), help="Ruta de salida del modelo .joblib")
    parser.add_argument("--output-metrics", type=str, default=str(MODEL_METRICS_PATH), help="Ruta de salida de métricas .json")
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else None
    run_training_pipeline(
        silver_path=Path(args.silver_path),
        csv_path=csv_path,
        sample_size=args.sample_size,
        output_model=Path(args.output_model),
        output_metrics=Path(args.output_metrics),
    )


if __name__ == "__main__":
    main()
