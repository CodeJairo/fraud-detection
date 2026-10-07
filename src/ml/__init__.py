"""
Módulo de Machine Learning para Detección de Fraude.
Contiene el pipeline de entrenamiento (LightGBM) y utilidades de inferencia.
"""

from pathlib import Path
from typing import Any, Optional

import joblib

from src.config.settings import MODEL_PATH


def load_fraud_model(model_path: Optional[Path] = None) -> Optional[Any]:
    """
    Carga de forma segura el pipeline de Machine Learning entrenado.
    Retorna None si el archivo no existe.
    """
    target_path = Path(model_path) if model_path else MODEL_PATH
    if not target_path.exists():
        return None
    try:
        return joblib.load(target_path)
    except Exception as e:
        print(f"⚠️ Error al cargar modelo desde {target_path}: {e}")
        return None
