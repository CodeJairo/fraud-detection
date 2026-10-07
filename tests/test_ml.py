from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.ml import load_fraud_model
from src.ml.train import (
    FEATURE_COLUMNS,
    build_pipeline,
    save_artifacts,
    train_model,
)


@pytest.fixture
def synthetic_training_data() -> pl.DataFrame:
    """Genera un dataset sintético representativo para validar el entrenamiento de ML."""
    np.random.seed(42)
    n = 100
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    acc_types = ["CLIENT", "MERCHANT"]

    # 90 legítimas, 10 fraudes
    is_fraud = [0] * 90 + [1] * 10

    return pl.DataFrame({
        "step": list(range(1, n + 1)),
        "type": [types[i % len(types)] for i in range(n)],
        "amount": [float(100.0 * (i + 1)) for i in range(n)],
        "oldbalanceOrg": [float(500.0 * (i + 1)) for i in range(n)],
        "newbalanceOrig": [float(400.0 * (i + 1)) for i in range(n)],
        "oldbalanceDest": [float(200.0 * (i + 1)) for i in range(n)],
        "newbalanceDest": [float(300.0 * (i + 1)) for i in range(n)],
        "balance_error_orig": [0.0 if i < 90 else -5000.0 for i in range(n)],
        "balance_error_dest": [0.0 if i < 90 else 5000.0 for i in range(n)],
        "orig_account_type": [acc_types[i % 2] for i in range(n)],
        "dest_account_type": [acc_types[(i + 1) % 2] for i in range(n)],
        "isFraud": is_fraud,
    })


def test_build_pipeline_structure():
    """Valida la arquitectura del pipeline de Scikit-Learn y LightGBM."""
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_model_metrics(synthetic_training_data):
    """Valida la ejecución del entrenamiento y la generación de métricas."""
    pipeline, metrics = train_model(synthetic_training_data, test_size=0.2, random_state=42)

    assert metrics["train_samples"] == 80
    assert metrics["test_samples"] == 20
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert len(metrics["features"]) == len(FEATURE_COLUMNS)
    assert len(metrics["confusion_matrix"]) == 2


def test_save_and_load_artifacts(synthetic_training_data, tmp_path):
    """Valida la persistencia y carga segura del modelo serializado."""
    pipeline, metrics = train_model(synthetic_training_data, test_size=0.2, random_state=42)

    model_file = tmp_path / "models" / "fraud_model.joblib"
    metrics_file = tmp_path / "models" / "model_metrics.json"

    save_artifacts(pipeline, metrics, model_path=model_file, metrics_path=metrics_file)

    assert model_file.exists()
    assert metrics_file.exists()

    loaded_model = load_fraud_model(model_file)
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict_proba")


def test_inference_probability_bounds(synthetic_training_data):
    """Verifica que las probabilidades estimadas pertenezcan estrictamente al rango [0.0, 1.0]."""
    pipeline, _ = train_model(synthetic_training_data, test_size=0.2, random_state=42)

    features_pdf = synthetic_training_data.select(FEATURE_COLUMNS).to_pandas()
    probs = pipeline.predict_proba(features_pdf)[:, 1]

    assert len(probs) == len(synthetic_training_data)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_handling_unseen_categorical_levels(synthetic_training_data):
    """Verifica la robustez del encoder ante valores categóricos no observados en el entrenamiento."""
    pipeline, _ = train_model(synthetic_training_data, test_size=0.2, random_state=42)

    unseen_df = pl.DataFrame({
        "step": [999],
        "type": ["NEW_UNKNOWN_TYPE"],
        "amount": [50000.0],
        "oldbalanceOrg": [50000.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [0.0],
        "balance_error_orig": [0.0],
        "balance_error_dest": [0.0],
        "orig_account_type": ["UNKNOWN_TYPE"],
        "dest_account_type": ["UNKNOWN_TYPE"],
    })

    pdf = unseen_df.select(FEATURE_COLUMNS).to_pandas()
    # No debe arrojar excepción gracias a handle_unknown="ignore"
    probs = pipeline.predict_proba(pdf)[:, 1]
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0
