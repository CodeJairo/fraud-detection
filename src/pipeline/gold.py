"""
Pipeline de la Capa Gold: Métricas de Negocio, Inferencia ML y Motor de Reglas de Fraude.
Aplica reglas contables, inferencia con LightGBM, genera alertas de fraude y perfiles de riesgo por usuario.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Optional, Tuple
import uuid

import polars as pl

from src.config.settings import (
    GOLD_ALERT_MIN_SCORE,
    GOLD_ALERTS_PATH,
    GOLD_BURST_THRESHOLD,
    GOLD_COMPRESSION,
    GOLD_LARGE_AMOUNT_THRESHOLD,
    GOLD_ML_WEIGHT,
    GOLD_PROFILES_PATH,
    GOLD_RULES_WEIGHT,
    MODEL_PATH,
    SILVER_DATA_PATH,
)


class GoldProcessor:
    """
    Motor Híbrido de Detección de Fraude y Agregaciones Analíticas de la Capa Gold.
    Combina reglas deterministas contables con inferencia probabilística de Machine Learning (LightGBM).
    """

    def __init__(
        self,
        silver_path: Path = SILVER_DATA_PATH,
        alerts_path: Path = GOLD_ALERTS_PATH,
        profiles_path: Path = GOLD_PROFILES_PATH,
        compression: str = GOLD_COMPRESSION,
        large_amount_threshold: float = GOLD_LARGE_AMOUNT_THRESHOLD,
        burst_threshold: int = GOLD_BURST_THRESHOLD,
        min_alert_score: int = GOLD_ALERT_MIN_SCORE,
        model_path: Optional[Path] = MODEL_PATH,
        rules_weight: float = GOLD_RULES_WEIGHT,
        ml_weight: float = GOLD_ML_WEIGHT,
    ):
        self.silver_path = Path(silver_path)
        self.alerts_path = Path(alerts_path)
        self.profiles_path = Path(profiles_path)
        self.compression = compression
        self.large_amount_threshold = large_amount_threshold
        self.burst_threshold = burst_threshold
        self.min_alert_score = min_alert_score
        self.model_path = Path(model_path) if model_path else None
        self.rules_weight = rules_weight
        self.ml_weight = ml_weight

        # Cargar modelo entrenado si existe
        self.model = None
        if self.model_path and self.model_path.exists():
            try:
                import joblib
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"⚠️ [GOLD] No se pudo cargar el modelo de ML desde {self.model_path}: {e}")

    def scan_silver_data(self, partition_date: Optional[str] = None) -> pl.LazyFrame:
        """Escanea los archivos Parquet de la Capa Silver como un Polars LazyFrame."""
        if partition_date:
            target_pattern = self.silver_path / f"ingestion_date={partition_date}" / "*.parquet"
            files = list(self.silver_path.glob(f"ingestion_date={partition_date}/*.parquet"))
        else:
            target_pattern = self.silver_path / "**" / "*.parquet"
            files = list(self.silver_path.glob("**/*.parquet"))

        if not files:
            raise FileNotFoundError(f"No se encontraron archivos Parquet en Silver para el patrón: {target_pattern}")

        return pl.scan_parquet(str(target_pattern))

    def evaluate_rules(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Evalúa vectorizadamente las reglas de fraude de negocio:
        - Regla 1 (Inconsistencia Severa): TRANSFER/CASH_OUT con error de balance en montos elevados (50 pts).
        - Regla 2 (Transferencia a Cero): TRANSFER de alto monto a cuenta con balance en cero (35 pts).
        - Regla 3 (Ráfaga / Frecuencia): Múltiples operaciones del mismo usuario en la misma hora (step) (25 pts).
        """
        lf_with_rules = lf.with_columns([
            # Regla 1: Inconsistencia contable en transacciones relevantes (50 pts)
            (
                pl.col("type").is_in(["TRANSFER", "CASH_OUT"])
                & (pl.col("balance_error_orig").abs() > 0.01)
                & (pl.col("amount") >= self.large_amount_threshold)
            ).alias("rule_severe_balance_error"),

            # Regla 2: Transferencia grande a cuenta destino sin saldo registrado (35 pts)
            (
                (pl.col("type") == "TRANSFER")
                & (pl.col("amount") >= self.large_amount_threshold)
                & (pl.col("oldbalanceDest") == 0.0)
                & (pl.col("newbalanceDest") == 0.0)
            ).alias("rule_large_zero_dest"),

            # Regla 3: Ráfaga de transacciones en la misma hora (25 pts)
            (
                pl.len().over(["nameOrig", "step"]) >= self.burst_threshold
            ).alias("rule_burst_frequency"),
        ])

        # Calcular Score de Reglas puro (0 - 100)
        lf_scored = lf_with_rules.with_columns([
            (
                pl.col("rule_severe_balance_error").cast(pl.Int32) * 50
                + pl.col("rule_large_zero_dest").cast(pl.Int32) * 35
                + pl.col("rule_burst_frequency").cast(pl.Int32) * 25
            ).alias("rules_score"),
            pl.lit(0.0).alias("ml_probability"),
        ]).with_columns([
            pl.col("rules_score").alias("risk_score"),
        ])

        # Asignar nivel de severidad y concatenar reglas violadas
        lf_categorized = lf_scored.with_columns([
            pl.when(pl.col("risk_score") >= 60).then(pl.lit("HIGH"))
              .when(pl.col("risk_score") >= 35).then(pl.lit("MEDIUM"))
              .when(pl.col("risk_score") > 0).then(pl.lit("LOW"))
              .otherwise(pl.lit("NONE"))
              .alias("risk_level"),

            pl.concat_list([
                pl.when(pl.col("rule_severe_balance_error")).then(pl.lit("SEVERE_BALANCE_ERROR")),
                pl.when(pl.col("rule_large_zero_dest")).then(pl.lit("LARGE_TRANSFER_ZERO_DEST")),
                pl.when(pl.col("rule_burst_frequency")).then(pl.lit("HIGH_FREQUENCY_BURST")),
            ]).list.drop_nulls().list.join(", ").alias("rules_triggered"),
        ])

        return lf_categorized

    def apply_ml_inference(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Aplica inferencia probabilística usando el modelo LightGBM preentrenado (si está disponible).
        Calcula el score híbrido combinando las reglas deterministas y la probabilidad de ML:
            risk_score = round(rules_weight * rules_score + ml_weight * (ml_probability * 100))
        """
        if self.model is None or len(df) == 0:
            return df

        try:
            from src.ml.train import FEATURE_COLUMNS
            # Validar que existan todas las features necesarias
            if not all(col in df.columns for col in FEATURE_COLUMNS):
                return df

            pdf = df.select(FEATURE_COLUMNS).to_pandas()
            probs = self.model.predict_proba(pdf)[:, 1]
            probs_rounded = [round(float(p), 4) for p in probs]

            df_with_ml = df.with_columns(pl.Series("ml_probability", probs_rounded))

            # Calcular score combinado ponderado
            df_scored = df_with_ml.with_columns([
                (
                    self.rules_weight * pl.col("rules_score")
                    + self.ml_weight * (pl.col("ml_probability") * 100.0)
                ).round(0).cast(pl.Int32).alias("risk_score")
            ])

            # Recalcular niveles de riesgo según el nuevo puntaje híbrido
            df_final = df_scored.with_columns([
                pl.when(pl.col("risk_score") >= 60).then(pl.lit("HIGH"))
                  .when(pl.col("risk_score") >= 35).then(pl.lit("MEDIUM"))
                  .when(pl.col("risk_score") > 0).then(pl.lit("LOW"))
                  .otherwise(pl.lit("NONE"))
                  .alias("risk_level")
            ])
            return df_final
        except Exception as e:
            print(f"⚠️ [GOLD] Error en inferencia de ML: {e}", file=sys.stderr)
            return df

    def extract_fraud_alerts(self, df_evaluated: pl.DataFrame) -> pl.DataFrame:
        """Filtra y estructura las transacciones que superan el umbral de alerta."""
        df_alerts = df_evaluated.filter(pl.col("risk_score") >= self.min_alert_score)

        if len(df_alerts) == 0:
            return pl.DataFrame()

        alert_ids = [uuid.uuid4().hex[:12] for _ in range(len(df_alerts))]
        now_utc = datetime.now(timezone.utc).isoformat()

        df_alerts = df_alerts.with_columns([
            pl.Series("alert_id", alert_ids),
            pl.lit(now_utc).alias("alert_timestamp"),
        ])

        # Seleccionar columnas clave de la alerta incluyendo probabilidad de ML
        columns_order = [
            "alert_id", "step", "type", "amount", "nameOrig", "nameDest",
            "orig_account_type", "dest_account_type", "rules_triggered",
            "rules_score", "ml_probability", "risk_score", "risk_level",
            "isFraud", "kafka_offset", "alert_timestamp", "ingestion_date"
        ]
        return df_alerts.select([col for col in columns_order if col in df_alerts.columns])

    def build_user_risk_profiles(self, df_evaluated: pl.DataFrame) -> pl.DataFrame:
        """Construye la vista analítica agregada de perfil de riesgo por usuario (nameOrig)."""
        if len(df_evaluated) == 0:
            return pl.DataFrame()

        # Agrupar por cuenta de origen
        df_profiles = df_evaluated.group_by("nameOrig").agg([
            pl.col("orig_account_type").first().alias("account_type"),
            pl.len().cast(pl.Int64).alias("total_transactions"),
            pl.col("amount").sum().round(2).alias("total_volume"),
            pl.col("amount").mean().round(2).alias("avg_transaction_amount"),
            pl.col("amount").max().round(2).alias("max_transaction_amount"),
            (pl.col("type") == "TRANSFER").sum().cast(pl.Int64).alias("transfer_count"),
            (pl.col("type") == "CASH_OUT").sum().cast(pl.Int64).alias("cash_out_count"),
            (pl.col("risk_score") >= self.min_alert_score).sum().cast(pl.Int64).alias("total_alerts"),
            (pl.col("risk_level") == "HIGH").sum().cast(pl.Int64).alias("high_risk_alerts"),
            pl.col("balance_error_orig").abs().mean().round(2).alias("avg_balance_error"),
            pl.col("step").max().cast(pl.Int64).alias("last_activity_step"),
            pl.col("ingestion_date").first().alias("ingestion_date"),
        ])

        now_utc = datetime.now(timezone.utc).isoformat()

        # Asignar nivel de riesgo global del usuario
        df_profiles = df_profiles.with_columns([
            pl.when((pl.col("high_risk_alerts") > 0) | (pl.col("total_alerts") >= 2)).then(pl.lit("HIGH"))
              .when((pl.col("total_alerts") == 1) | (pl.col("total_volume") >= 500000.0)).then(pl.lit("MEDIUM"))
              .otherwise(pl.lit("LOW"))
              .alias("user_risk_level"),
            pl.lit(now_utc).alias("profile_updated_at"),
        ])

        return df_profiles

    def process(self, partition_date: Optional[str] = None) -> Tuple[int, int]:
        """
        Ejecuta el procesamiento de la Capa Gold:
        1. Evalúa reglas de fraude sobre Silver.
        2. Aplica inferencia de ML e integra el score híbrido.
        3. Genera y guarda fraud_alerts.
        4. Genera y guarda user_risk_profile.
        """
        print(f"🏆 [GOLD] Iniciando procesamiento (fecha: {partition_date or 'Todas las disponibles'})...")
        print(f"📂 Origen Silver: {self.silver_path}")
        print(f"🚨 Destino Alertas: {self.alerts_path}")
        print(f"👤 Destino Perfiles: {self.profiles_path}")
        if self.model is not None:
            print(f"🧠 [GOLD] Modelo ML Activo: {self.model_path} (Ponderación: {self.rules_weight*100:.0f}% Reglas / {self.ml_weight*100:.0f}% ML)")
        else:
            print("ℹ️ [GOLD] Modelo ML inactivo o no encontrado. Evaluando con reglas puras.")

        try:
            lf_silver = self.scan_silver_data(partition_date)
        except FileNotFoundError as e:
            print(f"⚠️ {e}", file=sys.stderr)
            return (0, 0)

        # 1. Evaluar reglas deterministas
        lf_evaluated = self.evaluate_rules(lf_silver)
        df_evaluated = lf_evaluated.collect()

        total_transactions = len(df_evaluated)
        if total_transactions == 0:
            print("⚠️ [GOLD] No se encontraron registros en Silver para procesar.")
            return (0, 0)

        # 2. Inferencia de ML e integración híbrida
        df_evaluated = self.apply_ml_inference(df_evaluated)

        # 3. Alertas de Fraude
        df_alerts = self.extract_fraud_alerts(df_evaluated)
        alerts_count = len(df_alerts)

        if alerts_count > 0:
            unique_dates = df_alerts["ingestion_date"].unique().to_list()
            for date_val in unique_dates:
                date_str = str(date_val)
                out_dir = self.alerts_path / f"ingestion_date={date_str}"
                out_dir.mkdir(parents=True, exist_ok=True)

                df_part = df_alerts.filter(pl.col("ingestion_date") == date_val)
                batch_id = uuid.uuid4().hex[:8]
                ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                file_path = out_dir / f"alerts_{ts_str}_{batch_id}.parquet"
                df_part.write_parquet(file_path, compression=self.compression)
                print(f"🚨 [GOLD] {len(df_part)} alertas guardadas en {file_path.name}")
        else:
            print("ℹ️ [GOLD] No se generaron alertas que superen el umbral.")

        # 4. Perfiles de Riesgo de Usuario
        df_profiles = self.build_user_risk_profiles(df_evaluated)
        profiles_count = len(df_profiles)

        if profiles_count > 0:
            unique_dates = df_profiles["ingestion_date"].unique().to_list()
            for date_val in unique_dates:
                date_str = str(date_val)
                out_dir = self.profiles_path / f"ingestion_date={date_str}"
                out_dir.mkdir(parents=True, exist_ok=True)

                df_part = df_profiles.filter(pl.col("ingestion_date") == date_val)
                batch_id = uuid.uuid4().hex[:8]
                ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                file_path = out_dir / f"profiles_{ts_str}_{batch_id}.parquet"
                df_part.write_parquet(file_path, compression=self.compression)
                print(f"👤 [GOLD] {len(df_part)} perfiles de usuario guardados en {file_path.name}")

        print(f"✨ [GOLD] Procesamiento finalizado. Transacciones: {total_transactions} | Alertas: {alerts_count} | Perfiles: {profiles_count}")
        return (alerts_count, profiles_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesador de la Capa Gold (Reglas de Fraude, ML y Perfiles)")
    parser.add_argument("--date", type=str, default=None, help="Fecha de ingesta a procesar (YYYY-MM-DD)")
    parser.add_argument("--silver-path", type=str, default=str(SILVER_DATA_PATH), help="Ruta de archivos Silver")
    parser.add_argument("--alerts-path", type=str, default=str(GOLD_ALERTS_PATH), help="Ruta de alertas Gold")
    parser.add_argument("--profiles-path", type=str, default=str(GOLD_PROFILES_PATH), help="Ruta de perfiles Gold")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH), help="Ruta del modelo de ML")
    parser.add_argument("--rules-weight", type=float, default=GOLD_RULES_WEIGHT, help="Ponderación de reglas (0.0 - 1.0)")
    parser.add_argument("--ml-weight", type=float, default=GOLD_ML_WEIGHT, help="Ponderación de ML (0.0 - 1.0)")
    args = parser.parse_args()

    processor = GoldProcessor(
        silver_path=Path(args.silver_path),
        alerts_path=Path(args.alerts_path),
        profiles_path=Path(args.profiles_path),
        model_path=Path(args.model_path) if args.model_path else None,
        rules_weight=args.rules_weight,
        ml_weight=args.ml_weight,
    )
    processor.process(partition_date=args.date)


if __name__ == "__main__":
    main()
