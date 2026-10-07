"""
Pipeline de la Capa Gold: Métricas de Negocio y Motor de Reglas de Fraude.
Aplica reglas contables y de comportamiento, genera alertas de fraude y perfiles de riesgo por usuario.
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
    GOLD_PROFILES_PATH,
    SILVER_DATA_PATH,
)


class GoldProcessor:
    """
    Motor de Detección de Fraude y Agregaciones Analíticas de la Capa Gold.
    Procesa datasets curados de Silver para emitir alertas y construir perfiles de riesgo.
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
    ):
        self.silver_path = Path(silver_path)
        self.alerts_path = Path(alerts_path)
        self.profiles_path = Path(profiles_path)
        self.compression = compression
        self.large_amount_threshold = large_amount_threshold
        self.burst_threshold = burst_threshold
        self.min_alert_score = min_alert_score

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
        - Regla 1 (Inconsistencia Severa): TRANSFER/CASH_OUT con error de balance en montos elevados.
        - Regla 2 (Transferencia a Cero): TRANSFER de alto monto a cuenta con balance en cero.
        - Regla 3 (Ráfaga / Frecuencia): Múltiples operaciones del mismo usuario en la misma hora (step).
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

        # Calcular Score de Riesgo ponderado (0 - 100)
        lf_scored = lf_with_rules.with_columns([
            (
                pl.col("rule_severe_balance_error").cast(pl.Int32) * 50
                + pl.col("rule_large_zero_dest").cast(pl.Int32) * 35
                + pl.col("rule_burst_frequency").cast(pl.Int32) * 25
            ).alias("risk_score")
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

        # Seleccionar columnas clave de la alerta
        columns_order = [
            "alert_id", "step", "type", "amount", "nameOrig", "nameDest",
            "orig_account_type", "dest_account_type", "rules_triggered",
            "risk_score", "risk_level", "isFraud", "kafka_offset",
            "alert_timestamp", "ingestion_date"
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
        2. Genera y guarda fraud_alerts.
        3. Genera y guarda user_risk_profile.
        """
        print(f"🏆 [GOLD] Iniciando procesamiento (fecha: {partition_date or 'Todas las disponibles'})...")
        print(f"📂 Origen Silver: {self.silver_path}")
        print(f"🚨 Destino Alertas: {self.alerts_path}")
        print(f"👤 Destino Perfiles: {self.profiles_path}")

        try:
            lf_silver = self.scan_silver_data(partition_date)
        except FileNotFoundError as e:
            print(f"⚠️ {e}", file=sys.stderr)
            return (0, 0)

        # Evaluar reglas vectorizadamente
        lf_evaluated = self.evaluate_rules(lf_silver)
        df_evaluated = lf_evaluated.collect()

        total_transactions = len(df_evaluated)
        if total_transactions == 0:
            print("⚠️ [GOLD] No se encontraron registros en Silver para procesar.")
            return (0, 0)

        # 1. Alertas de Fraude
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

        # 2. Perfiles de Riesgo de Usuario
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
    parser = argparse.ArgumentParser(description="Procesador de la Capa Gold (Reglas de Fraude y Perfiles)")
    parser.add_argument("--date", type=str, default=None, help="Fecha de ingesta a procesar (YYYY-MM-DD)")
    parser.add_argument("--silver-path", type=str, default=str(SILVER_DATA_PATH), help="Ruta de archivos Silver")
    parser.add_argument("--alerts-path", type=str, default=str(GOLD_ALERTS_PATH), help="Ruta de alertas Gold")
    parser.add_argument("--profiles-path", type=str, default=str(GOLD_PROFILES_PATH), help="Ruta de perfiles Gold")
    args = parser.parse_args()

    processor = GoldProcessor(
        silver_path=Path(args.silver_path),
        alerts_path=Path(args.alerts_path),
        profiles_path=Path(args.profiles_path),
    )
    processor.process(partition_date=args.date)


if __name__ == "__main__":
    main()
