"""
Pipeline de la Capa Silver: Limpieza, normalización y feature engineering.
Transforma archivos Parquet crudos de Bronze en datasets depurados listos para modelado.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Optional
import uuid

import polars as pl

from src.config.settings import (
    BRONZE_DATA_PATH,
    SILVER_COMPRESSION,
    SILVER_DATA_PATH,
)


class SilverProcessor:
    """
    Procesador de la Capa Silver implementado con Polars para alta eficiencia y bajo consumo de memoria.
    Realiza deduplicación por clave de negocio, normalización y cálculo de discrepancias contables.
    """

    def __init__(
        self,
        bronze_path: Path = BRONZE_DATA_PATH,
        silver_path: Path = SILVER_DATA_PATH,
        compression: str = SILVER_COMPRESSION,
    ):
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.compression = compression

    def scan_bronze_data(self, partition_date: Optional[str] = None) -> pl.LazyFrame:
        """
        Escanea archivos Parquet de la Capa Bronze como un Polars LazyFrame.
        Permite filtrar por una fecha de ingesta específica o procesar todo el histórico.
        """
        if partition_date:
            target_pattern = self.bronze_path / f"ingestion_date={partition_date}" / "*.parquet"
        else:
            target_pattern = self.bronze_path / "**" / "*.parquet"

        files = list(self.bronze_path.glob(f"ingestion_date={partition_date}/*.parquet" if partition_date else "**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos Parquet en Bronze para el patrón: {target_pattern}")

        # pl.scan_parquet procesa de forma perezosa y no carga todo a RAM de inmediato
        return pl.scan_parquet(str(target_pattern))

    def deduplicate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Elimina eventos duplicados basándose en la clave compuesta de negocio y origen.
        """
        dedup_keys = ["nameOrig", "nameDest", "amount", "step", "kafka_offset"]
        return lf.unique(subset=dedup_keys, keep="first")

    def clean_and_normalize(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Filtra registros anómalos, extrae tipos de cuenta (Cliente/Comerciante)
        y asegura el casteo estricto de tipos de datos.
        """
        # Filtrar montos no positivos o cuentas nulas
        lf_filtered = lf.filter(
            (pl.col("amount") > 0)
            & pl.col("nameOrig").is_not_null()
            & pl.col("nameDest").is_not_null()
        )

        # Extracción categórica de tipos de cuenta a partir del prefijo (C = Client, M = Merchant)
        lf_normalized = lf_filtered.with_columns([
            pl.when(pl.col("nameOrig").str.starts_with("C")).then(pl.lit("CLIENT"))
              .when(pl.col("nameOrig").str.starts_with("M")).then(pl.lit("MERCHANT"))
              .otherwise(pl.lit("UNKNOWN")).alias("orig_account_type"),

            pl.when(pl.col("nameDest").str.starts_with("C")).then(pl.lit("CLIENT"))
              .when(pl.col("nameDest").str.starts_with("M")).then(pl.lit("MERCHANT"))
              .otherwise(pl.lit("UNKNOWN")).alias("dest_account_type"),

            # Asegurar tipado numérico estricto
            pl.col("step").cast(pl.Int64),
            pl.col("amount").cast(pl.Float64),
            pl.col("oldbalanceOrg").cast(pl.Float64),
            pl.col("newbalanceOrig").cast(pl.Float64),
            pl.col("oldbalanceDest").cast(pl.Float64),
            pl.col("newbalanceDest").cast(pl.Float64),
            pl.col("isFraud").cast(pl.Int8),
            pl.col("isFlaggedFraud").cast(pl.Int8),
        ])
        return lf_normalized

    def add_balance_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Ingeniería de características de balance contable:
        - balance_error_orig: Discrepancia en cuenta emisora (debería ser 0 en transacciones sin anomalías).
        - balance_error_dest: Discrepancia en cuenta receptora (debería ser 0 en abonos directos).
        """
        return lf.with_columns([
            ((pl.col("oldbalanceOrg") - pl.col("amount")) - pl.col("newbalanceOrig"))
            .round(4)
            .alias("balance_error_orig"),

            ((pl.col("oldbalanceDest") + pl.col("amount")) - pl.col("newbalanceDest"))
            .round(4)
            .alias("balance_error_dest"),
        ])

    def enrich_audit(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Agrega metadatos de auditoría propios de la transformación Silver."""
        now_utc = datetime.now(timezone.utc).isoformat()
        return lf.with_columns(pl.lit(now_utc).alias("silver_processed_at"))

    def process(self, partition_date: Optional[str] = None) -> int:
        """
        Ejecuta el pipeline completo de la Capa Silver:
        Lectura -> Deduplicación -> Limpieza -> Feature Engineering -> Auditoría -> Escritura Parquet.
        """
        print(f"🔄 [SILVER] Iniciando procesamiento (fecha: {partition_date or 'Todas las disponibles'})...")
        print(f"📂 Origen Bronze: {self.bronze_path}")
        print(f"📂 Destino Silver: {self.silver_path}")

        try:
            lf_bronze = self.scan_bronze_data(partition_date)
        except FileNotFoundError as e:
            print(f"⚠️ {e}", file=sys.stderr)
            return 0

        # Encadenar transformaciones perezosas
        lf_pipeline = (
            lf_bronze
            .pipe(self.deduplicate)
            .pipe(self.clean_and_normalize)
            .pipe(self.add_balance_features)
            .pipe(self.enrich_audit)
        )

        # Materializar dataset procesado
        df_silver = lf_pipeline.collect()
        total_rows = len(df_silver)

        if total_rows == 0:
            print("⚠️ [SILVER] El pipeline no produjo registros después de filtros y deduplicación.")
            return 0

        # Escribir dataset particionado por fecha de ingesta original
        unique_dates = df_silver["ingestion_date"].unique().to_list()
        for date_val in unique_dates:
            date_str = str(date_val)
            partition_dir = self.silver_path / f"ingestion_date={date_str}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            df_partition = df_silver.filter(pl.col("ingestion_date") == date_val)
            batch_id = uuid.uuid4().hex[:8]
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = partition_dir / f"silver_clean_{timestamp_str}_{batch_id}.parquet"

            df_partition.write_parquet(output_file, compression=self.compression)
            print(f"💾 [SILVER] Partición '{date_str}': {len(df_partition)} registros escritos en {output_file.name}")

        print(f"✨ [SILVER] Proceso completado exitosamente. Total de registros limpios: {total_rows}")
        return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesador de la Capa Silver (Limpieza y Enriquecimiento)")
    parser.add_argument("--date", type=str, default=None, help="Fecha de ingesta a procesar (formato YYYY-MM-DD)")
    parser.add_argument("--bronze-path", type=str, default=str(BRONZE_DATA_PATH), help="Ruta de archivos Bronze")
    parser.add_argument("--silver-path", type=str, default=str(SILVER_DATA_PATH), help="Ruta de destino Silver")
    args = parser.parse_args()

    processor = SilverProcessor(
        bronze_path=Path(args.bronze_path),
        silver_path=Path(args.silver_path),
    )
    processor.process(partition_date=args.date)


if __name__ == "__main__":
    main()
