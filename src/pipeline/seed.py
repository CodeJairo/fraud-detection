"""
Módulo de Inicialización y Generación de Datos Sintéticos (Seed Pipeline).
Permite poblar y ejecutar en vivo el flujo Medallion completo (Bronze -> Silver -> Gold)
de forma reproducible en entornos sin persistencia previa (ej. Railway o demos en la nube),
respetando la separación estricta entre código y datos.
"""

import argparse
from datetime import datetime, timezone
import random
from typing import Dict, List, Optional
import uuid

import polars as pl

from src.config.settings import (
    BRONZE_DATA_PATH,
    GOLD_ALERTS_PATH,
    GOLD_PROFILES_PATH,
    MODEL_PATH,
    SILVER_DATA_PATH,
)
from src.pipeline.gold import GoldProcessor
from src.pipeline.silver import SilverProcessor


def generate_synthetic_transactions(n_transactions: int = 1000, seed: int = 42) -> pl.DataFrame:
    """
    Genera un conjunto de transacciones representativo con perfiles legítimos y anómalos.
    Incluye cuentas ancla para las demostraciones forenses y de KYC en el dashboard.
    """
    random.seed(seed)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_utc = datetime.now(timezone.utc)
    now_ts = int(now_utc.timestamp() * 1000)

    rows: List[dict] = []

    # 1. Cuentas ancla para demostración rápida en 1 clic
    # Cuenta A (Caso Crítico): C1057507014 (Vaciado ilícito con discrepancia contable masiva)
    for i in range(3):
        rows.append({
            "step": 1,
            "type": "TRANSFER",
            "amount": 320000.0 + (i * 15000),
            "nameOrig": "C1057507014",
            "oldbalanceOrg": 350000.0,
            "newbalanceOrig": 0.0,  # Discrepancia contable severa
            "nameDest": f"M_CRITICAL_DEST_{i}",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": 1,
            "ingested_at": now_utc.isoformat(),
            "ingestion_date": today_str,
            "kafka_partition": 0,
            "kafka_offset": 1000 + i,
            "kafka_timestamp": now_ts,
        })

    # Cuenta B (Riesgo Medio): C684230144 (Transferencia a cuenta mula sin saldo previo ni posterior)
    for i in range(2):
        rows.append({
            "step": 1,
            "type": "TRANSFER",
            "amount": 220000.0 + (i * 5000),
            "nameOrig": "C684230144",
            "oldbalanceOrg": 250000.0,
            "newbalanceOrig": 25000.0,
            "nameDest": f"C_MULE_DEST_{i}",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
            "ingested_at": now_utc.isoformat(),
            "ingestion_date": today_str,
            "kafka_partition": 1,
            "kafka_offset": 2000 + i,
            "kafka_timestamp": now_ts,
        })

    # Cuenta C (Cliente Normal): C1231006815 (Pagos cotidianos legítimos sin errores)
    for i in range(5):
        amt = round(random.uniform(25.0, 180.0), 2)
        old_bal = round(1500.0 - (i * 200), 2)
        rows.append({
            "step": random.randint(1, 3),
            "type": "PAYMENT",
            "amount": amt,
            "nameOrig": "C1231006815",
            "oldbalanceOrg": old_bal,
            "newbalanceOrig": round(old_bal - amt, 2),
            "nameDest": f"M_MERCHANT_REGULAR_{i}",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
            "ingested_at": now_utc.isoformat(),
            "ingestion_date": today_str,
            "kafka_partition": 2,
            "kafka_offset": 3000 + i,
            "kafka_timestamp": now_ts,
        })

    # 2. Generación aleatoria para volumen estadístico restante
    remaining = max(10, n_transactions - len(rows))
    types_pool = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    weights_pool = [0.45, 0.20, 0.20, 0.10, 0.05]

    for idx in range(remaining):
        op_type = random.choices(types_pool, weights=weights_pool)[0]
        step_val = random.randint(1, 5)
        offset_val = 4000 + idx
        user_id = f"C{random.randint(100000000, 999999999)}"
        dest_prefix = "M" if op_type == "PAYMENT" else "C"
        dest_id = f"{dest_prefix}{random.randint(100000000, 999999999)}"

        if op_type in ["TRANSFER", "CASH_OUT"]:
            # 15% de probabilidad de generar comportamiento anómalo de fraude
            is_anomaly = random.random() < 0.15
            if is_anomaly:
                amount = round(random.uniform(200000.0, 900000.0), 2)
                old_orig = amount
                new_orig = 0.0 if random.random() < 0.5 else round(amount * 0.1, 2)
                old_dest = 0.0
                new_dest = 0.0
                is_fraud = 1
            else:
                amount = round(random.uniform(500.0, 80000.0), 2)
                old_orig = round(amount + random.uniform(1000.0, 50000.0), 2)
                new_orig = round(old_orig - amount, 2)
                old_dest = round(random.uniform(0.0, 30000.0), 2)
                new_dest = round(old_dest + amount, 2)
                is_fraud = 0
        elif op_type == "CASH_IN":
            amount = round(random.uniform(100.0, 25000.0), 2)
            old_orig = round(random.uniform(50.0, 10000.0), 2)
            new_orig = round(old_orig + amount, 2)
            old_dest = 0.0
            new_dest = 0.0
            is_fraud = 0
        else:  # PAYMENT / DEBIT
            amount = round(random.uniform(10.0, 1500.0), 2)
            old_orig = round(amount + random.uniform(50.0, 5000.0), 2)
            new_orig = round(old_orig - amount, 2)
            old_dest = 0.0
            new_dest = 0.0
            is_fraud = 0

        rows.append({
            "step": step_val,
            "type": op_type,
            "amount": amount,
            "nameOrig": user_id,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": dest_id,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": 0,
            "ingested_at": now_utc.isoformat(),
            "ingestion_date": today_str,
            "kafka_partition": idx % 3,
            "kafka_offset": offset_val,
            "kafka_timestamp": now_ts,
        })

    return pl.DataFrame(rows)


def run_pipeline_seed(n_transactions: int = 1000, seed: int = 42) -> Dict[str, any]:
    """
    Ejecuta el ciclo de vida completo de generación e ingesta Medallion:
    1. Genera y persiste micro-lote en Bronze.
    2. Ejecuta limpieza y balance features en Silver con Polars.
    3. Ejecuta motor de reglas e inferencia LightGBM en Gold.
    """
    start_time = datetime.now(timezone.utc)
    print(f"🌱 [SEED] Iniciando generación de {n_transactions} transacciones sintéticas...")

    # 1. Crear Bronze Parquet
    df_bronze = generate_synthetic_transactions(n_transactions, seed=seed)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bronze_dir = BRONZE_DATA_PATH / f"ingestion_date={today_str}"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    batch_uuid = uuid.uuid4().hex[:8]
    bronze_file = bronze_dir / f"bronze_batch_seed_{batch_uuid}.parquet"
    df_bronze.write_parquet(bronze_file, compression="snappy")
    print(f"🥉 [BRONZE] Guardado micro-lote semilla con {len(df_bronze)} registros en {bronze_file.name}")

    # 2. Ejecutar Silver con Polars
    silver_proc = SilverProcessor(bronze_path=BRONZE_DATA_PATH, silver_path=SILVER_DATA_PATH)
    silver_count = silver_proc.process(partition_date=today_str)
    print(f"🥈 [SILVER] Procesados {silver_count} registros curados.")

    # 3. Ejecutar Gold con Motor Híbrido (Reglas + LightGBM)
    gold_proc = GoldProcessor(
        silver_path=SILVER_DATA_PATH,
        alerts_path=GOLD_ALERTS_PATH,
        profiles_path=GOLD_PROFILES_PATH,
        model_path=MODEL_PATH if MODEL_PATH.exists() else None,
    )
    alerts_count, profiles_count = gold_proc.process(partition_date=today_str)
    print(f"🥇 [GOLD] Generadas {alerts_count} alertas y {profiles_count} perfiles de usuario.")

    elapsed_sec = (datetime.now(timezone.utc) - start_time).total_seconds()
    print(f"✨ [SEED] Pipeline completado en {elapsed_sec:.2f}s")

    return {
        "status": "success",
        "transactions_generated": len(df_bronze),
        "silver_records": silver_count,
        "fraud_alerts_created": alerts_count,
        "user_profiles_created": profiles_count,
        "execution_time_seconds": round(elapsed_sec, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Seed Pipeline Generator para Medallion Lakehouse")
    parser.add_argument("--transactions", type=int, default=1000, help="Cantidad de transacciones a generar (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla pseudoaleatoria (default: 42)")
    args = parser.parse_args()

    run_pipeline_seed(n_transactions=args.transactions, seed=args.seed)


if __name__ == "__main__":
    main()
