import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env si existe
load_dotenv()

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PAYSIM_CSV_PATH = DATA_DIR / "paysim.csv"

# Configuración de Kafka / Redpanda
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
KAFKA_TOPIC_RAW = os.getenv("KAFKA_TOPIC_RAW", "transactions.raw")

# Configuración del Productor
PRODUCER_CHUNK_SIZE = int(os.getenv("PRODUCER_CHUNK_SIZE", "500"))
PRODUCER_DELAY_SECONDS = float(os.getenv("PRODUCER_DELAY_SECONDS", "0.001"))

# Configuración de la Capa Bronze
KAFKA_CONSUMER_GROUP_BRONZE = os.getenv("KAFKA_CONSUMER_GROUP_BRONZE", "bronze-ingestion-group")
BRONZE_DATA_PATH = Path(os.getenv("BRONZE_DATA_PATH", str(DATA_DIR / "bronze" / "transactions")))
BRONZE_BATCH_SIZE = int(os.getenv("BRONZE_BATCH_SIZE", "2000"))
BRONZE_BATCH_TIMEOUT_SEC = float(os.getenv("BRONZE_BATCH_TIMEOUT_SEC", "5.0"))

# Configuración de la Capa Silver
SILVER_DATA_PATH = Path(os.getenv("SILVER_DATA_PATH", str(DATA_DIR / "silver" / "transactions")))
SILVER_COMPRESSION = os.getenv("SILVER_COMPRESSION", "snappy")

# Configuración de la Capa Gold (Métricas y Alertas de Fraude)
GOLD_ALERTS_PATH = Path(os.getenv("GOLD_ALERTS_PATH", str(DATA_DIR / "gold" / "fraud_alerts")))
GOLD_PROFILES_PATH = Path(os.getenv("GOLD_PROFILES_PATH", str(DATA_DIR / "gold" / "user_risk_profile")))
GOLD_COMPRESSION = os.getenv("GOLD_COMPRESSION", "snappy")

# Umbrales del motor de reglas de fraude
GOLD_LARGE_AMOUNT_THRESHOLD = float(os.getenv("GOLD_LARGE_AMOUNT_THRESHOLD", "200000.0"))
GOLD_BURST_THRESHOLD = int(os.getenv("GOLD_BURST_THRESHOLD", "2"))
GOLD_ALERT_MIN_SCORE = int(os.getenv("GOLD_ALERT_MIN_SCORE", "35"))
