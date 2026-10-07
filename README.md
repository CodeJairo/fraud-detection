# 🛡️ Fintech Fraud Detection Platform

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Redpanda](https://img.shields.io/badge/Streaming-Redpanda%20(Kafka)-FD3B2A?style=for-the-badge&logo=apachekafka&logoColor=white)](https://redpanda.com/)
[![Polars](https://img.shields.io/badge/Data_Engine-Polars-CD792C?style=for-the-badge&logo=polars&logoColor=white)](https://pola.rs/)
[![LightGBM](https://img.shields.io/badge/ML_Engine-LightGBM-2E7D32?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Microservice-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-32%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)

Plataforma integral de **detección de fraude financiero en streaming, procesamiento analítico por lotes y Machine Learning supervisado**, construida siguiendo los estándares de la **Arquitectura Medallion** (*Bronze, Silver, Gold*). 

El sistema ingesta flujos masivos de eventos transaccionales en tiempo real, aplica reconciliación contable de balances, entrena y evalúa clasificadores Gradient Boosting (**LightGBM**), ejecuta inferencia probabilística híbrida con reglas de negocio, y expone tanto un **microservicio REST asíncrono** de alto rendimiento como un **dashboard interactivo** para analistas de riesgo y cumplimiento normativo (*Compliance*).

---

## 🤖 Metodología de Desarrollo y Orquestación de Agentes

> [!NOTE]
> **Desarrollo Asistido por Agentes de Código Autónomos (Google DeepMind Antigravity)**  
> Este proyecto fue concebido, estructurado e implementado aplicando metodologías avanzadas de **Ingeniería Agentica y Pair Programming asistido por IA**.
>
> El desarrollo siguió un ciclo riguroso de control y validación:
> 1. **Planificación Arquitectónica Declarativa (`/plan`)**: Descomposición sistemática de requisitos en artefactos de diseño técnico, contratos de interfaz y diagramas antes de ejecutar cualquier modificación en el código.
> 2. **Desarrollo Guiado por Pruebas (TDD) y Auto-Verificación**: Ejecución de suites unitarias y de integración end-to-end con retroalimentación automática y resolución reactiva de errores en tiempo de ejecución.
> 3. **Commits Modulares Atómicos**: Seguimiento estricto del estándar *Conventional Commits* en etapas delimitadas (`infra`, `build`, `feat(config)`, `feat(producer)`, `feat(bronze)`, `feat(silver)`, `feat(gold)`, `feat(api)`, `feat(ui)`, `feat(ml)`).
> 4. **Resiliencia de Sistemas**: Diagnóstico y corrección autónoma de problemas de infraestructura (parámetros de Schema Registry en Redpanda, optimización de memoria en Polars, manejo de clases desbalanceadas en ML y control de concurrencia).

---

## 🏛️ Arquitectura General del Sistema

El flujo de datos sigue un ciclo desacoplado de eventos, computación vectorizada y modelos probabilísticos:

```mermaid
flowchart TD
    subgraph Data_Source ["Fuente de Datos"]
        CSV["Dataset PaySim\n(6M+ transacciones)"]
    end

    subgraph Streaming_Ingestion ["Ingesta en Streaming"]
        PROD["Productor Asíncrono\n(src/producer/producer.py)"]
        BROKER["Redpanda Broker (Kafka API)\nTopic: transactions.raw (3 Particiones)"]
        CSV --> PROD
        PROD -->|"Particionado por key=nameOrig"| BROKER
    end

    subgraph Medallion_Architecture ["Pipeline de Datos Medallion"]
        BRONZE["🥉 Capa Bronze (Raw Ingestion)\n- Micro-batching (tamaño/timeout)\n- Metadatos de auditoría (UTC, offset)\n- Garantía At-Least-Once\n- Parquet Snappy particionado"]
        SILVER["🥈 Capa Silver (Curated & Enriched)\n- Polars Lazy Scan (Out-of-core)\n- Deduplicación por clave de negocio\n- Extracción de tipo de cuenta (Client/Merchant)\n- Feature Engineering de balances"]
        
        BROKER --> BRONZE
        BRONZE -->|"data/bronze/transactions/"| SILVER
    end

    subgraph ML_Layer ["Machine Learning Supervisado"]
        TRAIN["🧠 Entrenamiento (src/ml/train.py)\n- Split Estratificado 80/20\n- Pipeline Scikit-Learn + OneHotEncoder\n- LightGBM Classifier (class_weight='balanced')\n- Evaluación PR-AUC & ROC-AUC"]
        MODEL["models/fraud_model.joblib\nmodels/model_metrics.json"]
        SILVER --> TRAIN --> MODEL
    end

    subgraph Gold_Layer ["🥇 Capa Gold (Hybrid Risk Engine)"]
        GOLD["GoldProcessor (src/pipeline/gold.py)\n- Motor de 3 Reglas Contables (0 - 100 pts)\n- Inferencia Batch de ML (0.0 - 1.0)\n- Score Híbrido: 50% Reglas + 50% ML\n- Tablas: fraud_alerts & user_risk_profile"]
        SILVER --> GOLD
        MODEL -.->|"Inferencia Probabilística"| GOLD
    end

    subgraph Serving_Layer ["Capa de Consumo y Exposición"]
        API["🚀 Microservicio FastAPI (Puerto 8000)\n- GET /health (Inventario Medallion & Estado ML)\n- GET /alerts (Filtros por riesgo, regla y min_ml_prob)\n- GET /users/{id}/risk-profile"]
        UI["📊 Dashboard Streamlit (Puerto 8501)\n- KPIs Ejecutivos & Estado de Modelo ML\n- Matriz Híbrida: Reglas vs. Probabilidad ML\n- Auditoría de Alertas & Exportación CSV\n- Ficha Técnica de Riesgo del Cliente"]

        GOLD -->|"data/gold/fraud_alerts/\ndata/gold/user_risk_profile/"| API
        API <-->|"Peticiones HTTP (requests)"| UI
    end
```

---

## ⚙️ Especificación Técnica de las Capas

### 🥉 1. Capa Bronze: Ingesta Cruda en Streaming
* **Objetivo**: Capturar el flujo continuo de transacciones desde Redpanda sin alterar los valores de origen, agregando trazabilidad de auditoría.
* **Componentes**:
  * **Productor ([src/producer/producer.py](file:///home/codejairo/Proyectos/fintech-fraud-detection/src/producer/producer.py))**: Lee el dataset de PaySim por chunks y serializa en JSON. Utiliza `nameOrig` como clave de partición, asegurando que las transacciones del mismo usuario aterricen en la misma partición y mantengan orden secuencial estricto.
  * **Consumidor Bronze ([src/pipeline/bronze.py](file:///home/codejairo/Proyectos/fintech-fraud-detection/src/pipeline/bronze.py))**: Consumo por micro-lotes (basado en tamaño `batch_size=500` o tiempo `batch_timeout_sec=5.0s`).
  * **Garantía *At-Least-Once***: `enable.auto.commit = False`. El commit manual síncrono del offset a Kafka se ejecuta **exclusivamente después** de que el archivo Parquet ha sido persistido y cerrado en disco.
  * **Metadatos de auditoría**: Inyección de `ingested_at` (UTC ISO 8601), `ingestion_date`, `kafka_partition`, `kafka_offset` y `kafka_timestamp`.
  * **Almacenamiento**: `data/bronze/transactions/ingestion_date=YYYY-MM-DD/bronze_batch_*.parquet`.

---

### 🥈 2. Capa Silver: Limpieza, Normalización y Feature Engineering
* **Objetivo**: Convertir los datos crudos en un dataset estructurado, libre de duplicados y enriquecido con variables contables.
* **Motor Analítico**: Implementado en **Polars** (`polars>=2.0.0`) aprovechando ejecución vectorizada en Rust sin requerir inicialización de la JVM de Java.
* **Transformaciones ([src/pipeline/silver.py](file:///home/codejairo/Proyectos/fintech-fraud-detection/src/pipeline/silver.py))**:
  * **Deduplicación de Negocio**: Eliminación de duplicados mediante la clave compuesta:
    $$\text{Clave} = [\text{nameOrig}, \text{nameDest}, \text{amount}, \text{step}, \text{kafka\_offset}]$$
  * **Normalización de Cuentas**: Extracción categórica a partir de prefijos (`C` $\rightarrow$ `CLIENT`, `M` $\rightarrow$ `MERCHANT`).
  * **Ingeniería de Características de Balances (Ecuaciones Contables)**:
    $$\text{balance\_error\_orig} = (\text{oldbalanceOrg} - \text{amount}) - \text{newbalanceOrig}$$
    $$\text{balance\_error\_dest} = (\text{oldbalanceDest} + \text{amount}) - \text{newbalanceDest}$$
    *(En transferencias legítimas el error es 0. En transacciones fraudulentas de vaciado, revela discrepancias masivas inmediatas).*
  * **Almacenamiento**: `data/silver/transactions/ingestion_date=YYYY-MM-DD/silver_clean_*.parquet`.

---

### 🧠 3. Módulo de Machine Learning (LightGBM)
* **Objetivo**: Entrenar un modelo de clasificación supervisado de Gradient Boosting sobre los datos limpios de Silver para capturar relaciones no lineales y patrones ocultos de fraude.
* **Pipeline de Entrenamiento ([src/ml/train.py](file:///home/codejairo/Proyectos/fintech-fraud-detection/src/ml/train.py))**:
  * **Features Numéricas**: `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `balance_error_orig`, `balance_error_dest`.
  * **Features Categóricas**: `type`, `orig_account_type`, `dest_account_type` procesadas con `OneHotEncoder(handle_unknown='ignore')`.
  * **Manejo de Desbalance**: Configuración de `class_weight='balanced'` en `LGBMClassifier` para contrarrestar la baja tasa de positivos ($<1\%$).
  * **Validación**: División estratificada 80/20 evaluando **ROC-AUC** y **PR-AUC (Precision-Recall AUC)**.
  * **Persistencia**: Serialización atómica a `models/fraud_model.joblib` y métricas en `models/model_metrics.json`.

---

### 🥇 4. Capa Gold: Motor Híbrido de Reglas y Scoring
* **Objetivo**: Combinar la certeza de las reglas deterministas con la probabilidad del clasificador de Machine Learning para emitir alertas y construir perfiles analíticos.
* **Reglas de Negocio ([src/pipeline/gold.py](file:///home/codejairo/Proyectos/fintech-fraud-detection/src/pipeline/gold.py))**:
  * **Regla 1 (Inconsistencia Severa de Saldo - 50 pts)**: $\text{balance\_error\_orig} \ne 0$ en transacciones de tipo `TRANSFER` o `CASH_OUT` con montos $\ge \$200,000$.
  * **Regla 2 (Transferencia a Destino con Saldo Cero - 35 pts)**: Operación `TRANSFER` $\ge \$200,000$ hacia cuentas sin balance previo ni posterior (mulas financieras).
  * **Regla 3 (Ráfaga Operativa en la Misma Hora - 25 pts)**: Cuenta de origen con $\ge 2$ transacciones en el mismo paso temporal (`step`).
* **Score Híbrido Balanceado (50% Reglas + 50% ML)**:
  $$\text{risk\_score} = \text{round}\left(0.5 \times \text{rules\_score} + 0.5 \times (\text{ml\_probability} \times 100)\right)$$
  *(Si el modelo ML no está disponible, el procesador aplica fallback automático utilizando exclusivamente `rules_score`).*
* **Nivel de Severidad**:
  * `HIGH`: $\text{risk\_score} \ge 60$
  * `MEDIUM`: $35 \le \text{risk\_score} < 60$
  * `LOW`: $0 < \text{risk\_score} < 35$ (Informativo)
* **Datasets Resultantes**:
  * **`fraud_alerts`**: Registro de alertas con score $\ge 35$, UUID único, `ml_probability`, `rules_score`, marcas de tiempo y lista de reglas violadas (`rules_triggered`).
  * **`user_risk_profile`**: Perfil analítico agrupado por cliente (`nameOrig`) calculando volumen total acumulado, promedio de errores contables, desglose por tipo de operación, tasa de alertas emitidas y nivel global (`LOW`, `MEDIUM`, `HIGH`).

---

## 🚀 5. Capa de Servicio: API REST y Dashboard

### Microservicio FastAPI ([src/api/main.py](file:///home/codejairo/Proyectos/fintech-fraud-detection/src/api/main.py))
Servidor asíncrono con Uvicorn y validación de tipos con Pydantic:

| Método | Endpoint | Descripción | Parámetros Query |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Estado del servicio, conteo de archivos por capa y estado del modelo ML | Ninguno |
| `GET` | `/alerts` | Consulta de alertas de fraude con score híbrido y probabilidad de ML | `risk_level`, `rule`, `min_ml_prob`, `limit`, `offset` |
| `GET` | `/users/{name_orig}/risk-profile` | Consulta directa del perfil de riesgo acumulado de un cliente | `name_orig` (Path) |

* **Documentación Interactiva Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Dashboard Streamlit ([src/ui/app.py](file:///home/codejairo/Proyectos/fintech-fraud-detection/src/ui/app.py))
Panel web interactivo con Plotly para analistas de riesgo:
1. **📊 Resumen General**: Monitoreo de salud del Medallion, indicador del estado del modelo de Machine Learning (`LightGBM Activo`), gráficos de distribución y **Matriz Híbrida de Decisión** (gráfico de dispersión interactivo: *Score de Reglas vs. Probabilidad ML*).
2. **🚨 Monitor de Alertas**: Filtrado dinámico por nivel de riesgo, término de regla y control deslizante de *Probabilidad Mínima ML (%)*, tabla interactiva con porcentaje formateado y exportación directa en **CSV**.
3. **👤 Perfil de Cliente**: Búsqueda por ID de cuenta (`nameOrig`), tarjeta de KPIs con badges visuales de riesgo y visor JSON expandible.

---

## 🧪 Pruebas Automatizadas

El proyecto cuenta con una suite integral de **32 pruebas automatizadas** que cubren el ciclo completo (ingesta, transformaciones, pipeline ML, inferencia Gold, endpoints REST y UI):

```bash
uv run pytest tests/ -v
```

```text
tests/test_api.py::test_api_health PASSED                                [  3%]
tests/test_api.py::test_api_alerts_unfiltered PASSED                     [  6%]
tests/test_api.py::test_api_alerts_filtered_by_risk_level PASSED         [  9%]
tests/test_api.py::test_api_alerts_filtered_by_rule PASSED               [ 12%]
tests/test_api.py::test_api_alerts_filtered_by_ml_probability PASSED     [ 15%]
tests/test_api.py::test_api_user_risk_profile_existing PASSED            [ 18%]
tests/test_api.py::test_api_user_risk_profile_not_found PASSED           [ 21%]
tests/test_bronze.py::test_transaction_event_schema_validation PASSED    [ 25%]
tests/test_bronze.py::test_bronze_record_enrichment PASSED               [ 28%]
tests/test_bronze.py::test_bronze_flush_batch_to_parquet PASSED          [ 31%]
tests/test_gold.py::test_gold_rule_1_balance_inconsistency PASSED        [ 34%]
tests/test_gold.py::test_gold_rule_2_large_transfer_zero_dest PASSED     [ 37%]
tests/test_gold.py::test_gold_rule_3_burst_frequency PASSED              [ 40%]
tests/test_gold.py::test_gold_combined_score_and_alert_extraction PASSED [ 43%]
tests/test_gold.py::test_gold_user_risk_profile_aggregation PASSED       [ 46%]
tests/test_gold.py::test_gold_pipeline_end_to_end PASSED                 [ 50%]
tests/test_gold.py::test_gold_fallback_without_model PASSED              [ 53%]
tests/test_gold.py::test_gold_with_ml_model_scoring PASSED               [ 56%]
tests/test_ml.py::test_build_pipeline_structure PASSED                   [ 59%]
tests/test_ml.py::test_train_model_metrics PASSED                        [ 62%]
tests/test_ml.py::test_save_and_load_artifacts PASSED                    [ 65%]
tests/test_ml.py::test_inference_probability_bounds PASSED               [ 68%]
tests/test_ml.py::test_handling_unseen_categorical_levels PASSED         [ 71%]
tests/test_silver.py::test_account_type_extraction PASSED                [ 75%]
tests/test_silver.py::test_balance_error_calculations PASSED             [ 78%]
tests/test_silver.py::test_deduplication PASSED                          [ 81%]
tests/test_silver.py::test_clean_and_filter_invalid_rows PASSED          [ 84%]
tests/test_silver.py::test_silver_pipeline_end_to_end PASSED             [ 87%]
tests/test_ui.py::test_fetch_data_success PASSED                         [ 90%]
tests/test_ui.py::test_fetch_data_404_error PASSED                       [ 93%]
tests/test_ui.py::test_fetch_data_connection_error PASSED                [ 96%]
tests/test_ui.py::test_app_module_imports PASSED                         [100%]

======================== 32 passed in 3.26s =========================
```

---

## 🛠️ Guía de Puesta en Marcha (Quickstart)

### 1. Requisitos Previos
* **Python 3.14+**
* [uv](https://docs.astral.sh/uv/) instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
* **Docker & Docker Compose**

### 2. Clonación e Instalación
```bash
git clone https://github.com/tu-usuario/fintech-fraud-detection.git
cd fintech-fraud-detection

# Sincronizar el entorno virtual con uv
uv sync
```

### 3. Iniciar Servicios de Infraestructura
```bash
# Levantar Redpanda Broker y Redpanda Console
docker compose up -d

# Crear el topic con 3 particiones
docker exec redpanda rpk topic create transactions.raw --partitions 3
```
* Acceso a **Redpanda Console**: [http://localhost:8080](http://localhost:8080)

### 4. Ejecución del Pipeline Medallion & Machine Learning
```bash
# 1. Transmitir transacciones al topic de Redpanda
uv run python -m src.producer.producer --limit 2000

# 2. Ingestar micro-lotes en la Capa Bronze (Parquet)
uv run python -m src.pipeline.bronze --max-messages 2000

# 3. Limpiar y enriquecer en la Capa Silver (Polars)
uv run python -m src.pipeline.silver

# 4. Entrenar el clasificador de Machine Learning (LightGBM)
uv run python -m src.ml.train

# 5. Evaluar reglas y ejecutar inferencia híbrida en la Capa Gold
uv run python -m src.pipeline.gold
```

### 5. Iniciar la Capa de Exposición (API & Dashboard)

* **Terminal 1: Backend (FastAPI)**
  ```bash
  uv run uvicorn src.api.main:app --port 8000 --reload
  ```
  * Swagger UI disponible en: [http://localhost:8000/docs](http://localhost:8000/docs)

* **Terminal 2: Frontend (Streamlit)**
  ```bash
  uv run streamlit run src/ui/app.py
  ```
  * Dashboard disponible en: [http://localhost:8501](http://localhost:8501)

---

## 📁 Estructura del Repositorio

```text
fintech-fraud-detection/
├── docker-compose.yml       # Definición de servicios Redpanda & Redpanda Console
├── pyproject.toml           # Dependencias del proyecto y configuración de pytest
├── uv.lock                  # Lockfile determinista de dependencias
├── README.md                # Documentación técnica y arquitectura del sistema
├── models/                  # Artefactos serializados de ML (ignorado por Git si se desea)
│   ├── fraud_model.joblib   # Pipeline Scikit-Learn + LightGBM Classifier
│   └── model_metrics.json   # Métricas de entrenamiento (ROC-AUC, PR-AUC, Confusion Matrix)
├── data/                    # Almacenamiento local de datos (ignorado por Git)
│   ├── paysim.csv           # Dataset de simulación transaccional
│   ├── bronze/transactions/ # Parquet crudos particionados con metadatos
│   ├── silver/transactions/ # Parquet curados con features de balances
│   └── gold/                # Datasets analíticos finales
│       ├── fraud_alerts/    # Alertas híbridas (Reglas + Inferencia ML)
│       └── user_risk_profile/ # Perfiles acumulados de clientes
├── src/
│   ├── api/                 # Microservicio REST
│   │   ├── __init__.py
│   │   └── main.py          # Endpoints FastAPI (/health, /alerts, /users)
│   ├── config/              # Configuración y esquemas
│   │   ├── __init__.py
│   │   ├── schemas.py       # Modelos Pydantic v2 (Bronze, Silver, Gold, API)
│   │   └── settings.py      # Variables de entorno y umbrales configurables
│   ├── ml/                  # Módulo de Machine Learning
│   │   ├── __init__.py      # Utilidad de carga de modelo (load_fraud_model)
│   │   └── train.py         # Pipeline de entrenamiento LightGBM y métricas
│   ├── pipeline/            # Procesamiento de datos Medallion
│   │   ├── __init__.py      # Exportaciones perezosas (PEP 562)
│   │   ├── bronze.py        # Consumidor Redpanda a Parquet (at-least-once)
│   │   ├── silver.py        # Limpieza, deduplicación y features con Polars
│   │   └── gold.py          # Motor híbrido de reglas, inferencia ML y scoring
│   ├── producer/            # Productor de streaming
│   │   ├── __init__.py
│   │   └── producer.py      # Productor Kafka particionado por cuenta origen
│   └── ui/                  # Interfaz gráfica
│       ├── __init__.py
│       └── app.py           # Dashboard interactivo Streamlit + Plotly
└── tests/                   # Suite de pruebas automatizadas
    ├── __init__.py
    ├── test_api.py          # Tests de endpoints FastAPI (TestClient)
    ├── test_bronze.py       # Tests de ingesta y serialización Bronze
    ├── test_gold.py         # Tests de reglas contables, inferencia y perfiles Gold
    ├── test_ml.py           # Tests de entrenamiento, persistencia e inferencia ML
    ├── test_silver.py       # Tests de deduplicación y balance features Silver
    └── test_ui.py           # Tests de lógica de consumo del Dashboard
```

---

## 📜 Licencia y Autoría

Desarrollado como proyecto de referencia de ingeniería de datos y detección de fraude en fintech.  
**Autor**: CodeJairo (<jairoehr@gmail.com>)
