"""
Centro de Comando y Monitoreo de Riesgo de Fraude Fintech.
Dashboard interactivo construido con Streamlit y Plotly sobre la Arquitectura Medallion,
integrando reglas contables de negocio y modelos de Machine Learning (LightGBM).
"""

from datetime import datetime
import os
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Configuración de página y metadatos
st.set_page_config(
    page_title="Fintech Fraud Detection - Risk Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


def fetch_data(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Realiza peticiones HTTP a la API REST con manejo controlado de errores."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=json_data, timeout=5.0)
        else:
            response = requests.get(url, params=params, timeout=5.0)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"error": response.json().get("detail", "Recurso no encontrado"), "status_code": 404}
        else:
            return {"error": f"Error {response.status_code}: {response.text}", "status_code": response.status_code}
    except requests.exceptions.ConnectionError:
        return {"connection_error": True}
    except Exception as e:
        return {"error": str(e)}


def show_connection_error():
    """Muestra un banner informativo si el microservicio FastAPI no está disponible."""
    st.error(
        f"No se pudo conectar con el microservicio API en `{API_BASE_URL}`.\n\n"
        "Asegúrate de iniciar el backend con el siguiente comando:\n\n"
        "```bash\n"
        "uv run uvicorn src.api.main:app --port 8000 --reload\n"
        "```"
    )


# ==============================================================================
# BARRA LATERAL (SIDEBAR NAVIGATION)
# ==============================================================================
st.sidebar.title("🛡️ Fraud Engine")
st.sidebar.caption("Pipeline Medallion | Streaming | Machine Learning")
st.sidebar.markdown("---")

view_option = st.sidebar.radio(
    "Navegación del Centro de Riesgo:",
    [
        "🏛️ Arquitectura y Resumen Ejecutivo",
        "🚨 Monitor Forense de Alertas",
        "👤 Auditoría de Cliente (KYC)",
        "🧪 Simulador de Transacciones (Sandbox)",
    ],
)

st.sidebar.markdown("---")

# Verificación de estado de la API en la barra lateral
health_check = fetch_data("/health")
if isinstance(health_check, dict) and not health_check.get("connection_error") and health_check.get("status") == "healthy":
    st.sidebar.success("🟢 API Backend: En línea (200 OK)")
    ml_status = health_check.get("medallion_status", {}).get("ml_model_active", False)
    st.sidebar.info(f"🤖 Motor ML: {'LightGBM Activo' if ml_status else 'Solo Reglas'}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Simulador de Pipeline en Vivo:**")
    if st.sidebar.button("⚡ Inyectar Lote (1,000 eventos)", use_container_width=True, help="Ejecuta en vivo el pipeline Medallion: genera eventos, procesa con Polars y evalúa con LightGBM"):
        with st.spinner("Procesando pipeline Medallion en vivo..."):
            res = fetch_data("/pipeline/simulate", method="POST", params={"count": 1000})
            if res and res.get("status") == "success":
                st.sidebar.success(f"¡Lote completado en {res.get('execution_time_seconds', 0)}s!")
                st.rerun()
            else:
                st.sidebar.error("Error al procesar el lote en vivo.")
else:
    st.sidebar.warning("🔴 API Backend: Fuera de línea")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Componentes del Stack:**
    - ⚡ Ingesta: Redpanda (Kafka API)
    - ⚡ Procesamiento: Polars (Rust Engine)
    - 🤖 Machine Learning: LightGBM
    - 🚀 Microservicio: FastAPI & Pydantic v2
    - 📊 Interfaz: Streamlit & Plotly
    """
)

# ==============================================================================
# ENCABEZADO PRINCIPAL
# ==============================================================================
st.title("🛡️ Radar de Fraude & Centro de Riesgo Fintech")
st.caption("Arquitectura Medallion en Streaming con Reglas Contables de Negocio e Inferencia Machine Learning")

# Guía didáctica de arquitectura
with st.expander("¿Cómo funciona esta plataforma? (Guía de arquitectura del sistema)", expanded=False):
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown("##### 1. Capa Bronze 🥉")
        st.caption("Ingesta continua en streaming desde Redpanda (Kafka) particionada por usuario. Micro-batching y almacenamiento inmutable en Parquet con metadatos de auditoría UTC.")
    with col_e2:
        st.markdown("##### 2. Capa Silver 🥈")
        st.caption("Procesamiento vectorizado con Polars. Deduplicación por clave de negocio, normalización y cálculo de discrepancias contables en balances emisores y receptores.")
    with col_e3:
        st.markdown("##### 3. Capa Gold y ML 🥇")
        st.caption("Motor híbrido: evalúa 3 reglas contables y ejecuta inferencia probabilística con un clasificador LightGBM. Genera un score combinado (50% Reglas + 50% ML).")
    with col_e4:
        st.markdown("##### 4. Servicio y Control 🏆")
        st.caption("Microservicio FastAPI para consultas en sub-milisegundos y este panel de Streamlit con monitoreo forense, perfiles de cliente y simulador interactivo.")

st.markdown("---")


# ==============================================================================
# VISTA 1: ARQUITECTURA & RESUMEN EJECUTIVO
# ==============================================================================
if "Arquitectura" in view_option:
    st.subheader("🏛️ Métricas de Operación y Estado de las Capas Medallion")

    health = fetch_data("/health")

    if isinstance(health, dict) and health.get("connection_error"):
        show_connection_error()
    elif health and "medallion_status" in health:
        m = health["medallion_status"]

        # Fila 1: KPIs de Infraestructura Medallion
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric(
            "Archivos Bronze",
            m["bronze_files"],
            help="Archivos Parquet crudos particionados por fecha e ingeridos desde Redpanda",
        )
        kpi2.metric(
            "Archivos Silver",
            m["silver_files"],
            help="Datasets depurados, deduplicados y enriquecidos con balance features con Polars",
        )
        kpi3.metric(
            "Alertas de Fraude",
            m["total_fraud_alerts"],
            help="Transacciones que superaron el umbral de riesgo (Score >= 35)",
        )
        kpi4.metric(
            "Perfiles de Cliente",
            m["total_user_profiles"],
            help="Clientes auditados con perfiles analíticos agregados en Capa Gold",
        )
        ml_on = m.get("ml_model_active", False)
        kpi5.metric(
            "Modelo ML",
            "LightGBM Activo" if ml_on else "Inactivo",
            help="Indica si el pipeline de LightGBM está cargado y ejecutando inferencia en producción",
        )

        st.markdown("---")

        # Cargar alertas para analítica ejecutiva
        alerts_data = fetch_data("/alerts", params={"limit": 500})
        if alerts_data and isinstance(alerts_data, list) and len(alerts_data) > 0:
            df_alerts = pd.DataFrame(alerts_data)

            # Métricas de impacto económico
            total_intercepted = df_alerts["amount"].sum()
            avg_amount_fraud = df_alerts["amount"].mean()
            high_count = len(df_alerts[df_alerts["risk_level"] == "HIGH"])

            ec1, ec2, ec3 = st.columns(3)
            ec1.metric(
                "💰 Volumen en Riesgo Interceptado",
                f"${total_intercepted:,.2f}",
                help="Monto acumulado en transacciones sospechosas detectadas por las reglas y el modelo",
            )
            ec2.metric(
                "🎯 Monto Promedio por Alerta",
                f"${avg_amount_fraud:,.2f}",
                help="Ticket promedio de operaciones fraudulentas interceptadas",
            )
            ec3.metric(
                "🚨 Alertas Críticas (HIGH)",
                f"{high_count} ({high_count/len(df_alerts)*100:.1f}%)",
                help="Transacciones con score >= 60 que requieren bloqueo preventivo inmediato",
            )

            st.markdown("---")

            # Pestañas analíticas internas
            tab_dist, tab_hybrid, tab_rules = st.tabs(
                [
                    "📊 Distribución de Alertas",
                    "⚖️ Matriz Híbrida (Reglas vs. ML)",
                    "📖 Diccionario de Reglas",
                ]
            )

            with tab_dist:
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.markdown("##### Proporción de Alertas por Severidad")
                    fig_pie = px.pie(
                        df_alerts,
                        names="risk_level",
                        hole=0.45,
                        color="risk_level",
                        color_discrete_map={
                            "HIGH": "#EF553B",
                            "MEDIUM": "#FFA15A",
                            "LOW": "#636EFA",
                        },
                    )
                    fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with g_col2:
                    st.markdown("##### Alertas por Tipo de Transacción")
                    fig_bar = px.histogram(
                        df_alerts,
                        x="type",
                        color="risk_level",
                        barmode="group",
                        color_discrete_map={
                            "HIGH": "#EF553B",
                            "MEDIUM": "#FFA15A",
                            "LOW": "#636EFA",
                        },
                        labels={"type": "Tipo de Operación", "count": "Cantidad"},
                    )
                    fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("##### Dispersión de Montos en Operaciones Sospechosas")
                fig_box = px.box(
                    df_alerts,
                    x="type",
                    y="amount",
                    color="risk_level",
                    points="outliers",
                    color_discrete_map={"HIGH": "#EF553B", "MEDIUM": "#FFA15A", "LOW": "#636EFA"},
                    labels={"amount": "Monto de la Operación ($)", "type": "Tipo"},
                )
                fig_box.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_box, use_container_width=True)

            with tab_hybrid:
                st.markdown("##### Matriz de Decisión Híbrida: Reglas Deterministas vs. LightGBM")
                st.info(
                    "Interpretación de Cuadrantes:\n"
                    "- Superior Derecho (Alto Riesgo): Coincidencia total. Las reglas detectaron inconsistencias severas y el modelo predice fraude con alta certeza.\n"
                    "- Superior Izquierdo (Fraude Oculto): El modelo detecta patrones no lineales sutiles aunque no se haya violado una regla contable explícita.\n"
                    "- Inferior Derecho (Anomalía Aislada): Violación de regla que requiere verificación manual pero con score ML moderado."
                )

                score_col = "rules_score" if "rules_score" in df_alerts.columns else "risk_score"
                ml_col = "ml_probability" if "ml_probability" in df_alerts.columns else None

                if ml_col and df_alerts[ml_col].max() > 0:
                    fig_scatter = px.scatter(
                        df_alerts,
                        x=score_col,
                        y=ml_col,
                        color="risk_level",
                        size="amount",
                        hover_data=["nameOrig", "nameDest", "type", "risk_score"],
                        labels={
                            score_col: "Puntaje de Reglas de Negocio (0 - 100)",
                            ml_col: "Probabilidad Estimada por LightGBM (0.0 - 1.0)",
                            "risk_level": "Severidad",
                        },
                        color_discrete_map={"HIGH": "#EF553B", "MEDIUM": "#FFA15A", "LOW": "#636EFA"},
                    )
                    fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("El modelo de ML aún no ha generado probabilidades en este lote.")

            with tab_rules:
                st.markdown("##### Reglas de Negocio Implementadas en la Capa Gold")
                r1_col, r2_col, r3_col = st.columns(3)
                with r1_col:
                    st.markdown("### Regla 1 (50 pts)")
                    st.markdown("**Inconsistencia Severa de Saldos**")
                    st.caption(
                        "Dispara cuando `balance_error_orig != 0` en operaciones TRANSFER o CASH_OUT con montos >= $200,000.\n\n"
                        "Justificación: En vaciados ilícitos de cuenta, los atacantes suelen transferir más fondos de los registrados contablemente."
                    )
                with r2_col:
                    st.markdown("### Regla 2 (35 pts)")
                    st.markdown("**Transferencia a Cero Destino**")
                    st.caption(
                        "Dispara cuando se transfieren >= $200,000 hacia cuentas destino con saldo previo y nuevo en cero.\n\n"
                        "Justificación: Patrón clásico de cuentas mula bancarias o cuentas puente utilizadas para extracción inmediata en efectivo."
                    )
                with r3_col:
                    st.markdown("### Regla 3 (25 pts)")
                    st.markdown("**Ráfaga Operativa (Burst)**")
                    st.caption(
                        "Dispara cuando un usuario realiza >= 2 transacciones de alto impacto en el mismo paso temporal (`step` = 1 hora).\n\n"
                        "Justificación: Detección de ataques de velocidad y dispersión automática programada mediante bots o scripts."
                    )
        else:
            st.info("ℹ️ No hay transacciones procesadas en la Capa Gold actualmente.")
            st.markdown("Puedes ejecutar el pipeline Medallion completo en vivo con un solo clic:")
            if st.button("⚡ Inyectar y Procesar Lote Transaccional en Vivo (1,000 eventos)", type="primary"):
                with st.spinner("Procesando pipeline Medallion (Bronze -> Silver con Polars -> Gold con LightGBM)..."):
                    res = fetch_data("/pipeline/simulate", method="POST", params={"count": 1000})
                    if res and res.get("status") == "success":
                        st.success(f"¡Lote procesado en {res.get('execution_time_seconds', 0)}s! Alertas generadas: {res.get('fraud_alerts_created')}. Recargando...")
                        st.rerun()
                    else:
                        st.error(f"Error al ejecutar simulación: {res}")
    else:
        st.warning("No se pudo obtener el estado de salud de la API.")


# ==============================================================================
# VISTA 2: MONITOR FORENSE DE ALERTAS
# ==============================================================================
elif "Monitor" in view_option:
    st.subheader("🚨 Auditoría Forense y Explorador de Alertas de Fraude")
    st.caption("Filtra, analiza el detalle forense de cada transacción sospechosa y exporta reportes para Compliance.")

    f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1])
    with f_col1:
        risk_filter = st.selectbox("Filtrar por Severidad:", ["TODOS", "HIGH", "MEDIUM", "LOW"])
    with f_col2:
        rule_filter = st.text_input("Filtrar por Regla (ej. BALANCE, ZERO, BURST):")
    with f_col3:
        ml_prob_filter = st.slider("Probabilidad Mínima ML (%):", min_value=0, max_value=100, value=0, step=5)
    with f_col4:
        limit_filter = st.slider("Límite de Registros:", min_value=10, max_value=500, value=50, step=10)

    params: Dict[str, Any] = {"limit": limit_filter}
    if risk_filter != "TODOS":
        params["risk_level"] = risk_filter
    if rule_filter.strip():
        params["rule"] = rule_filter.strip()
    if ml_prob_filter > 0:
        params["min_ml_prob"] = ml_prob_filter / 100.0

    alerts_res = fetch_data("/alerts", params=params)

    if isinstance(alerts_res, dict) and alerts_res.get("connection_error"):
        show_connection_error()
    elif isinstance(alerts_res, list) and len(alerts_res) > 0:
        df = pd.DataFrame(alerts_res)
        st.caption(f"Mostrando **{len(df)}** alertas coincidentes con los criterios de búsqueda.")

        cols_display = [
            "alert_id", "step", "type", "amount", "nameOrig", "nameDest",
            "ml_probability", "risk_score", "risk_level", "rules_triggered", "alert_timestamp"
        ]
        available_cols = [c for c in cols_display if c in df.columns]

        df_display = df[available_cols].copy()
        if "amount" in df_display.columns:
            df_display["amount"] = df_display["amount"].apply(lambda x: f"${x:,.2f}")
        if "ml_probability" in df_display.columns:
            df_display["ml_probability"] = df_display["ml_probability"].apply(lambda x: f"{x * 100:.1f}%")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        col_exp1, col_exp2 = st.columns([2, 1])
        with col_exp1:
            # Selector para inspección forense detallada
            alert_options = df["alert_id"].tolist()
            selected_alert_id = st.selectbox("Seleccionar ID de Alerta para Inspección Detallada:", alert_options)
        with col_exp2:
            st.write("")
            st.write("")
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar Reporte en CSV",
                data=csv_data,
                file_name=f"fraud_alerts_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Ficha forense detallada de la alerta seleccionada
        if selected_alert_id:
            alert_detail = df[df["alert_id"] == selected_alert_id].iloc[0]
            with st.container():
                st.markdown("---")
                st.markdown(f"#### 🔍 Diagnóstico Forense: Alerta `{selected_alert_id}`")

                det_c1, det_c2, det_c3, det_c4 = st.columns(4)
                det_c1.metric("Cuenta Origen", alert_detail.get("nameOrig", "N/A"))
                det_c2.metric("Cuenta Destino", alert_detail.get("nameDest", "N/A"))
                det_c3.metric(
                    "Score Combinado",
                    f"{alert_detail.get('risk_score', 0)} / 100",
                    delta=f"{alert_detail.get('risk_level', 'LOW')}",
                )
                ml_p_val = alert_detail.get("ml_probability", 0.0)
                det_c4.metric(
                    "Probabilidad LightGBM",
                    f"{ml_p_val * 100:.1f}%",
                    help="Probabilidad estimada de que esta transacción sea fraudulenta",
                )

                st.markdown(f"**Reglas Violadas**: `{alert_detail.get('rules_triggered', 'Ninguna')}`")
                st.markdown(f"**Monto Involucrado**: `${alert_detail.get('amount', 0):,.2f}` | **Paso Temporal**: Step {alert_detail.get('step', 'N/A')}")
    else:
        st.info("No se encontraron alertas que coincidan con los criterios seleccionados.")


# ==============================================================================
# VISTA 3: AUDITORÍA DE CLIENTE (KYC)
# ==============================================================================
elif "KYC" in view_option:
    st.subheader("👤 Auditoría de Cliente y Análisis de Riesgo (KYC)")
    st.caption("Explora el comportamiento transaccional histórico, acumulación de alertas y discrepancias de balance por usuario.")

    # Acceso rápido con 1 clic para no obligar al usuario a adivinar IDs
    st.markdown("**Cuentas de Demostración para Prueba Rápida:**")
    qc1, qc2, qc3 = st.columns(3)

    if "target_user" not in st.session_state:
        st.session_state.target_user = "C1057507014"

    if qc1.button("🚨 Caso Crítico: C1057507014", use_container_width=True):
        st.session_state.target_user = "C1057507014"
    if qc2.button("⚠️ Riesgo Medio: C684230144", use_container_width=True):
        st.session_state.target_user = "C684230144"
    if qc3.button("✅ Cliente Normal: C1231006815", use_container_width=True):
        st.session_state.target_user = "C1231006815"

    col_search, col_btn = st.columns([3, 1])
    with col_search:
        user_query = st.text_input(
            "Ingrese el ID de Cuenta Origen (nameOrig):",
            value=st.session_state.target_user,
            help="Ejemplo: C1057507014 (Alto Riesgo)",
        )
    with col_btn:
        st.write("")
        st.write("")
        search_pressed = st.button("🔍 Consultar Perfil", use_container_width=True)

    if user_query:
        profile_res = fetch_data(f"/users/{user_query.strip()}/risk-profile")

        if isinstance(profile_res, dict) and profile_res.get("connection_error"):
            show_connection_error()
        elif isinstance(profile_res, dict) and profile_res.get("status_code") == 404:
            st.warning(f"Usuario '{user_query.strip()}' no encontrado en la Capa Gold.")
        elif isinstance(profile_res, dict) and "nameOrig" in profile_res:
            p = profile_res
            st.success(f"Ficha técnica cargada para el cliente **{p['nameOrig']}**")

            # Determinación de severidad y recomendación de cumplimiento
            risk = p.get("user_risk_level", "LOW")
            if risk == "HIGH":
                risk_badge = "🔴 ALTO RIESGO (HIGH)"
                action_text = "Acción Recomendada: Bloqueo preventivo de operaciones salientes y solicitud de validación biométrica / KYC reforzado."
                alert_type_callout = st.error
            elif risk == "MEDIUM":
                risk_badge = "🟡 RIESGO MEDIO (MEDIUM)"
                action_text = "Acción Recomendada: Habilitar monitoreo reforzado y exigir segundo factor de autenticación (2FA) para montos elevados."
                alert_type_callout = st.warning
            else:
                risk_badge = "🟢 RIESGO BAJO (LOW)"
                action_text = "Acción Recomendada: Operación habitual autorizada. Sin indicios de comportamiento malicioso."
                alert_type_callout = st.success

            alert_type_callout(f"**Clasificación**: {risk_badge}\n\n{action_text}")

            # Fila 1: KPIs Principales
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Tipo de Cuenta", p.get("account_type", "CLIENT"))
            kpi2.metric("Total Transacciones", p.get("total_transactions", 0))
            kpi3.metric("Volumen Total", f"${p.get('total_volume', 0):,.2f}")
            kpi4.metric("Monto Máximo", f"${p.get('max_transaction_amount', 0):,.2f}")

            # Fila 2: Desglose Operativo y Discrepancias
            st.markdown("##### Desglose Operativo y Alertas Acumuladas")
            op1, op2, op3, op4 = st.columns(4)
            op1.metric("Transferencias (TRANSFER)", p.get("transfer_count", 0))
            op2.metric("Retiros (CASH_OUT)", p.get("cash_out_count", 0))
            op3.metric("Alertas Críticas Emitidas", p.get("high_risk_alerts", 0))
            op4.metric("Error de Balance Promedio", f"${p.get('avg_balance_error', 0):,.2f}")

            with st.expander("Ver Datos Crudos del Perfil (JSON)"):
                st.json(p)
        else:
            st.info("Presiona 'Consultar Perfil' para explorar los datos.")


# ==============================================================================
# VISTA 4: SIMULADOR DE TRANSACCIONES (SANDBOX)
# ==============================================================================
elif "Simulador" in view_option:
    st.subheader("🧪 Simulador de Transacciones en Tiempo Real (Sandbox)")
    st.caption("Prueba el motor híbrido (Reglas + LightGBM) evaluando transacciones simuladas al instante.")

    st.markdown("**Cargar Escenario Preconfigurado de Prueba:**")
    s_col1, s_col2, s_col3 = st.columns(3)

    if "sim_data" not in st.session_state:
        st.session_state.sim_data = {
            "type": "TRANSFER",
            "amount": 250000.0,
            "oldbalanceOrg": 250000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "nameOrig": "C_SIMULATED_USER",
            "nameDest": "M_MERCHANT_DEST",
        }

    if s_col1.button("🚨 Escenario A: Vaciado Total", use_container_width=True):
        st.session_state.sim_data = {
            "type": "TRANSFER",
            "amount": 350000.0,
            "oldbalanceOrg": 350000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "nameOrig": "C_ATTACKER_ACCOUNT",
            "nameDest": "M_MULE_ACCOUNT",
        }
    if s_col2.button("⚠️ Escenario B: Cuenta Mula", use_container_width=True):
        st.session_state.sim_data = {
            "type": "TRANSFER",
            "amount": 210000.0,
            "oldbalanceOrg": 250000.0,
            "newbalanceOrig": 40000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "nameOrig": "C_SUSPICIOUS_TRANSFER",
            "nameDest": "C_ZERO_BALANCE_MULE",
        }
    if s_col3.button("✅ Escenario C: Compra Cotidiana", use_container_width=True):
        st.session_state.sim_data = {
            "type": "PAYMENT",
            "amount": 120.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4880.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "nameOrig": "C_REGULAR_CUSTOMER",
            "nameDest": "M_LOCAL_GROCERY",
        }

    st.markdown("---")

    # Formulario interactivo
    form_c1, form_c2 = st.columns(2)
    with form_c1:
        st.markdown("##### Parámetros de la Transacción")
        sim_type = st.selectbox(
            "Tipo de Operación:",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(st.session_state.sim_data["type"]),
        )
        sim_amount = st.number_input(
            "Monto de la Transacción ($):",
            min_value=1.0,
            max_value=10000000.0,
            value=float(st.session_state.sim_data["amount"]),
            step=1000.0,
        )
        sim_name_orig = st.text_input("Cuenta de Origen:", value=st.session_state.sim_data["nameOrig"])
        sim_name_dest = st.text_input("Cuenta de Destino:", value=st.session_state.sim_data["nameDest"])

    with form_c2:
        st.markdown("##### Balances Contables Origen & Destino")
        sim_old_orig = st.number_input(
            "Saldo Origen Antes ($):",
            min_value=0.0,
            value=float(st.session_state.sim_data["oldbalanceOrg"]),
            step=1000.0,
        )
        sim_new_orig = st.number_input(
            "Saldo Origen Después ($):",
            min_value=0.0,
            value=float(st.session_state.sim_data["newbalanceOrig"]),
            step=1000.0,
        )
        sim_old_dest = st.number_input(
            "Saldo Destino Antes ($):",
            min_value=0.0,
            value=float(st.session_state.sim_data["oldbalanceDest"]),
            step=1000.0,
        )
        sim_new_dest = st.number_input(
            "Saldo Destino Después ($):",
            min_value=0.0,
            value=float(st.session_state.sim_data["newbalanceDest"]),
            step=1000.0,
        )

    # Discrepancias calculadas
    err_orig = (sim_old_orig - sim_amount) - sim_new_orig
    err_dest = (sim_old_dest + sim_amount) - sim_new_dest

    st.caption(f"Discrepancia contable calculada: Origen: **${err_orig:,.2f}** | Destino: **${err_dest:,.2f}**")

    st.markdown("---")
    eval_btn = st.button("⚡ Evaluar Transacción en Tiempo Real", type="primary", use_container_width=True)

    if eval_btn:
        payload = {
            "step": 1,
            "type": sim_type,
            "amount": sim_amount,
            "nameOrig": sim_name_orig,
            "oldbalanceOrg": sim_old_orig,
            "newbalanceOrig": sim_new_orig,
            "nameDest": sim_name_dest,
            "oldbalanceDest": sim_old_dest,
            "newbalanceDest": sim_new_dest,
        }

        eval_res = fetch_data("/evaluate", method="POST", json_data=payload)

        if isinstance(eval_res, dict) and eval_res.get("connection_error"):
            show_connection_error()
        elif eval_res and "risk_score" in eval_res:
            res = eval_res
            score = res.get("risk_score", 0)
            level = res.get("risk_level", "NONE")
            rec = res.get("recommendation", "")
            rules = res.get("rules_triggered", "NINGUNA")
            ml_p = res.get("ml_probability", 0.0)
            r_pts = res.get("rules_score", 0)

            st.markdown("### Veredicto del Motor Antifraude:")

            if level == "HIGH":
                st.error(f"### 🔴 NIVEL DE RIESGO: ALTO (Score: {score}/100)\n\n**Acción recomendada**: {rec}")
            elif level == "MEDIUM":
                st.warning(f"### 🟡 NIVEL DE RIESGO: MEDIO (Score: {score}/100)\n\n**Acción recomendada**: {rec}")
            elif level == "LOW":
                st.info(f"### 🔵 NIVEL DE RIESGO: BAJO (Score: {score}/100)\n\n**Acción recomendada**: {rec}")
            else:
                st.success(f"### 🟢 OPERACIÓN APROBADA (Score: {score}/100)\n\n**Acción recomendada**: {rec}")

            # Gráfico de Tacómetro / Gauge con Plotly
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Puntaje Híbrido de Riesgo (0-100)", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": res.get("decision_color", "#636EFA")},
                        "steps": [
                            {"range": [0, 35], "color": "#E8F5E9"},
                            {"range": [35, 60], "color": "#FFF3E0"},
                            {"range": [60, 100], "color": "#FFEBEE"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 60,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Desglose de Puntos
            st.markdown("##### Desglose de Factores de Riesgo:")
            d_c1, d_c2, d_c3 = st.columns(3)
            d_c1.metric(
                "Puntaje por Reglas (50%)",
                f"{r_pts} pts",
                help="Puntos otorgados por las 3 reglas deterministas de negocio contable",
            )
            d_c2.metric(
                "Probabilidad LightGBM (50%)",
                f"{ml_p * 100:.1f}%",
                help="Inferencia estadística sobre patrones transaccionales complejos",
            )
            d_c3.metric(
                "Reglas Violadas",
                rules,
                help="Reglas deterministas activadas por los valores de la transacción",
            )
        else:
            st.error(f"Error al evaluar la transacción: {eval_res}")
