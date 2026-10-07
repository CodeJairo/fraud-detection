"""
Dashboard interactivo de Detección de Fraude y Monitoreo de Riesgo con Streamlit y Plotly.
Consume los endpoints del microservicio FastAPI para explorar el estado del pipeline Medallion,
las alertas de fraude y los perfiles de riesgo de clientes en tiempo real.
"""

import os
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fintech Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)


def fetch_data(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    Realiza peticiones GET a la API con manejo controlado de errores y timeouts.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
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
    """Muestra un mensaje explicativo si el microservicio FastAPI no está disponible."""
    st.error(
        f"⚠️ **No se pudo conectar con el microservicio API en `{API_BASE_URL}`**\n\n"
        "Asegúrate de que el backend de FastAPI esté en ejecución con el siguiente comando:\n\n"
        "```bash\n"
        "uv run uvicorn src.api.main:app --port 8000\n"
        "```"
    )


# --- Barra Lateral de Navegación ---
st.sidebar.title("🛡️ Fraud Engine")
st.sidebar.caption("Pipeline Medallion + Streaming")
st.sidebar.markdown("---")
view_option = st.sidebar.radio(
    "Seleccionar Vista:",
    ["📊 Resumen General", "🚨 Monitor de Alertas", "👤 Perfil de Cliente"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Stack Tecnológico:**\n"
    "- Ingesta: Redpanda (Kafka)\n"
    "- Procesamiento: Polars & PyArrow\n"
    "- Microservicio: FastAPI\n"
    "- Dashboard: Streamlit & Plotly"
)

# Encabezado principal
st.title("🛡️ Sistema de Detección de Fraude & Monitoreo de Riesgo")
st.caption("Panel de control analítico en tiempo real sobre la Arquitectura Medallion")

# ==============================================================================
# VISTA 1: RESUMEN GENERAL
# ==============================================================================
if view_option == "📊 Resumen General":
    st.subheader("Estado de las Capas Medallion & Métricas Globales")
    health = fetch_data("/health")

    if isinstance(health, dict) and health.get("connection_error"):
        show_connection_error()
    elif health and "medallion_status" in health:
        m = health["medallion_status"]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📦 Archivos Bronze", m["bronze_files"])
        col2.metric("🧹 Archivos Silver", m["silver_files"])
        col3.metric("🚨 Alertas de Fraude", m["total_fraud_alerts"])
        col4.metric("👤 Perfiles Cliente", m["total_user_profiles"])
        ml_active = m.get("ml_model_active", False)
        col5.metric("🧠 Modelo ML", "LightGBM Activo" if ml_active else "Solo Reglas")

        st.markdown("---")

        # Cargar muestra de alertas para gráficos
        alerts_data = fetch_data("/alerts", params={"limit": 500})
        if alerts_data and isinstance(alerts_data, list) and len(alerts_data) > 0:
            df_alerts = pd.DataFrame(alerts_data)

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown("##### Proporción por Nivel de Severidad")
                fig_pie = px.pie(
                    df_alerts,
                    names="risk_level",
                    hole=0.4,
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
                    labels={"type": "Tipo de Transacción", "count": "Cantidad"},
                )
                fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("##### Distribución de Montos en Transacciones Sospechosas")
            fig_hist = px.box(
                df_alerts,
                x="type",
                y="amount",
                color="risk_level",
                points="outliers",
                color_discrete_map={"HIGH": "#EF553B", "MEDIUM": "#FFA15A", "LOW": "#636EFA"},
                labels={"amount": "Monto ($)", "type": "Tipo"},
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

            # Matriz Híbrida: Reglas vs. ML si hay probabilidades disponibles
            if "ml_probability" in df_alerts.columns and df_alerts["ml_probability"].max() > 0:
                st.markdown("##### Matriz de Decisión Híbrida: Reglas de Negocio vs. Probabilidad de ML")
                score_col = "rules_score" if "rules_score" in df_alerts.columns else "risk_score"
                fig_scatter = px.scatter(
                    df_alerts,
                    x=score_col,
                    y="ml_probability",
                    color="risk_level",
                    size="amount",
                    hover_data=["nameOrig", "nameDest", "type", "risk_score"],
                    labels={
                        score_col: "Puntaje de Reglas (0-100)",
                        "ml_probability": "Probabilidad ML (0.0 - 1.0)",
                        "risk_level": "Nivel de Riesgo",
                    },
                    color_discrete_map={"HIGH": "#EF553B", "MEDIUM": "#FFA15A", "LOW": "#636EFA"},
                )
                fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No hay suficientes alertas para generar gráficos estadísticos.")
    else:
        st.warning("No se pudo obtener el estado de salud de la API.")

# ==============================================================================
# VISTA 2: MONITOR DE ALERTAS
# ==============================================================================
elif view_option == "🚨 Monitor de Alertas":
    st.subheader("Explorador y Auditoría de Alertas de Fraude")

    f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1])
    with f_col1:
        risk_filter = st.selectbox("Filtrar por Nivel de Riesgo:", ["TODOS", "HIGH", "MEDIUM", "LOW"])
    with f_col2:
        rule_filter = st.text_input("Filtrar por Regla (ej. BALANCE, BURST, ZERO):")
    with f_col3:
        ml_prob_filter = st.slider("Prob. Mínima ML (%):", min_value=0, max_value=100, value=0, step=5)
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

        st.caption(f"Mostrando **{len(df)}** alertas encontradas.")

        cols_display = [
            "alert_id", "step", "type", "amount", "nameOrig", "nameDest",
            "ml_probability", "risk_score", "risk_level", "rules_triggered", "alert_timestamp"
        ]
        available_cols = [c for c in cols_display if c in df.columns]

        # Formatear montos y probabilidades para legibilidad
        df_display = df[available_cols].copy()
        if "amount" in df_display.columns:
            df_display["amount"] = df_display["amount"].apply(lambda x: f"${x:,.2f}")
        if "ml_probability" in df_display.columns:
            df_display["ml_probability"] = df_display["ml_probability"].apply(lambda x: f"{x * 100:.1f}%")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Botón de exportación para analistas
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar Alertas como CSV",
            data=csv_data,
            file_name="fraud_alerts_export.csv",
            mime="text/csv",
        )
    else:
        st.info("No se encontraron alertas que coincidan con los criterios seleccionados.")

# ==============================================================================
# VISTA 3: PERFIL DE CLIENTE
# ==============================================================================
elif view_option == "👤 Perfil de Cliente":
    st.subheader("Búsqueda y Ficha Técnica de Riesgo por Cliente")

    col_search, col_btn = st.columns([3, 1])
    with col_search:
        user_query = st.text_input(
            "Ingrese el ID de Cuenta Origen (nameOrig):",
            value="C1057507014",
            help="Ejemplo de cuenta con alto riesgo: C1057507014",
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
            st.warning(f"⚠️ {profile_res.get('error', 'Usuario no encontrado')}")
        elif isinstance(profile_res, dict) and "nameOrig" in profile_res:
            p = profile_res
            st.success(f"Ficha técnica cargada para el cliente **{p['nameOrig']}**")

            # Indicador de Riesgo
            risk = p.get("user_risk_level", "LOW")
            if risk == "HIGH":
                risk_badge = "🔴 ALTO RIESGO (HIGH)"
            elif risk == "MEDIUM":
                risk_badge = "🟡 RIESGO MEDIO (MEDIUM)"
            else:
                risk_badge = "🟢 RIESGO BAJO (LOW)"

            # Fila 1: KPIs Principales
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Clasificación de Riesgo", risk_badge)
            kpi2.metric("Total Transacciones", p.get("total_transactions", 0))
            kpi3.metric("Volumen Total", f"${p.get('total_volume', 0):,.2f}")
            kpi4.metric("Monto Máximo", f"${p.get('max_transaction_amount', 0):,.2f}")

            # Fila 2: Desglose Operativo y Alertas
            st.markdown("##### Desglose Operativo y Discrepancias")
            op1, op2, op3, op4 = st.columns(4)
            op1.metric("Transferencias (TRANSFER)", p.get("transfer_count", 0))
            op2.metric("Retiros (CASH_OUT)", p.get("cash_out_count", 0))
            op3.metric("Alertas Críticas Emitidas", p.get("high_risk_alerts", 0))
            op4.metric("Discrepancia Saldo Promedio", f"${p.get('avg_balance_error', 0):,.2f}")

            # Detalle técnico JSON expandible
            with st.expander("Ver Datos Crudos del Perfil (JSON)"):
                st.json(p)
        else:
            st.info("Presiona 'Consultar Perfil' para ver la información.")
