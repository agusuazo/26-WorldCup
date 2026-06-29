"""CSS global y utilidades de estilo para el dashboard público."""
import streamlit as st


def inject_global_css() -> None:
    """Inyecta CSS mobile-first y estilos de recuadros explicativos.

    Llamar al inicio de cada página, después de set_page_config().
    """
    st.markdown("""
<style>
/* ── Mobile: apilar columnas verticalmente ────────────────────────── */
@media (max-width: 768px) {
    /* Columnas de Streamlit en bloque */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 0 100% !important;
        min-width: 100% !important;
    }
    /* Métricas más compactas */
    [data-testid="stMetric"] label { font-size: 0.72rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    /* Tablas: scroll horizontal en vez de desborde */
    [data-testid="stDataFrame"],
    .stDataFrame { overflow-x: auto !important; }
    /* Padding reducido del contenedor principal */
    .block-container { padding: 0.75rem 0.5rem !important; }
    /* Plotly: limitar altura en mobile */
    .js-plotly-plot .plotly { max-height: 320px; }
    /* Sidebar ocupa toda la pantalla cuando está abierto */
    [data-testid="stSidebar"] { min-width: 260px !important; }
    /* Tabs: texto más pequeño */
    [data-testid="stTabs"] button { font-size: 0.78rem !important; padding: 6px 8px !important; }
    /* Ocultar barra de relleno de columnas vacías */
    [data-testid="stVerticalBlock"] > div:empty { display: none; }
}

/* ── Recuadros de glosario / explicación ────────────────────────────── */
.info-box {
    background: #f0f7ff;
    border-left: 4px solid #2196F3;
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.92rem;
    line-height: 1.6;
}
.info-box ul { margin: 6px 0 0 16px; padding: 0; }
.info-box li { margin-bottom: 4px; }
.glossary-term { font-weight: 700; color: #1565C0; }

/* ── Mejora visual general ───────────────────────────────────────────── */
/* Tabla de datos: filas con bordes más suaves */
[data-testid="stDataFrame"] table { border-collapse: collapse; }
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    white-space: nowrap;
}
/* Números en métricas alineados */
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)


def info_box(html: str) -> None:
    """Renderiza un recuadro informativo azul con HTML interno."""
    st.markdown(f'<div class="info-box">{html}</div>', unsafe_allow_html=True)
