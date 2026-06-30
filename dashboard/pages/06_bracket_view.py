"""Vista de Bracket — un torneo simulado completo: grupos, llaves y caminos.

Dos modos:
- "Escenario más probable": determinista — standings por puntos esperados,
  en eliminatorias avanza el favorito de cada cruce.
- "Simulación aleatoria": una iteración Monte Carlo; re-simular genera otro
  escenario posible.

El escenario más probable es UNA trayectoria ilustrativa (el camino modal),
no la predicción agregada: para probabilidades reales por equipo y ronda
está la página Tournament Simulator.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard.components.styles import inject_global_css, info_box

st.set_page_config(page_title="Vista de Bracket", page_icon="🗂️", layout="wide")
inject_global_css()
st.title("🗂️ Vista de Bracket — WC 2026 simulado")

with st.expander("ℹ️ Cómo leer esta página", expanded=False):
    info_box("""
    <b>Visualización de un torneo completo simulado.</b><br><br>
    <ul>
      <li><span class="glossary-term">Escenario más probable:</span>
          En cada cruce avanza el equipo que el modelo considera favorito. Es una trayectoria
          <b>ilustrativa</b> — no significa que sea lo que va a pasar, sino el camino más lógico
          según las probabilidades. Las chances reales por ronda están en Tournament Simulator.</li>
      <li><span class="glossary-term">Simulación aleatoria:</span>
          Una sola "tirada" del torneo donde los resultados se sortean según las probabilidades.
          Hacé clic en "Re-simular" para ver otra posible versión del Mundial.</li>
      <li><span class="glossary-term">📌 Resultado real:</span>
          Los partidos marcados con este ícono ya se jugaron y se muestra el resultado real,
          no simulado.</li>
      <li><span class="glossary-term">✅ Clasifica:</span> Equipo que avanza (1° o 2° del grupo).</li>
      <li><span class="glossary-term">3️⃣ Mejor tercero:</span>
          Equipo que terminó 3° en su grupo pero clasifica entre los 8 mejores terceros del torneo.</li>
      <li><span class="glossary-term">❌ Eliminado:</span> Equipo que no avanza de la fase de grupos.</li>
    </ul>
    """)

# ---- Simulación cacheada ------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Simulando torneo...")
def get_detail(mode: str, seed: int, conditioned: bool) -> dict:
    from dashboard.components.model_store import load_best_predictor
    from src.simulation.monte_carlo import (load_tournament_state,
                                            simulate_tournament_detail)
    predictor = load_best_predictor()
    state = load_tournament_state() if conditioned else None
    return simulate_tournament_detail(predictor, seed=seed, mode=mode,
                                      state=state)


# ---- Sidebar -------------------------------------------------------------
if "bracket_seed" not in st.session_state:
    st.session_state.bracket_seed = 42

with st.sidebar:
    st.header("Configuracion")
    mode_label = st.radio("Modo", ["Escenario más probable", "Simulación aleatoria"])
    mode = "expected" if mode_label == "Escenario más probable" else "sample"
    if mode == "sample":
        if st.button("🎲 Re-simular", type="primary"):
            st.session_state.bracket_seed += 1
        st.caption(f"Seed actual: {st.session_state.bracket_seed}")
    conditioned = st.toggle("Condicionar a resultados reales", value=True,
                            help="Los partidos ya jugados del torneo se "
                                 "muestran con su resultado real (📌)")
    st.divider()

detail = get_detail(mode, st.session_state.bracket_seed, conditioned)
played_pairs = detail.get("played_pairs", set())

all_wc_teams = sorted({t for g in detail["groups"].values()
                       for t, *_ in g["standings"]})
with st.sidebar:
    highlight = st.selectbox("Resaltar equipo", ["(ninguno)"] + all_wc_teams)
    if highlight == "(ninguno)":
        highlight = None
    st.divider()
    view_mode = st.radio("Vista de llaves", ["📋 Por rondas", "🏆 Bracket clásico"])

if mode == "expected":
    st.info("**Escenario más probable**: en cada cruce avanza el favorito del modelo. "
            "Es una trayectoria ilustrativa — las probabilidades reales por ronda están "
            "en Tournament Simulator.")

# ---- Fase de grupos -------------------------------------------------------
st.subheader("Fase de grupos — clasificación final")
is_expected = detail["mode"] == "expected"
fmt_pts = "{:.1f}" if is_expected else "{:.0f}"

group_names = sorted(detail["groups"].keys())
for row_start in range(0, 12, 4):
    cols = st.columns(4)
    for col, g_name in zip(cols, group_names[row_start:row_start + 4]):
        g = detail["groups"][g_name]
        q = g["qualified"]
        with col:
            n_real = sum(1 for h, a, *_ in g["results"]
                         if frozenset((h, a)) in played_pairs)
            pin = f" 📌{n_real}" if n_real else ""
            st.markdown(f"**Grupo {g_name}**{pin}")
            lines = []
            for pos, (team, pts, gd, gf) in enumerate(g["standings"], 1):
                if team == q["first"] or team == q["second"]:
                    mark = "✅"
                elif team == q["third"]:
                    mark = "3️⃣"
                else:
                    mark = "❌"
                style = "**" if team == highlight else ""
                lines.append(f"{pos}. {mark} {style}{team}{style} "
                             f"({fmt_pts.format(pts)} pts, {gd:+.1f})")
            st.markdown("\n".join(f"- {l}" for l in lines))

st.caption("✅ clasifica directo (1° y 2°) · 3️⃣ clasifica entre los 8 mejores terceros · "
           "❌ eliminado" + (" · puntos esperados (no enteros)" if is_expected else ""))

st.divider()

# ---- Llaves por tabs (mobile-friendly) ------------------------------------
st.subheader("Llaves de eliminación directa")
st.caption("Navegá entre rondas con las pestañas. 📌 = resultado real ya jugado.")

ROUND_LABELS = [("r32", "Ronda de 32"), ("r16", "Octavos"), ("qf", "Cuartos"),
                ("sf", "Semifinal"), ("final", "Final")]


def match_html(a: str, b: str, w: str, highlight: str | None,
               is_real: bool = False) -> str:
    """Bloque HTML de un cruce; ganador en negrita, resaltado si participa."""
    involved = highlight in (a, b) if highlight else False
    bg = "#fff3cd" if involved else "#f8f9fa"
    border = "#e0a800" if involved else "#dee2e6"
    pin = ("<div style='font-size:0.68rem;color:#888'>📌 resultado real</div>"
           if is_real else "")

    def line(t):
        is_w = t == w
        weight = "700" if is_w else "400"
        color = "#1a7f37" if is_w else "#555"
        icon = " ✓" if is_w else ""
        return (f"<div style='font-weight:{weight};color:{color};"
                f"font-size:0.85rem;line-height:1.6'>{t}{icon}</div>")

    return (f"<div style='background:{bg};border:1px solid {border};"
            f"border-radius:6px;padding:8px 10px;margin-bottom:8px'>"
            f"{line(a)}{line(b)}{pin}</div>")


def _bracket_card(a: str, b: str, w: str, highlight: str | None, is_real: bool) -> str:
    """Tarjeta compacta para la vista de bracket clásico."""
    involved = highlight in (a, b) if highlight else False
    card_bg = "#fff3cd" if involved else "#f8f9fa"
    card_border = "2px solid #e0a800" if involved else "1px solid #dee2e6"
    badge = "<div style='font-size:0.58rem;color:#999;padding:1px 6px'>📌</div>" if is_real else ""

    def row(t):
        is_w = (t == w) if w else False
        bg = "#d4edda" if is_w else "transparent"
        fg = "#1a7f37" if is_w else ("#aaa" if not w else "#555")
        fw = "700" if is_w else "400"
        return (f"<div style='background:{bg};color:{fg};font-weight:{fw};"
                f"font-size:0.78rem;padding:4px 7px;line-height:1.5'>{t}{' ✓' if is_w else ''}</div>")

    return (f"<div style='background:{card_bg};border:{card_border};border-radius:5px;"
            f"overflow:hidden;margin:2px 0;width:158px'>{badge}{row(a)}{row(b)}</div>")


def bracket_html(rounds: dict, played_pairs: set, highlight: str | None, champion: str) -> str:
    """HTML completo del bracket clásico horizontal (5 rondas + campeón)."""
    ROUND_ORDER = [("r32", "R32"), ("r16", "Octavos"), ("qf", "Cuartos"),
                   ("sf", "Semis"), ("final", "Final")]
    H = 900

    def connector():
        return (f"<div style='display:flex;flex-direction:column;width:16px'>"
                f"<div style='flex:1;border-right:1px solid #ccc;border-bottom:1px solid #ccc;"
                f"border-radius:0 0 4px 0'></div>"
                f"<div style='flex:1;border-right:1px solid #ccc;border-top:1px solid #ccc;"
                f"border-radius:0 4px 0 0'></div></div>")

    cols_html = ""
    for col_idx, (rname, rlabel) in enumerate(ROUND_ORDER):
        is_last = (col_idx == len(ROUND_ORDER) - 1)
        matches = rounds.get(rname, [])
        pairs_html = ""
        for i in range(0, len(matches), 2):
            pair = matches[i:i + 2]
            cards = ""
            for a, b, w in pair:
                is_real = frozenset((a, b)) in played_pairs
                cards += _bracket_card(a, b, w, highlight, is_real)
            if len(pair) == 2:
                conn = "" if is_last else connector()
            else:
                conn = ("" if is_last else
                        "<div style='width:16px;border-top:1px solid #ccc;align-self:center'></div>")
            pairs_html += (f"<div style='display:flex;flex-direction:row'>"
                           f"<div style='display:flex;flex-direction:column;"
                           f"justify-content:space-around;flex:1'>{cards}</div>"
                           f"{conn}</div>")

        cols_html += (f"<div style='display:flex;flex-direction:column;align-items:center;margin-right:4px'>"
                      f"<div style='font-size:0.65rem;text-transform:uppercase;color:#888;"
                      f"letter-spacing:0.05em;margin-bottom:6px'>{rlabel}</div>"
                      f"<div style='display:flex;flex-direction:column;justify-content:space-around;"
                      f"height:{H}px;width:176px'>{pairs_html}</div></div>")

    champ_bg = "#fff3cd" if highlight == champion else "#d4edda"
    champ_border = "#e0a800" if highlight == champion else "#1a7f37"
    champion_col = (f"<div style='display:flex;flex-direction:column;align-items:center;"
                    f"justify-content:center;width:120px;height:{H}px;margin-left:8px'>"
                    f"<div style='font-size:0.65rem;text-transform:uppercase;color:#888;"
                    f"margin-bottom:6px'>Campeón</div>"
                    f"<div style='text-align:center;padding:12px 10px;background:{champ_bg};"
                    f"border:2px solid {champ_border};border-radius:8px;font-weight:700;"
                    f"font-size:0.85rem;color:#155724'>🏆<br>{champion}</div></div>")

    return (f"<div style='display:flex;flex-direction:row;overflow-x:auto;font-family:sans-serif;"
            f"padding:8px;background:#fff;border-radius:8px'>{cols_html}{champion_col}</div>")


if view_mode == "📋 Por rondas":
    # Tabs para cada ronda — caben perfectamente en mobile
    tab_labels = [rlabel for _, rlabel in ROUND_LABELS] + ["🏆 Campeón"]
    tabs = st.tabs(tab_labels)

    for tab, (rname, rlabel) in zip(tabs[:-1], ROUND_LABELS):
        with tab:
            matches = detail["rounds"][rname]
            if not matches:
                st.caption("Sin cruces en esta ronda todavía.")
                continue
            # En mobile mostramos en 1 col; en desktop usamos 2 cols para aprovechar espacio
            cols = st.columns(min(2, len(matches)))
            for idx, (a, b, w) in enumerate(matches):
                is_real = frozenset((a, b)) in played_pairs
                with cols[idx % len(cols)]:
                    st.markdown(match_html(a, b, w, highlight, is_real=is_real),
                                unsafe_allow_html=True)

    with tabs[-1]:
        champion = detail["champion"]
        involved = highlight == champion if highlight else False
        bg = "#d4edda" if not involved else "#fff3cd"
        border = "#1a7f37" if not involved else "#e0a800"
        st.markdown(
            f"<div style='background:{bg};border:2px solid {border};"
            f"border-radius:8px;padding:20px 16px;text-align:center;"
            f"font-weight:700;font-size:1.2rem;color:#155724;margin-top:8px'>"
            f"🏆 {champion}</div>", unsafe_allow_html=True)
        st.caption("Campeón según este escenario simulado.")

else:
    st.caption("Todas las rondas en una sola vista — scroll horizontal si es necesario.")
    html = bracket_html(detail["rounds"], played_pairs, highlight, detail["champion"])
    components.html(html, height=940, scrolling=True)

# ---- Camino del equipo resaltado -------------------------------------------
if highlight:
    st.divider()
    st.subheader(f"Camino de {highlight}")

    path_parts = []
    for g_name, g in detail["groups"].items():
        teams_in_group = [t for t, *_ in g["standings"]]
        if highlight in teams_in_group:
            pos = teams_in_group.index(highlight) + 1
            q = g["qualified"]
            if highlight in (q["first"], q["second"]) or highlight == q["third"]:
                suffix = "" if pos <= 2 else " (mejor tercero)"
                path_parts.append(f"{pos}° del Grupo {g_name}{suffix}")
            else:
                path_parts.append(f"{pos}° del Grupo {g_name} — eliminado en grupos")
            break

    eliminated = False
    round_es = {"r32": "R32", "r16": "octavos", "qf": "cuartos",
                "sf": "semifinal", "final": "la final"}
    for rname, _label in ROUND_LABELS:
        for a, b, w in detail["rounds"][rname]:
            if highlight in (a, b):
                rival = b if a == highlight else a
                if w == highlight:
                    path_parts.append(f"vence a {rival} en {round_es[rname]}")
                else:
                    path_parts.append(f"cae ante {rival} en {round_es[rname]}")
                    eliminated = True
        if eliminated:
            break

    if detail["champion"] == highlight:
        path_parts.append("🏆 **CAMPEÓN DEL MUNDO**")

    st.markdown(" → ".join(path_parts))
