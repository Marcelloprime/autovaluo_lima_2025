import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACIÓN PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Valorización de Viviendas — Cercado de Lima",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.0rem; font-weight:700; color:#1a3a5c; margin-bottom:0; }
    .sub-title   { font-size:0.95rem; color:#5a6a7a; margin-top:0; margin-bottom:1.5rem; }
    .card        { background:#f7f9fc; border-radius:12px; padding:1.2rem 1.4rem;
                   border-left:4px solid #2c7be5; margin-bottom:1rem; }
    .metric-card { background:#ffffff; border-radius:10px; padding:1rem 1.2rem;
                   box-shadow:0 2px 8px rgba(0,0,0,0.08); text-align:center; }
    .metric-val  { font-size:1.6rem; font-weight:700; }
    .badge-rf    { background:#d4edda; color:#155724; border-radius:6px; padding:3px 10px; font-weight:600; }
    .badge-xgb   { background:#cce5ff; color:#004085; border-radius:6px; padding:3px 10px; font-weight:600; }
    .badge-mlp   { background:#fff3cd; color:#856404; border-radius:6px; padding:3px 10px; font-weight:600; }
    .winner      { border: 2px solid #28a745 !important; }
    .stAlert     { border-radius:8px; }
    div[data-testid="stMetric"] { background:#f7f9fc; border-radius:10px; padding:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DE ARTEFACTOS (cacheado)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf  = joblib.load("models/rf_model.joblib")
    xgb = joblib.load("models/xgb_model.joblib")
    mlp = joblib.load("models/mlp_model.joblib")
    imp = joblib.load("models/imputer.joblib")
    scl = joblib.load("models/scaler.joblib")
    fc  = joblib.load("models/feature_columns.joblib")
    fi_rf  = joblib.load("models/feat_imp_rf.joblib")
    fi_xgb = joblib.load("models/feat_imp_xgb.joblib")
    with open("models/metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    with open("models/preprocessing_info.json", encoding="utf-8") as f:
        prep = json.load(f)
    return rf, xgb, mlp, imp, scl, fc, fi_rf, fi_xgb, metrics, prep

rf, model_xgb, mlp, imputer, scaler, feature_columns, fi_rf, fi_xgb, metrics, prep = load_models()

# ─────────────────────────────────────────────
# FUNCIÓN DE PREPROCESAMIENTO
# ─────────────────────────────────────────────
def preprocess_input(inputs: dict) -> tuple:
    """Convierte inputs raw → X_imp (para RF/XGB) y X_scaled (para MLP)."""
    hoy = datetime.now()

    fecha_adq = inputs["fecha_adquisicion"]
    fecha_dec = inputs["fecha_declaracion"]
    anio_cons = inputs["anio_construccion"]

    ant_adq  = (hoy - datetime(fecha_adq.year, fecha_adq.month, fecha_adq.day)).days
    ant_dec  = (hoy - datetime(fecha_dec.year, fecha_dec.month, fecha_dec.day)).days
    ant_cons = hoy.year - anio_cons

    area_total = inputs["area_total_construida"]
    area_comun = inputs["area_comun_construida"]
    area_terr  = inputs["area_terreno"]

    pct_area_comun = area_comun / area_total if area_total > 0 else 0
    ratio_c_t      = area_total / area_terr  if area_terr  > 0 else 0

    row = {
        "pct_propiedad":              inputs["pct_propiedad"],
        "area_terreno":               area_terr,
        "area_total_construida":      area_total,
        "pisos":                      inputs["pisos"],
        "Antiguedad_Adquisicion_Dias": ant_adq,
        "Antiguedad_Declaracion_Dias": ant_dec,
        "ant_constru":                ant_cons,
        "pct_area_comun":             pct_area_comun,
        "ratio_construccion_terreno": ratio_c_t,
    }

    # Log transform (mismos shifts usados en entrenamiento)
    log_shifts = prep["log_shifts"]
    for col in prep["numerical_cols"]:
        if col == "autovaluo":
            continue
        if col in row:
            shift = log_shifts.get(col, 0)
            row[col] = np.log1p(row[col] - shift) if shift < 0 else np.log1p(row[col])

    # One-hot encoding manual (mismo drop_first=True del entrenamiento)
    tipo_cats = sorted([v for v in prep["tipo_propietario_values"] if v != "Condómino"])
    for cat in tipo_cats:
        col_name = f"tipo_propietario_{cat}"
        if col_name in feature_columns:
            row[col_name] = 1 if inputs["tipo_propietario"] == cat else 0

    mat_cats = sorted([v for v in prep["material_predio_values"] if v != "Adobe"])
    for cat in mat_cats:
        col_name = f"material_predio_{cat}"
        if col_name in feature_columns:
            row[col_name] = 1 if inputs["material_predio"] == cat else 0

    df_row = pd.DataFrame([{col: row.get(col, 0) for col in feature_columns}])
    X_imp    = pd.DataFrame(imputer.transform(df_row), columns=feature_columns)
    X_scaled = scaler.transform(X_imp)
    return X_imp, X_scaled


def predict_all(X_imp, X_scaled):
    pred_rf  = float(np.expm1(rf.predict(X_imp)[0]))
    pred_xgb = float(np.expm1(model_xgb.predict(X_imp)[0]))
    pred_mlp = float(np.expm1(mlp.predict(X_scaled)[0]))
    return {"Random Forest": pred_rf, "XGBoost": pred_xgb, "Red Neuronal": pred_mlp}


# ─────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Valorizador Lima")
    st.caption("Cercado de Lima · Ene–Mar 2025")
    st.markdown("---")
    page = st.radio(
        "Navegación",
        ["🔮 Predictor", "📊 Métricas de Modelos"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    **Modelos disponibles**
    - 🌲 Random Forest (RF)
    - ⚡ XGBoost (XGB)
    - 🧠 Red Neuronal (MLP)
    """)
    st.caption("Entrenados con 117 K registros · Test 20%")


# ══════════════════════════════════════════════
# PÁGINA 1: PREDICTOR
# ══════════════════════════════════════════════
if page == "🔮 Predictor":
    st.markdown('<p class="main-title">🔮 Predictor de Autovalúo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ingresa las características del predio y compara las predicciones de los tres modelos</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📐 Áreas")
            area_terreno         = st.number_input("Área de terreno (m²)", min_value=1.0, max_value=100000.0, value=80.0, step=1.0)
            area_total_construida = st.number_input("Área total construida (m²)", min_value=1.0, max_value=100000.0, value=70.0, step=1.0)
            area_comun_construida = st.number_input("Área común construida (m²)", min_value=0.0, max_value=500000.0, value=10.0, step=1.0)
            pisos                = st.number_input("Número de pisos", min_value=1, max_value=30, value=30, step=1)

        with col2:
            st.markdown("#### 🏗️ Características")
            material_predio = st.selectbox(
                "Material del predio",
                options=prep["material_predio_values"],
                index=prep["material_predio_values"].index("Concreto") if "Concreto" in prep["material_predio_values"] else 0,
            )
            tipo_propietario = st.selectbox(
                "Tipo de propietario",
                options=prep["tipo_propietario_values"],
            )
            pct_propiedad = st.slider("Porcentaje de propiedad (%)", 0.1, 100.0, 100.0, 0.5)

        with col3:
            st.markdown("#### 📅 Fechas y Antigüedad")
            fecha_adquisicion = st.date_input("Fecha de adquisición", value=datetime(2010, 1, 1), min_value=datetime(1900, 1, 1), max_value=datetime.now())
            fecha_declaracion  = st.date_input("Fecha de declaración",  value=datetime(2015, 6, 1), min_value=datetime(1900, 1, 1), max_value=datetime.now())
            anio_construccion  = st.number_input("Año de construcción", min_value=1900, max_value=datetime.now().year, value=2000, step=1)
            st.markdown("")
            st.markdown("")
            st.markdown("")

        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predecir con los 3 modelos", use_container_width=True, type="primary")

    if submitted:
        inputs = {
            "area_terreno": area_terreno,
            "area_total_construida": area_total_construida,
            "area_comun_construida": area_comun_construida,
            "pisos": pisos,
            "material_predio": material_predio,
            "tipo_propietario": tipo_propietario,
            "pct_propiedad": pct_propiedad,
            "fecha_adquisicion": fecha_adquisicion,
            "fecha_declaracion": fecha_declaracion,
            "anio_construccion": anio_construccion,
        }
        with st.spinner("Calculando predicciones..."):
            X_imp, X_scaled = preprocess_input(inputs)
            predicciones = predict_all(X_imp, X_scaled)

        st.markdown("### 📋 Resultados de Predicción")

        mejor_modelo = min(predicciones, key=lambda k: abs(predicciones[k] - np.mean(list(predicciones.values()))))
        promedio = np.mean(list(predicciones.values()))

        colors = {"Random Forest": "#28a745", "XGBoost": "#007bff", "Red Neuronal": "#ffc107"}
        badges = {
            "Random Forest": "badge-rf",
            "XGBoost":       "badge-xgb",
            "Red Neuronal":  "badge-mlp",
        }
        icons = {"Random Forest": "🌲", "XGBoost": "⚡", "Red Neuronal": "🧠"}

        c1, c2, c3, c4 = st.columns(4)
        cards = [c1, c2, c3]
        for (nombre, valor), card in zip(predicciones.items(), cards):
            with card:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.85rem;color:#666;margin-bottom:4px">{icons[nombre]} {nombre}</div>
                    <div class="metric-val" style="color:{colors[nombre]}">S/ {valor:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card" style="border:2px solid #6f42c1">
                <div style="font-size:0.85rem;color:#666;margin-bottom:4px">📌 Promedio</div>
                <div class="metric-val" style="color:#6f42c1">S/ {promedio:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gráfico de barras comparativo
        col_g, col_t = st.columns([3, 2])

        with col_g:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            names  = list(predicciones.keys())
            values = list(predicciones.values())
            bar_colors = [colors[n] for n in names]

            bars = ax.barh(names, values, color=bar_colors, height=0.5, edgecolor='white', linewidth=1.5)
            ax.axvline(promedio, color='#6f42c1', linestyle='--', linewidth=1.5, label=f'Promedio: S/ {promedio:,.0f}')

            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max(values)*0.005, bar.get_y() + bar.get_height()/2,
                        f"S/ {val:,.0f}", va='center', fontsize=10, fontweight='bold')

            ax.set_xlabel("Autovalúo predicho (S/)", fontsize=10)
            ax.set_title("Comparación de Predicciones", fontsize=12, fontweight='bold', pad=12)
            ax.set_xlim(0, max(values) * 1.22)
            ax.legend(fontsize=9, loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=10)
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_t:
            st.markdown("#### 📝 Tabla Comparativa")
            df_res = pd.DataFrame({
                "Modelo": [f"{icons[n]} {n}" for n in predicciones],
                "Autovalúo (S/)": [f"{v:,.0f}" for v in predicciones.values()],
                "R² (test)": [f"{metrics[n]['R2']:.4f}" for n in predicciones],
            })
            st.dataframe(df_res, hide_index=True, use_container_width=True)

            diff_pct = ((max(predicciones.values()) - min(predicciones.values())) / promedio * 100)
            st.info(f"🔍 Diferencia máx entre modelos: **{diff_pct:.1f}%** respecto al promedio")

            best = min(predicciones, key=lambda k: metrics[k]['RMSE'])
            st.success(f"✅ Mayor precisión histórica: **{best}** (RMSE más bajo)")


# ══════════════════════════════════════════════
# PÁGINA 2: MÉTRICAS
# ══════════════════════════════════════════════
else:
    st.markdown('<p class="main-title">📊 Métricas de Evaluación</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Comparación de los tres modelos entrenados · 20% Test set · 117 K registros (solo viviendas)</p>', unsafe_allow_html=True)

    # ── Tarjetas de resumen ──────────────────
    st.markdown("### 🏆 Resumen de Rendimiento")
    c1, c2, c3 = st.columns(3)

    model_info = {
        "Random Forest": {"icon": "🌲", "color": "#28a745", "card": c1, "badge": "badge-rf"},
        "XGBoost":       {"icon": "⚡", "color": "#007bff", "card": c2, "badge": "badge-xgb"},
        "Red Neuronal":  {"icon": "🧠", "color": "#ffc107", "card": c3, "badge": "badge-mlp"},
    }

    best_r2   = max(metrics, key=lambda k: metrics[k]["R2"])
    best_mae  = min(metrics, key=lambda k: metrics[k]["MAE"])
    best_rmse = min(metrics, key=lambda k: metrics[k]["RMSE"])
    best_time = min(metrics, key=lambda k: metrics[k]["Tiempo_s"])

    for nombre, info in model_info.items():
        m = metrics[nombre]
        badges_won = []
        if nombre == best_r2:   badges_won.append("🥇 Mejor R²")
        if nombre == best_mae:  badges_won.append("🥇 Menor MAE")
        if nombre == best_rmse: badges_won.append("🥇 Menor RMSE")
        if nombre == best_time: badges_won.append("⚡ Más rápido")

        border = "border:2px solid " + info["color"] if badges_won else "border:1px solid #dee2e6"
        badge_html = " ".join([f'<span style="font-size:0.72rem;background:#e8f5e9;border-radius:4px;padding:1px 7px;margin-right:4px">{b}</span>' for b in badges_won])

        with info["card"]:
            st.markdown(f"""
            <div class="metric-card" style="{border};margin-bottom:0.5rem">
                <div style="font-size:1.1rem;font-weight:700;color:{info['color']};margin-bottom:8px">
                    {info['icon']} {nombre}
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
                    <div style="background:#f0f0f0;border-radius:6px;padding:6px">
                        <div style="font-size:0.72rem;color:#666">R²</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#333">{m['R2']:.4f}</div>
                    </div>
                    <div style="background:#f0f0f0;border-radius:6px;padding:6px">
                        <div style="font-size:0.72rem;color:#666">MAE (S/)</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#333">{m['MAE']:,.0f}</div>
                    </div>
                    <div style="background:#f0f0f0;border-radius:6px;padding:6px">
                        <div style="font-size:0.72rem;color:#666">RMSE (S/)</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#333">{m['RMSE']:,.0f}</div>
                    </div>
                    <div style="background:#f0f0f0;border-radius:6px;padding:6px">
                        <div style="font-size:0.72rem;color:#666">Tiempo (s)</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#333">{m['Tiempo_s']:.2f}s</div>
                    </div>
                </div>
                <div>{badge_html}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabla comparativa completa ───────────
    st.markdown("---")
    st.markdown("### 📋 Tabla Comparativa Detallada")

    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics.columns = ["Modelo", "R²", "MAE (S/)", "RMSE (S/)", "Tiempo (s)"]
    df_metrics["R²"]      = df_metrics["R²"].map(lambda x: f"{x:.4f}")
    df_metrics["MAE (S/)"]  = df_metrics["MAE (S/)"].map(lambda x: f"{x:,.2f}")
    df_metrics["RMSE (S/)"] = df_metrics["RMSE (S/)"].map(lambda x: f"{x:,.2f}")
    df_metrics["Tiempo (s)"] = df_metrics["Tiempo (s)"].map(lambda x: f"{x:.2f} s")

    st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    # ── Gráficos comparativos ────────────────
    st.markdown("---")
    st.markdown("### 📈 Visualización de Métricas")

    nombres = list(metrics.keys())
    colores  = ["#28a745", "#007bff", "#ffc107"]
    r2_vals   = [metrics[m]["R2"]      for m in nombres]
    mae_vals  = [metrics[m]["MAE"]     for m in nombres]
    rmse_vals = [metrics[m]["RMSE"]    for m in nombres]
    time_vals = [metrics[m]["Tiempo_s"] for m in nombres]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Comparación de Modelos — Cercado de Lima 2025", fontsize=14, fontweight='bold', y=1.01)

    plot_data = [
        (axes[0,0], r2_vals,   "R² (Mayor es mejor)",        True,  0.8,  1.0),
        (axes[0,1], mae_vals,  "MAE — Error Absoluto (S/)",  False, None, None),
        (axes[1,0], rmse_vals, "RMSE — Error Cuadrático (S/)", False, None, None),
        (axes[1,1], time_vals, "Tiempo de Entrenamiento (s)", False, None, None),
    ]

    for ax, vals, title, higher_better, ymin, ymax in plot_data:
        bars = ax.bar(nombres, vals, color=colores, edgecolor='white', linewidth=1.5, width=0.5)
        best_idx = (vals.index(max(vals)) if higher_better else vals.index(min(vals)))
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:,.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=9)
        if ymin is not None:
            ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Importancia de variables ─────────────
    st.markdown("---")
    st.markdown("### 🔬 Importancia de Variables")

    tab1, tab2 = st.tabs(["🌲 Random Forest", "⚡ XGBoost"])

    def plot_feat_imp(df_fi, color, title):
        top10 = df_fi.head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top10["Variable"], top10["Importancia"], color=color, edgecolor='white', linewidth=1)
        for bar, v in zip(ax.patches, top10["Importancia"]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va='center', fontsize=9)
        ax.set_title(f"Top 10 Variables Más Importantes — {title}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Importancia relativa")
        ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        return fig

    with tab1:
        st.pyplot(plot_feat_imp(fi_rf, "#28a745", "Random Forest"))
        plt.close()
    with tab2:
        st.pyplot(plot_feat_imp(fi_xgb, "#007bff", "XGBoost"))
        plt.close()

    # ── Notas metodológicas ──────────────────
    st.markdown("---")
    with st.expander("ℹ️ Notas metodológicas"):
        st.markdown("""
        **Dataset:** Predios de Cercado de Lima, Enero–Marzo 2025.
        Filtrado a **solo viviendas** con eliminación del top 1% en área y autovalúo (outliers extremos).

        **Preprocesamiento:**
        - Transformación logarítmica (`log1p`) en todas las variables numéricas para normalizar distribuciones sesgadas
        - One-hot encoding para variables categóricas (`tipo_propietario`, `material_predio`, `afecto_imp_predial`)
        - Imputación por mediana para valores nulos
        - Estandarización (StandardScaler) solo para la Red Neuronal

        **Partición:** 80% entrenamiento / 20% test · `random_state=42`

        **Métricas calculadas en escala real** (inverso del log transform) para facilitar interpretación económica.

        | Métrica | Descripción |
        |---------|-------------|
        | **R²** | Proporción de varianza explicada (0–1, mayor es mejor) |
        | **MAE** | Error absoluto promedio en soles peruanos |
        | **RMSE** | Raíz del error cuadrático medio (penaliza errores grandes) |
        | **Tiempo** | Tiempo de entrenamiento en segundos (CPU) |
        """)