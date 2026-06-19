"""
NutriML-Pareja — Sistema predictivo con Machine Learning
Evaluación nutricional preconcepcional en parejas
Predicción de salud en la progenie — Juliaca 2026
Universidad Nacional de Juliaca (UNAJ)
Autor: Condori Ccosi Vidal
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN DE PÁGINA ────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriML-Pareja | UNAJ Juliaca",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS PERSONALIZADO ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .titulo-sistema {
        font-size: 1.8rem; font-weight: 700; color: #1a5276;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitulo {
        font-size: 1rem; color: #5d6d7e;
        text-align: center; margin-bottom: 1.5rem;
    }
    .seccion-madre {
        background: #eaf4fb; border-left: 4px solid #2980b9;
        border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
    }
    .seccion-padre {
        background: #eafaf1; border-left: 4px solid #27ae60;
        border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
    }
    .resultado-alto   { background:#fdedec; border:2px solid #e74c3c;
                        border-radius:10px; padding:1rem; text-align:center; }
    .resultado-medio  { background:#fef9e7; border:2px solid #f39c12;
                        border-radius:10px; padding:1rem; text-align:center; }
    .resultado-bajo   { background:#eafaf1; border:2px solid #27ae60;
                        border-radius:10px; padding:1rem; text-align:center; }
    .metrica-box { background:#f8f9fa; border-radius:8px; padding:0.7rem;
                   text-align:center; border:1px solid #dee2e6; }
    .altitud-nota { background:#fff3cd; border:1px solid #ffc107;
                    border-radius:6px; padding:0.5rem; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTE ALTITUD ──────────────────────────────────────────────────────
FACTOR_OPS = 2.0  # g/dL — corrección OPS para Juliaca 3824 m s.n.m.

# ── CARGAR MODELOS ─────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    modelos = joblib.load('modelos.pkl')
    prepro  = joblib.load('preprocesador.pkl')
    return modelos, prepro

modelos, prepro = cargar_modelos()

NOMBRES_ALGORITMOS = list(modelos.keys())
COLORES_ALGO = {
    'Árbol de Decisión':   '#95a5a6',
    'Regresión Logística': '#3498db',
    'SVM':                 '#1abc9c',
    'Random Forest':       '#9b59b6',
    'XGBoost':             '#e67e22',
}

FEATURES = [
    'imc_materno','hb_materna_ajust','ferritina','acido_folico','zinc',
    'ganancia_peso','edad_materna','paridad',
    'imc_paterno','hb_paterna_ajust','edad_paterna','divers_dietetica',
]
FEATURES_ES = [
    'IMC materno (kg/m²)','Hb materna ajustada (g/dL)','Ferritina (μg/L)',
    'Ácido fólico (nmol/L)','Zinc sérico (μg/dL)','Ganancia de peso (kg)',
    'Edad materna (años)','Paridad',
    'IMC paterno (kg/m²)','Hb paterna ajustada (g/dL)',
    'Edad paterna (años)','Diversidad dietética paterna (0-10)',
]

CLASES = ['Bajo peso al nacer (BPN)', 'Peso adecuado', 'Macrosomía']
EMOJIS = ['⚠️', '✅', '⚠️']
COLORES_CLASE = ['#e74c3c', '#27ae60', '#f39c12']

# ── ENCABEZADO ─────────────────────────────────────────────────────────────
st.markdown('<p class="titulo-sistema">🤰 NutriML-Pareja</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo">Sistema predictivo con Machine Learning · '
    'Evaluación nutricional preconcepcional en parejas · '
    'Universidad Nacional de Juliaca — 2026</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="altitud-nota">🏔️ <b>Juliaca, 3 824 m s. n. m.</b> — '
    'Los valores de hemoglobina se corrigen automáticamente con el '
    'factor de ajuste OPS/OMS (−2,0 g/dL) para esta altitud.</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── SIDEBAR — SELECCIÓN DE ALGORITMO ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    algoritmo = st.selectbox(
        "Algoritmo ML:",
        NOMBRES_ALGORITMOS,
        index=2,  # SVM por defecto (mejor modelo)
        help="SVM obtuvo el mejor F1-Score (0.736) en el pipeline ejecutado"
    )
    st.markdown(f"""
    **Métricas del modelo seleccionado:**
    | Métrica | Valor |
    |---------|-------|
    | F1-Score | 0.736 |
    | Accuracy | 76.5% |
    | AUC-ROC | 0.543 |
    """)
    st.markdown("---")
    st.markdown("**Acerca del sistema:**")
    st.caption(
        "NutriML-Pareja predice el peso al nacer de la progenie "
        "a partir del estado nutricional preconcepcional de ambos "
        "miembros de la pareja. Desarrollado como tesis de pregrado "
        "en Ingeniería de Software y Sistemas — UNAJ."
    )
    st.caption("⚠️ Herramienta de apoyo diagnóstico. No reemplaza el criterio clínico.")

# ── FORMULARIO DIVIDIDO EN 3 COLUMNAS ─────────────────────────────────────
st.subheader("📋 Datos de la pareja preconcepcional")

col1, col2, col3 = st.columns([1, 1, 1])

# — COLUMNA 1: Variables maternas antropométricas —
with col1:
    st.markdown('<div class="seccion-madre">', unsafe_allow_html=True)
    st.markdown("**🩺 Madre — Antropometría**")

    peso_m = st.number_input("Peso materno (kg)", 35.0, 120.0, 65.0, 0.1)
    talla_m = st.number_input("Talla materna (cm)", 140.0, 185.0, 158.0, 0.5)
    imc_materno = round(peso_m / (talla_m/100)**2, 1)
    st.metric("IMC materno calculado", f"{imc_materno} kg/m²",
              delta="Normal" if 18.5<=imc_materno<25 else
                    "Sobrepeso" if 25<=imc_materno<30 else
                    "Obesidad" if imc_materno>=30 else "Delgadez")

    ganancia_peso = st.slider("Ganancia de peso pregestacional (kg)", 0.0, 25.0, 10.9, 0.1)
    edad_materna  = st.number_input("Edad materna (años)", 18, 35, 27)
    paridad       = st.selectbox("Paridad (embarazos previos)", [0, 1, 2, 3],
                                  format_func=lambda x: f"{x} {'(nulípara)' if x==0 else 'previo(s)'}")
    st.markdown('</div>', unsafe_allow_html=True)

# — COLUMNA 2: Variables maternas bioquímicas + paternas —
with col2:
    st.markdown('<div class="seccion-madre">', unsafe_allow_html=True)
    st.markdown("**🔬 Madre — Bioquímica**")

    hb_materna_medida = st.number_input("Hemoglobina materna medida (g/dL)", 7.0, 18.0, 14.4, 0.1,
                                         help="Valor del laboratorio, sin ajuste por altitud")
    hb_materna_ajust  = round(hb_materna_medida - FACTOR_OPS, 1)
    anemia_m = hb_materna_medida < (12.0 + FACTOR_OPS)
    st.metric("Hb ajustada por altitud", f"{hb_materna_ajust} g/dL",
              delta="⚠️ Anemia" if anemia_m else "Normal",
              delta_color="inverse" if anemia_m else "normal")

    ferritina  = st.number_input("Ferritina sérica (μg/L)", 2.0, 100.0, 22.3, 0.5)
    acido_folico = st.number_input("Ácido fólico eritrocitario (nmol/L)", 100.0, 800.0, 348.6, 1.0)
    zinc       = st.number_input("Zinc sérico (μg/dL)", 30.0, 130.0, 68.4, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

# — COLUMNA 3: Variables paternas —
with col3:
    st.markdown('<div class="seccion-padre">', unsafe_allow_html=True)
    st.markdown("**👨 Padre — Variables preconcepcionales**")

    peso_p  = st.number_input("Peso paterno (kg)", 45.0, 130.0, 72.0, 0.1)
    talla_p = st.number_input("Talla paterna (cm)", 150.0, 195.0, 168.0, 0.5)
    imc_paterno = round(peso_p / (talla_p/100)**2, 1)
    st.metric("IMC paterno calculado", f"{imc_paterno} kg/m²",
              delta="Normal" if 18.5<=imc_paterno<25 else
                    "Sobrepeso" if 25<=imc_paterno<30 else
                    "Obesidad" if imc_paterno>=30 else "Delgadez")

    hb_paterna_medida = st.number_input("Hemoglobina paterna medida (g/dL)", 9.0, 20.0, 16.2, 0.1)
    hb_paterna_ajust  = round(hb_paterna_medida - FACTOR_OPS, 1)
    anemia_p = hb_paterna_medida < (13.0 + FACTOR_OPS)
    st.metric("Hb paterna ajustada", f"{hb_paterna_ajust} g/dL",
              delta="⚠️ Anemia" if anemia_p else "Normal",
              delta_color="inverse" if anemia_p else "normal")

    edad_paterna  = st.number_input("Edad paterna (años)", 18, 50, 30)
    divers_dietetica = st.slider("Diversidad dietética paterna (0–10)", 0.0, 10.0, 6.2, 0.1,
                                  help="0=dieta muy pobre, 10=dieta muy diversa")
    st.markdown('</div>', unsafe_allow_html=True)

# ── BOTÓN DE PREDICCIÓN ─────────────────────────────────────────────────────
st.markdown("---")
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predecir = st.button("🔍 Predecir salud en la progenie",
                          use_container_width=True, type="primary")

# ── PREDICCIÓN Y RESULTADOS ─────────────────────────────────────────────────
if predecir:
    # Vector de entrada
    X_input = np.array([[
        imc_materno, hb_materna_ajust, ferritina, acido_folico, zinc,
        ganancia_peso, float(edad_materna), float(paridad),
        imc_paterno, hb_paterna_ajust, float(edad_paterna), divers_dietetica
    ]])

    X_pp = prepro.transform(X_input)
    modelo_sel = modelos[algoritmo]
    pred_clase = modelo_sel.predict(X_pp)[0]
    pred_proba = modelo_sel.predict_proba(X_pp)[0]

    st.markdown("## 📊 Resultados de la predicción")

    # — RESULTADO PRINCIPAL —
    clase_nombre = CLASES[pred_clase]
    prob_pred    = pred_proba[pred_clase] * 100
    emoji        = EMOJIS[pred_clase]
    color_div    = ['resultado-alto', 'resultado-bajo', 'resultado-medio'][pred_clase]

    st.markdown(f"""
    <div class="{color_div}">
        <h2>{emoji} {clase_nombre}</h2>
        <p style="font-size:1.3rem;">Probabilidad: <b>{prob_pred:.1f}%</b></p>
        <p style="color:#5d6d7e;">Algoritmo: {algoritmo}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # — PROBABILIDADES POR CLASE —
    col_p1, col_p2, col_p3 = st.columns(3)
    for col, clase, prob, color in zip(
        [col_p1, col_p2, col_p3], CLASES, pred_proba, COLORES_CLASE
    ):
        with col:
            st.markdown(f"""
            <div class="metrica-box">
                <div style="font-size:1.5rem;font-weight:700;color:{color}">
                    {prob*100:.1f}%
                </div>
                <div style="font-size:0.85rem;color:#5d6d7e;">{clase}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # — DOS COLUMNAS: Gráfico probabilidades + SHAP —
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("📈 Probabilidades por clase")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        bars = ax1.barh(CLASES, pred_proba*100,
                        color=COLORES_CLASE, edgecolor='white',
                        linewidth=0.5, height=0.5)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel("Probabilidad (%)", fontsize=10)
        for bar, val in zip(bars, pred_proba*100):
            ax1.text(val+1, bar.get_y()+bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontsize=10)
        ax1.axvline(50, color='gray', linestyle='--', lw=0.8, alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col_g2:
        st.subheader("🔎 Importancia de variables (SHAP)")
        try:
            explainer   = shap.TreeExplainer(modelos['XGBoost'])
            shap_vals   = explainer.shap_values(X_pp)
            sm = shap_vals if isinstance(shap_vals, list) else \
                 [shap_vals[:,:,i] for i in range(shap_vals.shape[2])]
            sv_clase = sm[pred_clase][0]
            idx_ord  = np.argsort(np.abs(sv_clase))

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            colores_shap = ['#e74c3c' if 'patern' in FEATURES_ES[i].lower()
                            else '#3498db' for i in idx_ord]
            ax2.barh([FEATURES_ES[i] for i in idx_ord],
                     sv_clase[idx_ord],
                     color=colores_shap, edgecolor='white',
                     linewidth=0.4, height=0.6)
            ax2.axvline(0, color='black', lw=0.8)
            ax2.set_xlabel("Valor SHAP (impacto en predicción)", fontsize=9)
            ax2.tick_params(axis='y', labelsize=7.5)
            ax2.legend(handles=[
                mpatches.Patch(color='#3498db', label='Variable materna'),
                mpatches.Patch(color='#e74c3c', label='Variable paterna')
            ], fontsize=8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        except Exception:
            st.info("SHAP disponible solo para XGBoost. Selecciona XGBoost en el sidebar para ver este gráfico.")

    st.markdown("---")

    # — COMPARACIÓN TODOS LOS ALGORITMOS —
    st.subheader("⚖️ Comparación entre todos los algoritmos")
    comp_data = []
    for n, m in modelos.items():
        p = m.predict(X_pp)[0]
        pr = m.predict_proba(X_pp)[0]
        comp_data.append({
            'Algoritmo': n,
            'Predicción': CLASES[p],
            'Prob. BPN': f"{pr[0]*100:.1f}%",
            'Prob. Adecuado': f"{pr[1]*100:.1f}%",
            'Prob. Macrosomía': f"{pr[2]*100:.1f}%",
            'Seleccionado': '✓' if n == algoritmo else ''
        })
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp.set_index('Algoritmo'), use_container_width=True)

    # — RESUMEN CLÍNICO —
    st.markdown("---")
    st.subheader("📋 Resumen clínico para historia clínica")

    cat_imc_m = ("Delgadez" if imc_materno<18.5 else "Normal" if imc_materno<25
                 else "Sobrepeso" if imc_materno<30 else "Obesidad")
    cat_imc_p = ("Delgadez" if imc_paterno<18.5 else "Normal" if imc_paterno<25
                 else "Sobrepeso" if imc_paterno<30 else "Obesidad")

    st.markdown(f"""
    | Parámetro | Madre | Padre |
    |-----------|-------|-------|
    | IMC | {imc_materno} kg/m² ({cat_imc_m}) | {imc_paterno} kg/m² ({cat_imc_p}) |
    | Hb ajustada (OPS Juliaca) | {hb_materna_ajust} g/dL {'⚠️ Anemia' if anemia_m else '✅'} | {hb_paterna_ajust} g/dL {'⚠️ Anemia' if anemia_p else '✅'} |
    | Edad | {edad_materna} años | {edad_paterna} años |

    **Predicción NutriML-Pareja ({algoritmo}):**
    - Outcome neonatal esperado: **{clase_nombre}** (prob. {prob_pred:.1f}%)
    - Ganancia de peso pregestacional: {ganancia_peso} kg
    - Ferritina materna: {ferritina} μg/L {'⚠️ Deficiencia' if ferritina<15 else '✅'}
    - Ácido fólico: {acido_folico} nmol/L {'⚠️ Deficiencia' if acido_folico<305 else '✅'}

    *Sistema NutriML-Pareja · UNAJ Juliaca 2026 · Corrección Hb: −{FACTOR_OPS} g/dL (OPS, 3824 m s.n.m.)*
    """)

    st.caption("⚠️ Este resultado es de apoyo diagnóstico. La decisión clínica final corresponde al profesional de salud tratante.")
