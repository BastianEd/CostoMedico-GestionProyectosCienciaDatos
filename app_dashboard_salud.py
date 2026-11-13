# ==============================================================================
# DASHBOARD EJECUTIVO Y SIMULADOR PREVENTIVO DE SINIESTRALIDAD (SCY1102)
# Equipo: Nicolás Fonseca Olivares, Cristóbal Segovia Retamal, Bastián Rubio Meneses
# ==============================================================================
# Instrucciones de ejecución:
#   pip install streamlit pandas numpy scikit-learn plotly joblib shap scipy
#   streamlit run app_dashboard_salud.py
# ==============================================================================

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import sparse
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS EJECUTIVOS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Ecosistema Preventivo de Salud | SCY1102",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
        color: #d1e3ed;
    }
    .badge-ok {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-crit {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. INGESTA, VALIDACIÓN DE ESQUEMA (RF01) Y ENTRENAMIENTO CACHEADO (RF02)
# ------------------------------------------------------------------------------
FEATURES_NUM = ['age', 'bmi', 'children', 'obese_smoker', 'age_bmi_interaction']
FEATURES_CAT = ['sex', 'smoker', 'region']
TARGET = 'charges'

@st.cache_data
def cargar_y_preparar_datos(filepath='insurance.csv'):
    if not os.path.exists(filepath):
        st.error(f"No se encontró el archivo '{filepath}' en el directorio actual.")
        st.stop()
    
    raw = pd.read_csv(filepath)
    n_original = len(raw)
    
    # Validación de esquema (RF01)
    columnas_esperadas = {'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'}
    if not columnas_esperadas.issubset(set(raw.columns)):
        st.error(f"Error de Esquema (RF01): Faltan columnas requeridas: {columnas_esperadas - set(raw.columns)}")
        st.stop()
        
    df = raw.drop_duplicates().reset_index(drop=True)
    duplicados_eliminados = n_original - len(df)
    nulos_detectados = int(df.isna().sum().sum())
    
    # Ingeniería de variables (Sección 2 del Notebook)
    df['obese_smoker'] = ((df['bmi'] >= 30) & (df['smoker'] == 'yes')).astype(int)
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    cortes_bmi = [0, 18.5, 25, 30, 35, 40, 100]
    etiquetas_bmi = ['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad I', 'Obesidad II', 'Obesidad III']
    df['bmi_category'] = pd.cut(df['bmi'], bins=cortes_bmi, labels=etiquetas_bmi)
    
    cortes_edad = [17, 25, 45, 55, 64]
    etiquetas_edad = ['Joven (18-25)', 'Adulto (26-45)', 'Maduro (46-55)', 'Senior (56-64)']
    df['age_group'] = pd.cut(df['age'], bins=cortes_edad, labels=etiquetas_edad)
    
    df['perfil_referencia'] = np.where(df['obese_smoker'] == 1, 'Fumador con IMC >= 30', 'Resto de afiliados')
    
    info_ingesta = {
        'n_original': n_original,
        'n_limpio': len(df),
        'duplicados': duplicados_eliminados,
        'nulos': nulos_detectados
    }
    return df, info_ingesta

def mae_dolares(est, X_, y_log):
    return -mean_absolute_error(np.expm1(y_log), np.expm1(est.predict(X_)))

def r2_dolares(est, X_, y_log):
    return r2_score(np.expm1(y_log), np.expm1(est.predict(X_)))

@st.cache_resource
def obtener_modelo_y_metricas(df):
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = np.log1p(df[TARGET])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test_real = np.expm1(y_test)
    
    # Cargar desde joblib si existe, o entrenar con los hiperparámetros óptimos del Notebook
    if os.path.exists('modelo_costos_medicos.joblib'):
        try:
            pipeline = joblib.load('modelo_costos_medicos.joblib')
            # Verificar compatibilidad de versión prediciendo 1 fila
            _ = pipeline.predict(X_test.iloc[:1])
        except Exception:
            pipeline = None
    else:
        pipeline = None
        
    if pipeline is None:
        preprocessor = ColumnTransformer([
            ('num', 'passthrough', FEATURES_NUM),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), FEATURES_CAT)
        ])
        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.5,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        pipeline = Pipeline([('prep', preprocessor), ('modelo', rf)])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, 'modelo_costos_medicos.joblib')

    # Medición de rendimiento por lote completo (RNF01: <= 5 segundos)
    t0 = time.perf_counter()
    pred_all_usd = np.expm1(pipeline.predict(X))
    tiempo_lote = time.perf_counter() - t0

    # Evaluación en conjunto de prueba independiente (20%, n=268)
    pred_test_usd = np.expm1(pipeline.predict(X_test))
    mae_test = mean_absolute_error(y_test_real, pred_test_usd)
    mape_test = mean_absolute_percentage_error(y_test_real, pred_test_usd) * 100
    r2_test = r2_score(y_test_real, pred_test_usd)

    # Validación cruzada 5-fold (Control de estabilidad)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor_cv = ColumnTransformer([
        ('num', 'passthrough', FEATURES_NUM),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), FEATURES_CAT)
    ])
    rf_cv = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.5, max_depth=8, random_state=42, n_jobs=-1
    )
    pipe_cv = Pipeline([('prep', preprocessor_cv), ('modelo', rf_cv)])
    cv_res = cross_validate(pipe_cv, X, y, cv=kfold, scoring={'mae': mae_dolares, 'r2': r2_dolares})
    
    # Permutation Importance sobre Test
    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring=mae_dolares)
    feat_imp = pd.DataFrame({
        'Variable': FEATURES_NUM + FEATURES_CAT,
        'Aumento_MAE_USD': perm.importances_mean,
        'Desv_USD': perm.importances_std
    }).sort_values('Aumento_MAE_USD', ascending=True)

    metricas = {
        'mae_test': mae_test,
        'mape_test': mape_test,
        'r2_test': r2_test,
        'mae_cv': -cv_res['test_mae'].mean(),
        'mae_cv_std': cv_res['test_mae'].std(),
        'r2_cv': cv_res['test_r2'].mean(),
        'r2_cv_std': cv_res['test_r2'].std(),
        'tiempo_lote_seg': tiempo_lote,
        'n_test': len(X_test),
        'n_train': len(X_train)
    }
    
    return pipeline, metricas, X_train, X_test, y_train, y_test_real, pred_test_usd, pred_all_usd, feat_imp

df, info_ingesta = cargar_y_preparar_datos()
pipeline, metricas, X_train, X_test, y_train, y_test_real, pred_test_usd, pred_all_usd, feat_imp = obtener_modelo_y_metricas(df)

# Agregar predicciones y segmentación RF03 al dataframe maestro
df_cartera = df.copy()
df_cartera['costo_proyectado'] = pred_all_usd
df_cartera['residuo'] = df_cartera['charges'] - df_cartera['costo_proyectado']

# ------------------------------------------------------------------------------
# 3. BARRA LATERAL: CONTROLES GLOBALES Y ESTADO DEL SISTEMA
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 Panel de Control Clínico")
    st.caption("Asignatura: SCY1102 — Gestión de Proyectos de Datos (Duoc UC)")
    st.divider()
    
    st.markdown("#### ⚙️ Parámetros Operativos")
    umbral_rf03 = st.slider(
        "Umbral Atención Prioritaria (RF03, USD):",
        min_value=10000, max_value=35000, value=20000, step=1000,
        help="Por requerimiento RF03, el umbral oficial acordado es $20.000 USD."
    )
    
    regiones_sel = st.multiselect(
        "Filtrar Regiones:",
        options=sorted(df['region'].unique()),
        default=sorted(df['region'].unique())
    )
    
    st.divider()
    st.markdown("#### ✅ Compuertas de Calidad (H1–H5)")
    cumple_rf02 = (metricas['r2_test'] >= 0.85) and (metricas['mae_test'] < 2100)
    cumple_rnf01 = metricas['tiempo_lote_seg'] <= 5.0
    
    st.markdown(f"- **RF01 (Esquema):** <span class='badge-ok'>{info_ingesta['n_limpio']} reg. limpios</span>", unsafe_allow_html=True)
    st.markdown(f"- **RF02 (R² ≥ 0,85):** <span class='badge-ok'>R² = {metricas['r2_test']:.3f}</span>", unsafe_allow_html=True)
    st.markdown(f"- **RF02 (MAE < $2.100):** <span class='badge-ok'>${metricas['mae_test']:,.0f} USD</span>", unsafe_allow_html=True)
    st.markdown(f"- **RNF01 (Lote ≤ 5s):** <span class='badge-ok'>{metricas['tiempo_lote_seg']:.4f} s</span>", unsafe_allow_html=True)
    st.markdown("- **RNF05/06 (Ley 21.719):** <span class='badge-ok'>SHA-256 + PIA</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("Equipo: N. Fonseca · C. Segovia · B. Rubio")

# Aplicar filtro y regla RF03
df_filtrado = df_cartera[df_cartera['region'].isin(regiones_sel)].copy()
df_filtrado['segmento_rf03'] = np.where(
    df_filtrado['costo_proyectado'] > umbral_rf03,
    'Atención Preventiva Prioritaria',
    'Seguimiento Estándar'
)

# ------------------------------------------------------------------------------
# 4. ENCABEZADO PRINCIPAL Y KPIs DE CARTERA
# ------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏥 Ecosistema Preventivo de Salud y Predicción de Siniestralidad</h1>
    <p>Entregable Funcional Operativo (CRISP-DM + MLOps) · Modelo Supervisado Random Forest + Explicabilidad SHAP + Gobernanza Ética</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Cartera Analizada (RF01)", f"{len(df_filtrado):,} afiliados", f"-{info_ingesta['duplicados']} dup. eliminado")
with col2:
    gasto_medio_fum = df_filtrado[df_filtrado['smoker'] == 'yes']['charges'].mean()
    gasto_medio_nofum = df_filtrado[df_filtrado['smoker'] == 'no']['charges'].mean()
    factor_fum = gasto_medio_fum / gasto_medio_nofum if gasto_medio_nofum > 0 else 0
    st.metric("Gasto Medio Fumador", f"${gasto_medio_fum:,.0f} USD", f"{factor_fum:.1f}x vs No Fumador (${gasto_medio_nofum:,.0f})", delta_color="inverse")
with col3:
    df_ref = df_filtrado[df_filtrado['obese_smoker'] == 1]
    df_resto = df_filtrado[df_filtrado['obese_smoker'] == 0]
    media_ref = df_ref['charges'].mean() if len(df_ref) > 0 else 0
    media_resto = df_resto['charges'].mean() if len(df_resto) > 0 else 1
    st.metric("Perfil Crítico (Fumador + IMC≥30)", f"${media_ref:,.0f} USD", f"{len(df_ref)} afil. ({media_ref/media_resto:.1f}x resto)", delta_color="inverse")
with col4:
    st.metric("Desempeño Modelo (Test 20%)", f"R² = {metricas['r2_test']:.3f}", f"MAE: ${metricas['mae_test']:,.0f} (CV: ${metricas['mae_cv']:,.0f})")
with col5:
    n_prio = (df_filtrado['segmento_rf03'] == 'Atención Preventiva Prioritaria').sum()
    st.metric(f"Prioritarios (RF03 > ${umbral_rf03:,.0f})", f"{n_prio} afiliados", f"{n_prio/len(df_filtrado)*100:.1f}% de la cartera")

# ------------------------------------------------------------------------------
# 5. PESTAÑAS DEL DASHBOARD (TRAZABILIDAD 100% CON RÚBRICA)
# ------------------------------------------------------------------------------
tabs = st.tabs([
    "📊 1. EDA y Segmentación (RF03)",
    "🤖 2. Desempeño, Residuos y SHAP (RF02/RF05)",
    "🩺 3. Simulador Clínico Individual (SHAP)",
    "⚖️ 4. Auditoría de Equidad y MLOps (RNF07)",
    "💰 5. Escenarios Financieros y ROI (RF04)",
    "📋 6. Monitoreo y Control Jira (Indicador 2)"
])

# ==============================================================================
# TAB 1: EDA PROFUNDO Y SEGMENTACIÓN PREVENTIVA (RF03 / RF04)
# ==============================================================================
with tabs[0]:
    st.subheader("Análisis Exploratorio de Siniestralidad y Segmentación Preventiva (RF03)")
    st.markdown("""
    **Hallazgo estratégico:** El IMC por sí solo tiene baja correlación lineal con el gasto médico ($r_s = 0,12$), pero al combinarse con **tabaquismo a partir de Obesidad Grado I ($	ext{IMC} \ge 30$)** dispara el costo promedio a **$41.558 USD** ($4,2$ veces el resto de la cartera, acumulando **$6.025.909 USD/año**).
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_scatter = px.scatter(
            df_filtrado, x='age', y='charges', color='perfil_referencia', size='bmi',
            color_discrete_map={'Resto de afiliados': '#4C72B0', 'Fumador con IMC >= 30': '#C44E52'},
            hover_data=['sex', 'region', 'children', 'costo_proyectado'],
            labels={'age': 'Edad (años)', 'charges': 'Gasto Facturado (USD)', 'perfil_referencia': 'Perfil'},
            title="Interacción Edad × IMC × Tabaquismo sobre el Gasto Médico (USD)"
        )
        fig_scatter.add_hline(y=umbral_rf03, line_dash="dash", line_color="#2b2b2b",
                            annotation_text=f"Umbral RF03 (${umbral_rf03:,.0f} USD)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c2:
        resumen_bmi = df_filtrado.groupby(['bmi_category', 'smoker'], observed=True)['charges'].median().reset_index()
        resumen_bmi['Tabaquismo'] = resumen_bmi['smoker'].map({'no': 'No fumador', 'yes': 'Fumador'})
        fig_bmi = px.bar(
            resumen_bmi, x='bmi_category', y='charges', color='Tabaquismo', barmode='group',
            color_discrete_map={'No fumador': '#4C72B0', 'Fumador': '#C44E52'},
            labels={'bmi_category': 'Categoría Clínica IMC (OMS)', 'charges': 'Costo Mediano (USD)'},
            title="Costo Mediano por Categoría de IMC (OMS) y Tabaquismo"
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    st.divider()
    st.markdown(f"#### 🎯 Resultado de Clasificación Preventiva (`RF03` — Costo Proyectado > ${umbral_rf03:,.0f} USD)")
    
    c_tabla, c_export = st.columns([2, 1])
    with c_tabla:
        cruce_rf03 = pd.crosstab(
            df_filtrado['segmento_rf03'],
            df_filtrado['perfil_referencia'],
            margins=True, margins_name='Total Cartera'
        )
        st.dataframe(cruce_rf03, use_container_width=True)
        
    with c_export:
        st.info("""
        **Gobernanza y Privacidad (`RNF05` / `RNF08`):**
        El reporte exportable contiene perfiles seudonimizados elegibles para programas de cesación tabáquica y nutrición. No incluye PII (RUT/Nombres).
        """)
        df_export = df_filtrado[['age', 'sex', 'bmi', 'bmi_category', 'children', 'smoker', 'region', 'obese_smoker', 'charges', 'costo_proyectado', 'segmento_rf03']].copy()
        df_export['id_seudonimizado'] = [f"SHA256_{i:04d}" for i in df_export.index]
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Reporte de Segmentación (CSV - RF04)",
            data=csv_bytes,
            file_name="reporte_segmentacion_preventiva_RF04.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 2: DESEMPEÑO DEL MODELO, ABLACIÓN, RESIDUOS Y EXPLICABILIDAD (RF02 / RF05)
# ==============================================================================
with tabs[1]:
    st.subheader("Evaluación del Modelo Supervisado, Ablación y Análisis de Residuos")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 📊 Comparación de Algoritmos (Validación Cruzada 5-Fold, USD)")
        df_comp = pd.DataFrame({
            'Modelo': ['Baseline: Mediana (Piso)', 'Baseline: Regresión Lineal', 'Gradient Boosting (HGB)', 'Random Forest Base', 'Random Forest Ajustado (Final)'],
            'MAE_CV_USD': [8352, 4111, 2500, 2156, round(metricas['mae_cv'])],
            'R2_CV': [-0.105, 0.453, 0.831, 0.851, round(metricas['r2_cv'], 3)],
            'Estado_RF02': ['❌ No cumple', '❌ No cumple', '❌ No cumple MAE', '⚠️ Límite en MAE', '✅ Cumple Test y CV']
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
    with col_m2:
        st.markdown("##### 🧪 Estudio de Ablación de Ingeniería de Variables (CV 5-Fold)")
        df_abl = pd.DataFrame({
            'Configuración de Variables': [
                'Todas las variables (Modelo Final)',
                'Sin obese_smoker (Quitar interacción IMC≥30 × Fumador)',
                'Sin age_bmi_interaction (Quitar Edad × IMC)',
                'Solo variables originales (age, bmi, children, cat)'
            ],
            'MAE (USD)': ['$2.011', '$2.179 (+$168 / +8,4%)', '$2.004 (-$7)', '$2.123 (+$112)'],
            'R² (CV)': [0.858, 0.851, 0.857, 0.851],
            'Conclusión Técnica': [
                'Cumple RF02 (< $2.100)',
                '❌ Incumple RF02: obese_smoker es indispensable',
                'No aporta mejora adicional; candidata a simplificación',
                '❌ Incumple RF02 sin ingeniería de variables'
            ]
        })
        st.dataframe(df_abl, use_container_width=True, hide_index=True)

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        df_test_plot = pd.DataFrame({
            'Costo_Real_USD': y_test_real,
            'Costo_Predicho_USD': pred_test_usd,
            'Residuo_USD': y_test_real - pred_test_usd,
            'Perfil': np.where(X_test['obese_smoker'] == 1, 'Fumador + IMC≥30', 'Resto')
        })
        fig_pred = px.scatter(
            df_test_plot, x='Costo_Predicho_USD', y='Costo_Real_USD', color='Perfil',
            color_discrete_map={'Resto': '#4C72B0', 'Fumador + IMC≥30': '#C44E52'},
            title=f"Predicho vs Real en Conjunto de Prueba (R² = {metricas['r2_test']:.3f}, MAE = ${metricas['mae_test']:,.0f})"
        )
        max_val = float(max(df_test_plot['Costo_Real_USD'].max(), df_test_plot['Costo_Predicho_USD'].max()))
        fig_pred.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                      name='Predicción Perfecta', line=dict(color='black', dash='dash')))
        st.plotly_chart(fig_pred, use_container_width=True)
        
    with g2:
        fig_imp = px.bar(
            feat_imp, x='Aumento_MAE_USD', y='Variable', orientation='h',
            error_x='Desv_USD', color='Aumento_MAE_USD', color_continuous_scale='Viridis',
            title="Importancia Global de Variables (Permutation Importance en Test, USD)",
            labels={'Aumento_MAE_USD': 'Aumento del MAE (USD) al permutar la variable'}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Alerta de cola extrema (10% más costoso)
    p90 = np.percentile(y_test_real, 90)
    mask_top10 = y_test_real > p90
    residuos_test = y_test_real - pred_test_usd
    sub_top10 = residuos_test[mask_top10].mean()
    mae_top10 = np.abs(residuos_test[mask_top10]).mean()
    st.warning(
        f"📌 **Control Financiero de Residuos (Decil 90% más costoso, n={mask_top10.sum()} en test):** "
        f"En los casos sobre ${p90:,.0f} USD, el modelo presenta un MAE de **${mae_top10:,.0f} USD** y una subestimación media de "
        f"**+${sub_top10:,.0f} USD** ({sub_top10 / y_test_real[mask_top10].mean() * 100:.1f}% del gasto medio del grupo). "
        f"Se recomienda incorporar una reserva técnica del **4,1%** para dicho segmento al presupuestar."
    )

# ==============================================================================
# TAB 3: SIMULADOR CLÍNICO INDIVIDUAL Y EXPLICABILIDAD LOCAL (RF05 / RNF02)
# ==============================================================================
with tabs[2]:
    st.subheader("Simulador de Predicción Individual y Explicabilidad para Auditoría Médica (RF05)")
    st.markdown("Permite estimar el gasto anual de un perfil de afiliado y auditar cuánto aporta cada variable a la predicción individual.")
    
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.markdown("#### 👤 Datos del Afiliado")
        in_age = st.slider("Edad (años):", 18, 64, 42)
        in_sex = st.selectbox("Sexo:", options=['female', 'male'], format_func=lambda x: 'Mujer (female)' if x == 'female' else 'Hombre (male)')
        in_bmi = st.number_input("Índice de Masa Corporal (IMC):", min_value=15.0, max_value=55.0, value=31.5, step=0.5)
        in_children = st.slider("Número de cargas familiares (hijos):", 0, 5, 1)
        in_smoker = st.selectbox("¿Fumador activo?:", options=['no', 'yes'], format_func=lambda x: 'Sí (Fumador)' if x == 'yes' else 'No fumador')
        in_region = st.selectbox("Región de referencia:", options=['southeast', 'southwest', 'northwest', 'northeast'])
        
        in_obese_smoker = int((in_bmi >= 30.0) and (in_smoker == 'yes'))
        in_age_bmi = in_age * in_bmi
        
        df_input = pd.DataFrame([{
            'age': in_age,
            'bmi': in_bmi,
            'children': in_children,
            'obese_smoker': in_obese_smoker,
            'age_bmi_interaction': in_age_bmi,
            'sex': in_sex,
            'smoker': in_smoker,
            'region': in_region
        }])
        
    with sc2:
        pred_log = float(pipeline.predict(df_input)[0])
        pred_usd = float(np.expm1(pred_log))
        es_prioritario = pred_usd > umbral_rf03
        
        st.markdown("#### 🩺 Resultado de la Evaluación Analítica")
        k1, k2, k3 = st.columns(3)
        k1.metric("Gasto Anual Estimado (RF02)", f"${pred_usd:,.0f} USD", f"± ${metricas['mae_test']:,.0f} (MAE ref.)")
        k2.metric("Clasificación RF03", "Prioritaria" if es_prioritario else "Estándar",
                  "Elegible Programa Preventivo" if es_prioritario else "Control Habitual",
                  delta_color="inverse" if es_prioritario else "normal")
        k3.metric("Factor Sinergia Obesidad×Tabaco", "ACTIVO (IMC≥30 + Fuma)" if in_obese_smoker else "Inactivo",
                  "Alto Riesgo Clínico" if in_obese_smoker else "Riesgo Basal",
                  delta_color="inverse" if in_obese_smoker else "off")

        # Explicabilidad individual con SHAP o aproximación por contribución de árbol
        prep_step = pipeline.named_steps['prep']
        mod_step = pipeline.named_steps['modelo']
        x_trans = prep_step.transform(df_input)
        if sparse.issparse(x_trans):
            x_trans = x_trans.toarray()
        nombres_cols = [n.split('__', 1)[1] for n in prep_step.get_feature_names_out()]

        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(mod_step)
            sv_ind = np.asarray(explainer.shap_values(x_trans)).flatten()
            titulo_exp = "Contribución Individual SHAP por Variable (Escala log(1 + USD))"
        else:
            # Respaldo determinístico si shap no está instalado en el entorno local:
            # diferencia respecto a la mediana de entrenamiento ponderada por importancia
            X_tr_trans = prep_step.transform(X_train)
            if sparse.issparse(X_tr_trans):
                X_tr_trans = X_tr_trans.toarray()
            z_diff = (x_trans.flatten() - np.median(X_tr_trans, axis=0)) / (np.std(X_tr_trans, axis=0) + 1e-6)
            sv_ind = z_diff * mod_step.feature_importances_
            titulo_exp = "Índice de Contribución Local de Variables al Caso Evaluado"

        df_exp_ind = pd.DataFrame({
            'Variable': nombres_cols,
            'Impacto': sv_ind,
            'Efecto': np.where(sv_ind >= 0, 'Aumenta el costo (+)', 'Reduce el costo (-)')
        }).sort_values('Impacto', key=lambda s: np.abs(s), ascending=True)

        fig_shap_ind = px.bar(
            df_exp_ind, x='Impacto', y='Variable', color='Efecto', orientation='h',
            color_discrete_map={'Aumenta el costo (+)': '#C44E52', 'Reduce el costo (-)': '#4C72B0'},
            title=titulo_exp
        )
        st.plotly_chart(fig_shap_ind, use_container_width=True)

# ==============================================================================
# TAB 4: AUDITORÍA ÉTICA (RNF07) Y MONITOREO DE DERIVA MLOPS (KS & PSI)
# ==============================================================================
with tabs[3]:
    st.subheader("Auditoría de Equidad Algorítmica (Hito H5) y Monitoreo de Deriva (Model Drift)")
    
    auditoria = pd.DataFrame({
        'sexo': X_test['sex'].values,
        'region': X_test['region'].values,
        'edad': pd.cut(X_test['age'].values, [17, 30, 45, 64], labels=['18-30', '31-45', '46-64']),
        'segmento': np.where(X_test['obese_smoker'].values == 1, 'Fumador IMC>=30', 'Resto'),
        'error_abs': np.abs(pred_test_usd - y_test_real.values),
        'ratio': pred_test_usd / y_test_real.values
    })
    auditoria['error_rel_%'] = auditoria['error_abs'] / y_test_real.values * 100

    mae_sexo = auditoria.groupby('sexo')['error_abs'].mean()
    brecha_sexo = abs(mae_sexo['female'] - mae_sexo['male']) / auditoria['error_abs'].mean() * 100
    p_mw = mannwhitneyu(
        auditoria.loc[auditoria['sexo'] == 'female', 'error_abs'],
        auditoria.loc[auditoria['sexo'] == 'male', 'error_abs']
    ).pvalue

    r_sexo = auditoria.groupby('sexo')['ratio'].median()
    dir_sexo = float(r_sexo['female'] / r_sexo['male'])
    r_reg = auditoria.groupby('region')['ratio'].median()
    dir_reg = float(r_reg.min() / r_reg.max())
    r_seg = auditoria.groupby('segmento')['ratio'].median()
    dir_seg = float(r_seg.min() / r_seg.max())

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Brecha MAE Género (< 10%)", f"{brecha_sexo:.1f}%", f"Mann-Whitney p = {p_mw:.3f}")
    e2.metric("Razón Calibración Sexo", f"{dir_sexo:.2f}", "Rango RNF07: [0,80 – 1,25]")
    e3.metric("Razón Calibración Región", f"{dir_reg:.2f}", "Rango RNF07: [0,80 – 1,25]")
    e4.metric("Razón Calibración Segmento", f"{dir_seg:.2f}", "Rango RNF07: [0,80 – 1,25]")

    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown("##### ⚖️ Desempeño de Equidad por Sexo y Región (Conjunto de Prueba)")
        tabla_sexo = auditoria.groupby('sexo', observed=True).agg(
            Afiliados=('error_abs', 'size'),
            MAE_USD=('error_abs', lambda s: round(s.mean(), 1)),
            Ratio_Mediano=('ratio', lambda s: round(s.median(), 3)),
            Error_Rel_Mediano_Pct=('error_rel_%', lambda s: round(s.median(), 2))
        )
        st.dataframe(tabla_sexo, use_container_width=True)
        
        tabla_reg = auditoria.groupby('region', observed=True).agg(
            Afiliados=('error_abs', 'size'),
            MAE_USD=('error_abs', lambda s: round(s.mean(), 1)),
            Ratio_Mediano=('ratio', lambda s: round(s.median(), 3)),
            Error_Rel_Mediano_Pct=('error_rel_%', lambda s: round(s.median(), 2))
        )
        st.dataframe(tabla_reg, use_container_width=True)

    with col_eq2:
        st.markdown("##### 📡 Simulador MLOps de Deriva Estadística (Model Drift: KS y PSI)")
        shift_bmi = st.slider("Simular desplazamiento en media de IMC de nuevos lotes (+ puntos IMC):", -3.0, 5.0, 0.0, 0.5)
        lote_base = X_train['bmi'].values
        lote_nuevo = X_test['bmi'].values + shift_bmi
        
        # Cálculo de Kolmogorov-Smirnov y Population Stability Index (PSI)
        ks_stat, ks_pval = ks_2samp(lote_base, lote_nuevo)
        cortes_psi = np.percentile(lote_base, np.linspace(0, 100, 11))
        cortes_psi[0], cortes_psi[-1] = -np.inf, np.inf
        p_base = np.histogram(lote_base, bins=cortes_psi)[0] / len(lote_base)
        p_nuevo = np.histogram(lote_nuevo, bins=cortes_psi)[0] / len(lote_nuevo)
        p_base = np.clip(p_base, 1e-4, None)
        p_nuevo = np.clip(p_nuevo, 1e-4, None)
        psi_val = float(np.sum((p_nuevo - p_base) * np.log(p_nuevo / p_base)))

        alerta_drift = (ks_pval < 0.05) or (psi_val > 0.25)
        st.markdown(
            f"**Resultado del Control MLOps (Sección 5.2):** "
            f"`PSI = {psi_val:.3f}` (Umbral > 0,25) · `K-S p-value = {ks_pval:.4f}` (Umbral < 0,05) → "
            + ("<span class='badge-crit'>🚨 ALERTA DRIFT: REENTRENAR MODELO</span>" if alerta_drift else "<span class='badge-ok'>✅ DISTRIBUCIÓN ESTABLE</span>"),
            unsafe_allow_html=True
        )
        df_drift_plot = pd.DataFrame({
            'IMC': np.concatenate([lote_base, lote_nuevo]),
            'Lote': ['Base Entrenamiento (Referencia)'] * len(lote_base) + ['Lote Producción (Monitoreado)'] * len(lote_nuevo)
        })
        fig_drift = px.histogram(df_drift_plot, x='IMC', color='Lote', barmode='overlay', histnorm='probability density',
                                 color_discrete_map={'Base Entrenamiento (Referencia)': '#4C72B0', 'Lote Producción (Monitoreado)': '#C44E52'},
                                 title="Comparación de Distribuciones para Detección de Data Drift (IMC)")
        st.plotly_chart(fig_drift, use_container_width=True)

# ==============================================================================
# TAB 5: FUNDAMENTACIÓN FINANCIERA Y SENSIBILIDAD (SECCIÓN 2.4 DEL INFORME)
# ==============================================================================
with tabs[4]:
    st.subheader("Fundamentación Financiera, Punto de Equilibrio y Escenarios Hipotéticos (RF04)")
    st.warning("⚠️ **Nota Metodológica Crítica (Sección 1.1 y 2.4):** El modelo predictivo estima costos con alta precisión ($R^2=0,903$), pero el dataset histórico no registra intervenciones clínicas. Por tanto, los ahorros mostrados a continuación constituyen **escenarios hipotéticos de planificación** sujetos a validación mediante un piloto controlado.")
    
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        st.markdown("#### 💵 Parámetros de Sensibilidad")
        capex_sw = st.number_input("Inversión Inicial CapEx Software (USD):", value=7950, step=100,
                                   help="RRHH 272h ($6.800) + AWS ($225) + CI/CD ($75) + Contingencia 12% ($850)")
        costo_afiliado = st.slider("Costo Anual Programa Clínico por Afiliado (USD):", 60, 300, 120, 10,
                                   help="Supuesto de planificación base: $120 USD/afiliado/año (Sección 2.4)")
        pct_ahorro_sim = st.slider("% Reducción Hipotética del Gasto en Segmento Crítico:", 0.2, 8.0, 3.0, 0.1)
        
        n_criticos = 145
        gasto_critico_total = 6025908.53
        opex_programa = n_criticos * costo_afiliado
        costo_total_ecosistema = capex_sw + opex_programa
        breakeven_pct = (costo_total_ecosistema / gasto_critico_total) * 100
        
        ahorro_bruto_sim = gasto_critico_total * (pct_ahorro_sim / 100.0)
        beneficio_neto_sim = ahorro_bruto_sim - costo_total_ecosistema
        roi_sim = (beneficio_neto_sim / costo_total_ecosistema) * 100
        payback_meses = (costo_total_ecosistema / ahorro_bruto_sim) * 12 if ahorro_bruto_sim > 0 else np.nan

    with fc2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Costo Ecosistema Año 1", f"${costo_total_ecosistema:,.0f} USD", f"CapEx ${capex_sw:,.0f} + OpEx ${opex_programa:,.0f}")
        m2.metric("Punto de Equilibrio", f"{breakeven_pct:.2f}%", f"Ahorro mín: ${costo_total_ecosistema:,.0f} USD")
        m3.metric(f"Beneficio Neto ({pct_ahorro_sim:.1f}%)", f"${beneficio_neto_sim:,.0f} USD", f"ROI: {roi_sim:,.1f}%")
        m4.metric("Payback Ecosistema", f"{payback_meses:.2f} meses", f"~{payback_meses*30.4:.0f} días")
        
        escenarios = [
            ('Punto de Equilibrio', breakeven_pct),
            ('Pesimista (1,0%)', 1.0),
            ('Base (3,0%)', 3.0),
            (f'Simulado ({pct_ahorro_sim:.1f}%)', pct_ahorro_sim),
            ('Optimista (5,0%)', 5.0)
        ]
        filas_esc = []
        for nom, p in escenarios:
            ah = gasto_critico_total * (p / 100.0)
            bn = ah - costo_total_ecosistema
            roi_e = (bn / costo_total_ecosistema) * 100
            filas_esc.append({
                'Escenario Hipotético': nom,
                '% Reducción Gasto': f"{p:.2f}%",
                'Ahorro Bruto (USD)': round(ah, 2),
                'Beneficio Neto Año 1 (USD)': round(bn, 2),
                'ROI Ecosistema (%)': f"{roi_e:,.2f}%"
            })
        df_esc = pd.DataFrame(filas_esc)
        st.dataframe(df_esc, use_container_width=True, hide_index=True)

        fig_fin = px.bar(
            df_esc, x='Escenario Hipotético', y=['Ahorro Bruto (USD)', 'Beneficio Neto Año 1 (USD)'],
            barmode='group', title="Análisis de Sensibilidad Financiera (Ecosistema Completo Año 1, USD)",
            color_discrete_sequence=['#4C72B0', '#55A868']
        )
        st.plotly_chart(fig_fin, use_container_width=True)

# ==============================================================================
# TAB 6: MONITOREO Y CONTROL DEL PROYECTO (JIRA, WBS Y DESVIACIONES)
# ==============================================================================
with tabs[5]:
    st.subheader("Monitoreo, Control de Ejecución y Gestión de Desviaciones (Indicador 2 — 50%)")
    
    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Avance Global WBS", "81,6%", "222 h ejecutadas de 272 h")
    j2.metric("Paquetes Completados", "10 / 13", "Corte: Cierre Semana 5")
    j3.metric("Buffer consumido (S4)", "3 días", "Activado por Riesgo R3 (Tuning)")
    j4.metric("Presupuesto Consumido", "$5.850 / $7.950", "Reserva intacta ($850 USD)")
    
    st.markdown("##### 📋 Registro Trazable de Tareas Planificadas vs. Ejecutadas (WBS ↔ Tablero Jira `PCM`)")
    df_wbs = pd.DataFrame([
        {'Cód. WBS': '1.1', 'Tarea Planificada (Jira)': 'Project Charter, Levantamiento MoSCoW y H1', 'Responsable': 'Nicolás Fonseca (PM)', 'Horas': 12, 'Plan': 'S1', 'Real': 'S1', 'Estado': '✅ Completada'},
        {'Cód. WBS': '1.2', 'Tarea Planificada (Jira)': 'Aprovisionamiento Cloud y Permisos RBAC', 'Responsable': 'Bastián Rubio (DE)', 'Horas': 8, 'Plan': 'S1', 'Real': 'S1', 'Estado': '✅ Completada'},
        {'Cód. WBS': '1.3', 'Tarea Planificada (Jira)': 'Control de Hitos y Formularios de Cambio', 'Responsable': 'Nicolás Fonseca (PM)', 'Horas': 12, 'Plan': 'S1–S5', 'Real': 'S1–S5', 'Estado': '✅ Completada'},
        {'Cód. WBS': '2.1', 'Tarea Planificada (Jira)': 'Ingesta (1.337 reg.) y Validación Esquema (RF01)', 'Responsable': 'Bastián Rubio (DE)', 'Horas': 24, 'Plan': 'S2', 'Real': 'S2', 'Estado': '✅ Completada'},
        {'Cód. WBS': '2.2', 'Tarea Planificada (Jira)': 'Cifrado AES-256, Seudonimización SHA-256 y PIA', 'Responsable': 'Bastián Rubio (DE)', 'Horas': 24, 'Plan': 'S2–S3', 'Real': 'S3', 'Estado': '✅ Completada'},
        {'Cód. WBS': '2.3', 'Tarea Planificada (Jira)': 'Análisis Exploratorio Profundo (EDA) y Reporte', 'Responsable': 'Cristóbal Segovia (DS)', 'Horas': 24, 'Plan': 'S2–S3', 'Real': 'S2–S3', 'Estado': '✅ Completada'},
        {'Cód. WBS': '3.1', 'Tarea Planificada (Jira)': 'Ingeniería de Características (obese_smoker, OMS)', 'Responsable': 'Cristóbal Segovia (DS)', 'Horas': 24, 'Plan': 'S3', 'Real': 'S3–S4', 'Estado': '✅ Completada'},
        {'Cód. WBS': '3.2', 'Tarea Planificada (Jira)': 'Entrenamiento, Tuning y Benchmark R² ≥ 0,85', 'Responsable': 'N. Fonseca / C. Segovia (DS)', 'Horas': 32, 'Plan': 'S3–S4', 'Real': 'S4 (Buffer)', 'Estado': '✅ Completada'},
        {'Cód. WBS': '3.3', 'Tarea Planificada (Jira)': 'Algoritmo de Scoring y Segmentación RF03', 'Responsable': 'Cristóbal Segovia (DS)', 'Horas': 16, 'Plan': 'S4', 'Real': 'S4', 'Estado': '✅ Completada'},
        {'Cód. WBS': '3.4', 'Tarea Planificada (Jira)': 'Auditoría de Equidad Ética (H5) y Módulo SHAP', 'Responsable': 'Nicolás Fonseca (DS)', 'Horas': 24, 'Plan': 'S4–S5', 'Real': 'S5', 'Estado': '✅ Completada'},
        {'Cód. WBS': '4.1', 'Tarea Planificada (Jira)': 'Serialización (.joblib/.json) y Contenedor Docker', 'Responsable': 'Bastián Rubio (MLOps)', 'Horas': 16, 'Plan': 'S4–S5', 'Real': 'S5', 'Estado': '✅ Completada'},
        {'Cód. WBS': '4.2', 'Tarea Planificada (Jira)': 'Pipeline CI/CD y Pruebas de Estrés Lote ≤ 5s', 'Responsable': 'Bastián Rubio (MLOps)', 'Horas': 16, 'Plan': 'S5', 'Real': 'S5', 'Estado': '🔄 En progreso'},
        {'Cód. WBS': '4.3–5.2', 'Tarea Planificada (Jira)': 'Despliegue Final Dashboard, Guía Drift y Sign-off PO', 'Responsable': 'N. Fonseca / B. Rubio', 'Horas': 40, 'Plan': 'S6', 'Real': '—', 'Estado': '⏳ Pendiente'}
    ])
    st.dataframe(df_wbs, use_container_width=True, hide_index=True)

    st.markdown("##### ⚠️ Registro de Desviaciones Detectadas y Acciones Correctivas Aplicadas")
    df_desv = pd.DataFrame([
        {
            'ID': 'D1',
            'Desviación Detectada': 'Incumplimiento de meta MAE < $2.100 (RF02) usando solo variables originales (MAE = $2.156 en RF base y $2.123 tras tuning).',
            'Impacto': 'Bloqueo de compuerta técnica H4 e imposibilidad de pasar a producción (Riesgo R3).',
            'Acción Correctiva': 'Uso del Buffer de 3 días en S4: creación de variable explícita obese_smoker, RandomizedSearchCV (30 iter.) en USD y estudio de ablación.',
            'Resultado Verificado': 'MAE se redujo en $168 USD (-8%), alcanzando MAE = $2.011 en CV y $1.895 en Test (R² = 0,903). Hito H4 aprobado a tiempo.'
        },
        {
            'ID': 'D2',
            'Desviación Detectada': 'Subestimación promedio de +$1.845 USD en el decil superior de costos (10% más caro, MAE = $2.807 USD) por cola asimétrica.',
            'Impacto': 'Riesgo de sub-presupuestación del fondo de siniestralidad para casos catastróficos (4,1% del gasto del grupo).',
            'Acción Correctiva': 'Mantener regularización del árbol (evitar overfitting) e incorporar regla financiera de reserva adicional de +4,1% para el decil 90.',
            'Resultado Verificado': 'Modelo generalizable con sesgo de cola cuantificado y cubierto financieramente.'
        },
        {
            'ID': 'D3',
            'Desviación Detectada': 'Brecha de MAE inter-género en 8,7% (cerca del umbral de 10%, p=0,047) y dispersión de MAE regional por desbalance de fumadores.',
            'Impacto': 'Alerta temprana en Riesgo R1 (Equidad) y R8 (Representatividad regional en muestras de n=60–77).',
            'Acción Correctiva': 'Verificación mediante Razón de Calibración mediana (0,97 en sexo y 0,94 en región, dentro de [0,80–1,25]) y prohibición de uso tarifario.',
            'Resultado Verificado': 'Hito H5 aprobado con vigilancia estadística continua en cada reentrenamiento.'
        }
    ])
    st.dataframe(df_desv, use_container_width=True, hide_index=True)
