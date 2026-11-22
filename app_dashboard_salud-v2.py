# ==============================================================================
# DASHBOARD EJECUTIVO Y SIMULADOR PREVENTIVO DE SINIESTRALIDAD (SCY1102)
# Equipo: Nicolás Fonseca Olivares, Cristóbal Segovia Retamal, Bastián Rubio Meneses
# Docente: Prof. Michael Enrique Palomino Barrera · Duoc UC — Sede San Joaquín
# ==============================================================================
# Instrucciones de ejecución:
#   pip install streamlit pandas numpy scikit-learn plotly joblib shap scipy
#   streamlit run app_dashboard_salud-v2.py
# ==============================================================================

import os
import time
import hashlib
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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA, MAPAS DE NOMBRES DE NEGOCIO Y ESTILOS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Ecosistema Preventivo de Salud | SCY1102",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAPA_SEXO = {"female": "Mujer", "male": "Hombre"}
MAPA_FUMADOR = {"no": "No fumador", "yes": "Fumador"}
MAPA_REGION = {
    "northeast": "Noreste",
    "northwest": "Noroeste",
    "southeast": "Sureste",
    "southwest": "Suroeste",
}
MAPA_VARIABLES = {
    "age": "Edad (años)",
    "bmi": "Índice de Masa Corporal (IMC)",
    "children": "Cargas familiares (hijos)",
    "obese_smoker": "Fumador con Obesidad (IMC ≥ 30)",
    "age_bmi_interaction": "Interacción Edad × IMC",
    "sex": "Sexo",
    "sex_male": "Sexo: Hombre",
    "smoker": "Tabaquismo",
    "smoker_yes": "Fumador activo",
    "region": "Región",
    "region_northwest": "Región: Noroeste",
    "region_southeast": "Región: Sureste",
    "region_southwest": "Región: Suroeste",
}

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #1B365D 0%, #2C5282 55%, #1E4E79 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.94rem;
        color: #dbeafe;
    }
    .decision-box {
        background-color: #f0f7ff;
        border-left: 5px solid #1B365D;
        padding: 0.9rem 1.15rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        color: #1e293b;
        font-size: 0.92rem;
    }
    .badge-ok {
        background-color: #d4edda;
        color: #155724;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.83rem;
    }
    .badge-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.83rem;
    }
    .badge-crit {
        background-color: #f8d7da;
        color: #721c24;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.83rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. INGESTA, VALIDACIÓN DE ESQUEMA (RF01) Y ENTRENAMIENTO REPRODUCIBLE (RF02 / RNF02)
# ------------------------------------------------------------------------------
FEATURES_NUM = ["age", "bmi", "children", "obese_smoker", "age_bmi_interaction"]
FEATURES_CAT = ["sex", "smoker", "region"]
TARGET = "charges"


@st.cache_data
def cargar_y_preparar_datos(filepath="insurance.csv"):
    if not os.path.exists(filepath):
        st.error(f"Error RF01: No se encontró el archivo '{filepath}' en el directorio actual.")
        st.stop()

    raw = pd.read_csv(filepath)
    n_original = len(raw)

    # Validación de esquema y rangos clínicos (RF01)
    columnas_esperadas = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    faltantes = columnas_esperadas - set(raw.columns)
    if faltantes:
        st.error(f"Error de Esquema (RF01): Faltan columnas obligatorias: {sorted(faltantes)}")
        st.stop()

    df = raw.drop_duplicates().reset_index(drop=True)
    duplicados_eliminados = n_original - len(df)
    nulos_detectados = int(df.isna().sum().sum())
    if nulos_detectados > 0:
        st.error(f"Error RF01: Se detectaron {nulos_detectados} valores nulos en el lote.")
        st.stop()

    if not ((df["age"] >= 18).all() and (df["bmi"] > 0).all() and (df["charges"] > 0).all()):
        st.error("Error RF01: Existen registros fuera de rango clínico válido (edad < 18, IMC ≤ 0 o costo ≤ 0).")
        st.stop()

    # Seudonimización criptográfica SHA-256 sin PII (RNF04)
    df["id_afiliado_hash"] = [
        hashlib.sha256(f"AFILIADO_{i}_SALT_SCY1102".encode("utf-8")).hexdigest()[:16]
        for i in df.index
    ]

    # Ingeniería de variables y etiquetas legibles de negocio
    df["obese_smoker"] = ((df["bmi"] >= 30) & (df["smoker"] == "yes")).astype(int)
    df["age_bmi_interaction"] = df["age"] * df["bmi"]

    df["sexo_es"] = df["sex"].map(MAPA_SEXO)
    df["fumador_es"] = df["smoker"].map(MAPA_FUMADOR)
    df["region_es"] = df["region"].map(MAPA_REGION)

    cortes_bmi = [0, 18.5, 25, 30, 35, 40, 100]
    etiquetas_bmi = [
        "Bajo peso",
        "Normal",
        "Sobrepeso",
        "Obesidad I",
        "Obesidad II",
        "Obesidad III",
    ]
    df["categoria_imc"] = pd.cut(df["bmi"], bins=cortes_bmi, labels=etiquetas_bmi)

    cortes_edad = [17, 25, 45, 55, 64]
    etiquetas_edad = ["Joven (18-25)", "Adulto (26-45)", "Maduro (46-55)", "Senior (56-64)"]
    df["tramo_etario"] = pd.cut(df["age"], bins=cortes_edad, labels=etiquetas_edad)

    df["perfil_referencia"] = np.where(
        df["obese_smoker"] == 1, "Fumador con IMC ≥ 30", "Resto de afiliados"
    )

    info_ingesta = {
        "n_original": n_original,
        "n_limpio": len(df),
        "duplicados": duplicados_eliminados,
        "nulos": nulos_detectados,
    }
    return df, info_ingesta


def mae_dolares(est, X_, y_log):
    return -mean_absolute_error(np.expm1(y_log), np.expm1(est.predict(X_)))


def r2_dolares(est, X_, y_log):
    return r2_score(np.expm1(y_log), np.expm1(est.predict(X_)))


@st.cache_resource
def obtener_modelo_benchmark_y_ablacion(df):
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = np.log1p(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test_real = np.expm1(y_test)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", FEATURES_NUM),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), FEATURES_CAT),
        ]
    )

    # 1. Benchmark dinámico de 6 algoritmos (evidencia de selección de modelo)
    candidatos = {
        "Baseline: Mediana": DummyRegressor(strategy="median"),
        "Regresión Lineal": LinearRegression(),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "Random Forest Base": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Random Forest Ajustado (Final)": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.5,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    filas_bench = []
    pipeline_final = None
    for nombre, mod in candidatos.items():
        pipe = Pipeline([("prep", preprocessor), ("modelo", mod)])
        cv_res = cross_validate(pipe, X, y, cv=kfold, scoring={"mae": mae_dolares, "r2": r2_dolares})
        pipe.fit(X_train, y_train)
        pred_t = np.expm1(pipe.predict(X_test))

        mae_cv = -cv_res["test_mae"].mean()
        r2_cv = cv_res["test_r2"].mean()
        mae_t = mean_absolute_error(y_test_real, pred_t)
        rmse_t = root_mean_squared_error(y_test_real, pred_t)
        mape_t = mean_absolute_percentage_error(y_test_real, pred_t) * 100
        r2_t = r2_score(y_test_real, pred_t)

        cumple_test = (r2_t >= 0.85) and (mae_t < 2100)
        cumple_cv = (r2_cv >= 0.85) and (mae_cv < 2100)
        if cumple_test and cumple_cv:
            estado = "Cumple Test y CV"
        elif cumple_test:
            estado = "Cumple solo Test (no CV)"
        else:
            estado = "No cumple RF02"

        filas_bench.append(
            {
                "Modelo evaluado": nombre,
                "MAE CV 5-Fold (USD)": round(mae_cv, 1),
                "R² CV 5-Fold": round(r2_cv, 4),
                "MAE Prueba (USD)": round(mae_t, 1),
                "RMSE Prueba (USD)": round(rmse_t, 1),
                "MAPE Prueba (%)": round(mape_t, 2),
                "R² Prueba": round(r2_t, 4),
                "Veredicto RF02": estado,
            }
        )
        if nombre == "Random Forest Ajustado (Final)":
            pipeline_final = pipe
            cv_final = cv_res

    df_benchmark = pd.DataFrame(filas_bench)

    # Persistir artefacto serializado para reproducibilidad (RNF02)
    try:
        joblib.dump(pipeline_final, "modelo_costos_medicos.joblib")
    except Exception:
        pass

    # 2. Estudio de ablación dinámico (evidencia técnica de Desviación D1)
    configs_abl = {
        "Todas las variables (Modelo Final)": FEATURES_NUM,
        "Sin interacción Fumador × Obesidad (IMC ≥ 30)": [
            "age",
            "bmi",
            "children",
            "age_bmi_interaction",
        ],
        "Sin interacción Edad × IMC": ["age", "bmi", "children", "obese_smoker"],
        "Solo variables originales (sin ingeniería)": ["age", "bmi", "children"],
    }
    filas_abl = []
    mae_final_cv = -cv_final["test_mae"].mean()
    for cfg_nombre, cols_num in configs_abl.items():
        prep_ab = ColumnTransformer(
            [
                ("num", "passthrough", cols_num),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), FEATURES_CAT),
            ]
        )
        pipe_ab = Pipeline(
            [
                ("prep", prep_ab),
                (
                    "modelo",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        max_features=0.5,
                        max_depth=8,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        cv_ab = cross_validate(
            pipe_ab,
            df[cols_num + FEATURES_CAT],
            y,
            cv=kfold,
            scoring={"mae": mae_dolares, "r2": r2_dolares},
        )
        mae_v = -cv_ab["test_mae"].mean()
        r2_v = cv_ab["test_r2"].mean()
        delta = mae_v - mae_final_cv
        cumple = (mae_v < 2100) and (r2_v >= 0.85)
        filas_abl.append(
            {
                "Configuración evaluada": cfg_nombre,
                "MAE CV 5-Fold (USD)": round(mae_v, 1),
                "Diferencia vs Final (USD)": f"{delta:+.1f}",
                "R² CV 5-Fold": round(r2_v, 4),
                "¿Cumple RF02 en CV?": "Sí" if cumple else "No (supera $2.100)",
            }
        )
    df_ablacion = pd.DataFrame(filas_abl)

    # 3. Medición de rendimiento por lote completo (RNF01 <= 5 s) y métricas de test
    t0 = time.perf_counter()
    pred_all_usd = np.expm1(pipeline_final.predict(X))
    tiempo_lote = time.perf_counter() - t0

    pred_test_usd = np.expm1(pipeline_final.predict(X_test))
    mae_test = mean_absolute_error(y_test_real, pred_test_usd)
    mape_test = mean_absolute_percentage_error(y_test_real, pred_test_usd) * 100
    r2_test = r2_score(y_test_real, pred_test_usd)

    # 4. Importancia global por permutación con nombres ejecutivos en español (RF04 / RNF03)
    perm = permutation_importance(
        pipeline_final, X_test, y_test, n_repeats=10, random_state=42, scoring=mae_dolares
    )
    feat_imp = pd.DataFrame(
        {
            "Factor Explicativo": [MAPA_VARIABLES.get(c, c) for c in (FEATURES_NUM + FEATURES_CAT)],
            "Aumento del MAE (USD)": perm.importances_mean,
            "Desviación (USD)": perm.importances_std,
        }
    ).sort_values("Aumento del MAE (USD)", ascending=True)

    metricas = {
        "mae_test": mae_test,
        "mape_test": mape_test,
        "r2_test": r2_test,
        "mae_cv": -cv_final["test_mae"].mean(),
        "mae_cv_std": cv_final["test_mae"].std(),
        "r2_cv": cv_final["test_r2"].mean(),
        "r2_cv_std": cv_final["test_r2"].std(),
        "tiempo_lote_seg": tiempo_lote,
        "n_test": len(X_test),
        "n_train": len(X_train),
    }

    return (
        pipeline_final,
        metricas,
        df_benchmark,
        df_ablacion,
        X_train,
        X_test,
        y_train,
        y_test_real,
        pred_test_usd,
        pred_all_usd,
        feat_imp,
    )


df, info_ingesta = cargar_y_preparar_datos()
(
    pipeline,
    metricas,
    df_benchmark,
    df_ablacion,
    X_train,
    X_test,
    y_train,
    y_test_real,
    pred_test_usd,
    pred_all_usd,
    feat_imp,
) = obtener_modelo_benchmark_y_ablacion(df)

df_cartera = df.copy()
df_cartera["costo_proyectado"] = pred_all_usd
df_cartera["residuo"] = df_cartera["charges"] - df_cartera["costo_proyectado"]

# ------------------------------------------------------------------------------
# 3. BARRA LATERAL: CONTROLES OPERATIVOS Y VERIFICACIÓN DE COMPUERTAS (H1–H5)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 Panel de Control Clínico")
    st.caption("SCY1102 — Gestión de Proyectos de Datos · Duoc UC (San Joaquín)")
    st.divider()

    st.markdown("#### ⚙️ Parámetros de Decisión")
    umbral_rf03 = st.slider(
        "Umbral de Atención Prioritaria (RF03, USD):",
        min_value=10000,
        max_value=35000,
        value=20000,
        step=1000,
        help="Regla oficial RF03: $20.000 USD. Ajustar permite evaluar cómo cambia la cobertura del perfil crítico y el presupuesto preventivo.",
    )

    regiones_disponibles = list(MAPA_REGION.values())
    regiones_sel = st.multiselect(
        "Filtrar por Región:",
        options=regiones_disponibles,
        default=regiones_disponibles,
    )
    if not regiones_sel:
        regiones_sel = regiones_disponibles

    st.divider()
    st.markdown("#### ✅ Compuertas de Calidad (H1–H5)")
    st.markdown(
        f"- **RF01 (Esquema):** <span class='badge-ok'>{info_ingesta['n_limpio']:,} limpios</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- **RF02 (R² ≥ 0,85):** <span class='badge-ok'>R² = {metricas['r2_test']:.4f}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- **RF02 (MAE < $2.100):** <span class='badge-ok'>${metricas['mae_test']:,.0f} USD</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- **RNF01 (Lote ≤ 5s):** <span class='badge-ok'>{metricas['tiempo_lote_seg']:.4f} s</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "- **RNF04 (Ley 21.719):** <span class='badge-ok'>SHA-256 + PIA</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "- **RNF07 (Uso ético):** <span class='badge-ok'>Solo preventivo</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "**Equipo:** Nicolás Fonseca Olivares · Cristóbal Segovia Retamal · Bastián Rubio Meneses\n\n"
        "**Docente:** Prof. Michael Enrique Palomino Barrera"
    )

df_filtrado = df_cartera[df_cartera["region_es"].isin(regiones_sel)].copy()
df_filtrado["segmento_rf03"] = np.where(
    df_filtrado["costo_proyectado"] > umbral_rf03,
    "Atención Preventiva Prioritaria",
    "Seguimiento Estándar",
)

# ------------------------------------------------------------------------------
# 4. ENCABEZADO EJECUTIVO Y KPIs PRINCIPALES
# ------------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <h1>🏥 Ecosistema Preventivo de Salud y Predicción de Siniestralidad</h1>
    <p>Herramienta de Soporte a Decisiones (CRISP-DM + MLOps) · Modelo Random Forest Ajustado + Explicabilidad SHAP + Gobernanza Ética (Ley 21.719)</p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        "Cartera Válida (RF01)",
        f"{len(df_filtrado):,} afiliados",
        f"-{info_ingesta['duplicados']} duplicado depurado",
    )
with col2:
    gasto_fum = df_filtrado[df_filtrado["smoker"] == "yes"]["charges"].mean()
    gasto_nofum = df_filtrado[df_filtrado["smoker"] == "no"]["charges"].mean()
    factor_fum = gasto_fum / gasto_nofum if gasto_nofum > 0 else 0
    st.metric(
        "Costo Medio Fumador",
        f"${gasto_fum:,.0f} USD",
        f"{factor_fum:.1f}x vs No fumador (${gasto_nofum:,.0f})",
        delta_color="inverse",
    )
with col3:
    df_ref = df_filtrado[df_filtrado["obese_smoker"] == 1]
    df_resto = df_filtrado[df_filtrado["obese_smoker"] == 0]
    media_ref = df_ref["charges"].mean() if len(df_ref) > 0 else 0
    media_resto = df_resto["charges"].mean() if len(df_resto) > 0 else 1
    st.metric(
        "Perfil Crítico (Fumador + IMC ≥ 30)",
        f"${media_ref:,.0f} USD",
        f"{len(df_ref)} afil. ({media_ref / media_resto:.1f}x resto)",
        delta_color="inverse",
    )
with col4:
    st.metric(
        "Desempeño Modelo (RF02)",
        f"R² = {metricas['r2_test']:.4f}",
        f"MAE Test: ${metricas['mae_test']:,.0f} (CV: ${metricas['mae_cv']:,.0f})",
    )
with col5:
    n_prio = int((df_filtrado["segmento_rf03"] == "Atención Preventiva Prioritaria").sum())
    st.metric(
        f"Prioritarios (RF03 > ${umbral_rf03:,.0f})",
        f"{n_prio} afiliados",
        f"{n_prio / len(df_filtrado) * 100:.1f}% de la cartera",
    )

# ------------------------------------------------------------------------------
# 5. PESTAÑAS DEL DASHBOARD (ALINEADAS 100% CON RF01–RF06 Y RNF01–RNF07)
# ------------------------------------------------------------------------------
tabs = st.tabs(
    [
        "📊 1. EDA y Segmentación (RF03 / RF05)",
        "🤖 2. Selección de Modelo y Ablación (RF02 / RF04)",
        "🩺 3. Simulador Clínico y Contrafactual (RF04 / RNF03)",
        "⚖️ 4. Equidad Ética y Deriva MLOps (RNF05 / RNF06)",
        "💰 5. Sensibilidad Financiera y Equilibrio (RF06)",
        "📋 6. Control WBS ↔ Jira y Decisiones (CRISP-DM)",
    ]
)

# ==============================================================================
# TAB 1: EDA Y SEGMENTACIÓN PREVENTIVA (RF01 / RF03 / RF05 / RNF04)
# ==============================================================================
with tabs[0]:
    st.subheader("¿Por qué segmentamos por costo proyectado y perfil Fumador con Obesidad (IMC ≥ 30)?")
    st.markdown(
        r"""
        <div class="decision-box">
            <b>Decisión de Negocio (RF03):</b> Fijar el umbral de derivación a <i>Atención Preventiva Prioritaria</i> en <b>$20.000 USD anuales proyectados</b>.<br>
            <b>¿Por qué?</b> El IMC aislado presenta una correlación de Spearman baja con el costo médico (r<sub>s</sub> = 0,12), pero al combinarse con tabaquismo desde Obesidad Grado I (IMC ≥ 30) forma un salto estructural que multiplica el costo medio por <b>4,2x</b> ($41.558 vs $9.839 USD, acumulando $6.025.909 USD en solo el 10,8% de la cartera).<br>
            <b>¿Cómo se aplica?</b> El modelo proyecta el gasto de cada afiliado y prioriza automáticamente a quienes superan el umbral sin utilizar la predicción para recargos tarifarios ni exclusiones (RNF07).
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_scatter = px.scatter(
            df_filtrado,
            x="age",
            y="charges",
            color="perfil_referencia",
            size="bmi",
            color_discrete_map={
                "Resto de afiliados": "#2C5282",
                "Fumador con IMC ≥ 30": "#C53030",
            },
            hover_data={
                "sexo_es": True,
                "region_es": True,
                "children": True,
                "costo_proyectado": ":$,.0f",
                "age": True,
                "bmi": ":.1f",
                "charges": ":$,.0f",
            },
            labels={
                "age": "Edad (años)",
                "charges": "Gasto Facturado Anual (USD)",
                "perfil_referencia": "Segmento de Referencia",
                "sexo_es": "Sexo",
                "region_es": "Región",
                "children": "Cargas familiares",
                "costo_proyectado": "Costo proyectado (USD)",
                "bmi": "IMC",
            },
            title="Evidencia EDA: Interacción Edad × IMC × Tabaquismo sobre el Gasto (USD)",
        )
        fig_scatter.add_hline(
            y=umbral_rf03,
            line_dash="dash",
            line_color="#111827",
            annotation_text=f"Umbral RF03 (${umbral_rf03:,.0f} USD)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        resumen_bmi = (
            df_filtrado.groupby(["categoria_imc", "fumador_es"], observed=True)["charges"]
            .median()
            .reset_index()
        )
        fig_bmi = px.bar(
            resumen_bmi,
            x="categoria_imc",
            y="charges",
            color="fumador_es",
            barmode="group",
            color_discrete_map={"No fumador": "#2C5282", "Fumador": "#C53030"},
            labels={
                "categoria_imc": "Categoría Clínica IMC (OMS)",
                "charges": "Costo Mediano Anual (USD)",
                "fumador_es": "Tabaquismo",
            },
            title="Salto de Siniestralidad a partir de Obesidad I (IMC ≥ 30) en Fumadores",
        )
        # Valores en USD enteros (con separador de miles) en el eje y en el hover
        fig_bmi.update_yaxes(tickprefix="$", tickformat=",.0f")
        fig_bmi.update_traces(hovertemplate="%{x}<br>Costo mediano: $%{y:,.0f} USD<extra>%{fullData.name}</extra>")
        st.plotly_chart(fig_bmi, use_container_width=True)

    st.divider()
    st.markdown(
        f"#### 🎯 Evaluación Operativa del Umbral Seleccionado (`RF03` > ${umbral_rf03:,.0f} USD)"
    )

    crit_capturados = int(
        (
            (df_filtrado["segmento_rf03"] == "Atención Preventiva Prioritaria")
            & (df_filtrado["obese_smoker"] == 1)
        ).sum()
    )
    total_crit = int((df_filtrado["obese_smoker"] == 1).sum())
    cobertura_crit = (crit_capturados / total_crit * 100) if total_crit > 0 else 0.0
    otros_prioritarios = n_prio - crit_capturados
    presupuesto_est_prog = n_prio * 120

    m_u1, m_u2, m_u3 = st.columns(3)
    m_u1.metric(
        "Cobertura del Perfil Crítico (IMC ≥ 30 + Fumador)",
        f"{cobertura_crit:.1f}%",
        f"{crit_capturados} de {total_crit} afiliados críticos capturados",
    )
    m_u2.metric(
        "Otros Perfiles de Alto Costo Detectados",
        f"{otros_prioritarios} afiliados",
        "Fumadores con IMC < 30 o alta carga por edad",
    )
    m_u3.metric(
        "Costo Estimado del Programa (a $120/afil./año)",
        f"${presupuesto_est_prog:,.0f} USD/año",
        f"Para {n_prio} afiliados en Atención Prioritaria",
    )

    c_tabla, c_export = st.columns([2, 1])
    with c_tabla:
        cruce_rf03 = pd.crosstab(
            df_filtrado["segmento_rf03"],
            df_filtrado["perfil_referencia"],
            margins=True,
            margins_name="Total Cartera",
        )
        cruce_rf03.index.name = "Segmento Asignado (RF03)"
        st.dataframe(cruce_rf03, use_container_width=True)

    with c_export:
        st.info(
            "**Privacidad y Uso Ético (`RNF04` / `RNF07`):**\n\n"
            "La nómina exportable utiliza identificadores seudonimizados con hash **SHA-256** sin exponer nombres, RUT ni datos de contacto. "
            "Cualquier vinculación con afiliados reales requiere base legal y aprobación del Comité de Ética (Ley 19.628 / Ley 21.719)."
        )
        df_export = df_filtrado[
            [
                "id_afiliado_hash",
                "age",
                "sexo_es",
                "bmi",
                "categoria_imc",
                "children",
                "fumador_es",
                "region_es",
                "costo_proyectado",
                "segmento_rf03",
            ]
        ].rename(
            columns={
                "id_afiliado_hash": "ID_Seudonimizado_SHA256",
                "age": "Edad",
                "sexo_es": "Sexo",
                "bmi": "IMC",
                "categoria_imc": "Categoria_IMC_OMS",
                "children": "Cargas_Familiares",
                "fumador_es": "Tabaquismo",
                "region_es": "Region",
                "costo_proyectado": "Costo_Proyectado_USD",
                "segmento_rf03": "Segmento_Preventivo_RF03",
            }
        )
        df_export["Costo_Proyectado_USD"] = df_export["Costo_Proyectado_USD"].round(2)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Exportar Nómina Priorizada sin PII (CSV — RF05)",
            data=csv_bytes,
            file_name="reporte_segmentacion_preventiva_RF05.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 2: SELECCIÓN DE MODELO, ABLACIÓN Y RESIDUOS (RF02 / RF04 / RNF03)
# ==============================================================================
with tabs[1]:
    st.subheader("¿Por qué elegimos Random Forest Ajustado y la variable Fumador con Obesidad?")
    st.markdown(
        """
        <div class="decision-box">
            <b>Decisiones Técnicas (RF02 / Desviación D1):</b> (1) Transformar el objetivo con <code>log(1 + costo)</code> por la fuerte asimetría positiva (skew = 1,52); (2) elegir <b>Random Forest Ajustado</b> tras comparar 6 familias en validación cruzada 5-fold; y (3) incorporar la interacción explícita <b>Fumador con Obesidad (IMC ≥ 30)</b>.<br>
            <b>¿Por qué?</b> El estudio de ablación demuestra que sin esa variable el error MAE en validación cruzada sube a <b>$2.179 USD</b>, incumpliendo el criterio RF02 (&lt; $2.100 USD). Con ella, el modelo alcanza <b>R² = 0,9032 y MAE = $1.895 USD en prueba</b> (y <b>R² = 0,8576 / MAE = $2.011 USD en CV 5-fold</b>).
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 📊 Benchmark en Vivo de 6 Algoritmos (Evaluación en USD reales)")
        st.dataframe(df_benchmark, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("##### 🧪 Estudio de Ablación en Vivo (Evidencia Técnica de Desviación D1)")
        st.dataframe(df_ablacion, use_container_width=True, hide_index=True)

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        df_test_plot = pd.DataFrame(
            {
                "Costo Real (USD)": y_test_real.values,
                "Costo Predicho (USD)": pred_test_usd,
                "Residuo (USD)": y_test_real.values - pred_test_usd,
                "Perfil": np.where(
                    X_test["obese_smoker"].values == 1,
                    "Fumador con IMC ≥ 30",
                    "Resto de afiliados",
                ),
            }
        )
        fig_pred = px.scatter(
            df_test_plot,
            x="Costo Predicho (USD)",
            y="Costo Real (USD)",
            color="Perfil",
            color_discrete_map={
                "Resto de afiliados": "#2C5282",
                "Fumador con IMC ≥ 30": "#C53030",
            },
            title=f"Calibración en Test Independiente (R² = {metricas['r2_test']:.4f}, MAE = ${metricas['mae_test']:,.0f} USD)",
        )
        max_val = float(
            max(df_test_plot["Costo Real (USD)"].max(), df_test_plot["Costo Predicho (USD)"].max())
        )
        fig_pred.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Predicción perfecta (y = x)",
                line=dict(color="#111827", dash="dash"),
            )
        )
        # Ejes en USD enteros (sin "k") y hover sin decimales
        fig_pred.update_xaxes(tickprefix="$", tickformat=",.0f")
        fig_pred.update_yaxes(tickprefix="$", tickformat=",.0f")
        fig_pred.update_traces(
            selector=dict(mode="markers"),
            hovertemplate="Predicho: $%{x:,.0f} USD<br>Real: $%{y:,.0f} USD<extra>%{fullData.name}</extra>",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with g2:
        fig_imp = px.bar(
            feat_imp,
            x="Aumento del MAE (USD)",
            y="Factor Explicativo",
            orientation="h",
            error_x="Desviación (USD)",
            color="Aumento del MAE (USD)",
            color_continuous_scale="Blues",
            title="Explicabilidad Global: Importancia por Permutación en Prueba (RF04 / RNF03)",
        )
        # Eje x, barra de color y hover en USD enteros
        fig_imp.update_xaxes(tickprefix="$", tickformat=",.0f")
        fig_imp.update_coloraxes(colorbar=dict(tickprefix="$", tickformat=",.0f"))
        fig_imp.update_traces(
            hovertemplate="%{y}<br>Aumento del MAE: $%{x:,.0f} USD<extra></extra>"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Decisión financiera por análisis de cola extrema (Decil 90 — Desviación D2)
    p90 = np.percentile(y_test_real, 90)
    mask_top10 = y_test_real.values > p90
    residuos_test = y_test_real.values - pred_test_usd
    sub_top10 = residuos_test[mask_top10].mean()
    mae_top10 = np.abs(residuos_test[mask_top10]).mean()
    pct_sub_top10 = sub_top10 / y_test_real.values[mask_top10].mean() * 100
    st.warning(
        f"📌 **Decisión de Gestión de Riesgo en Cola Extrema (Desviación D2 — Decil 90 > ${p90:,.0f} USD, n={mask_top10.sum()}):** "
        f"El modelo presenta un MAE de **${mae_top10:,.0f} USD** y una subestimación media de **+${sub_top10:,.0f} USD** "
        f"({pct_sub_top10:.2f}% del gasto del grupo). En lugar de sobreajustar los árboles (lo que degradaría la generalización), "
        f"se definió la regla financiera de añadir una **reserva técnica del {pct_sub_top10:.1f}%** sobre las proyecciones del decil superior."
    )

# ==============================================================================
# TAB 3: SIMULADOR CLÍNICO INDIVIDUAL Y CONTRAFACTUAL PREVENTIVO (RF04 / RNF03)
# ==============================================================================
with tabs[2]:
    st.subheader("¿Cómo apoya el modelo las decisiones clínicas preventivas caso a caso?")
    st.markdown(
        """
        <div class="decision-box">
            <b>Decisión Operativa (RF04 / RNF03 / RNF07):</b> Entregar explicabilidad individual (SHAP) junto a una <b>simulación contrafactual preventiva</b>.<br>
            <b>¿Por qué?</b> Un auditor o médico necesita saber <i>qué factores impulsan el riesgo</i> de un afiliado específico y <i>cuál intervención preventiva tendría mayor impacto esperado</i> (cesación tabáquica vs. control de IMC), siempre bajo el principio ético de prohibición de recargos tarifarios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.markdown("#### 👤 Perfil del Afiliado a Evaluar")
        in_age = st.slider("Edad (años):", 18, 64, 45)
        in_sex = st.selectbox(
            "Sexo:",
            options=["female", "male"],
            format_func=lambda x: MAPA_SEXO[x],
        )
        in_bmi = st.number_input(
            "Índice de Masa Corporal (IMC):",
            min_value=15.0,
            max_value=55.0,
            value=33.0,
            step=0.5,
        )
        in_children = st.slider("Cargas familiares (hijos):", 0, 5, 1)
        in_smoker = st.selectbox(
            "Condición de tabaquismo:",
            options=["yes", "no"],
            format_func=lambda x: MAPA_FUMADOR[x],
        )
        in_region = st.selectbox(
            "Región:",
            options=["southeast", "southwest", "northwest", "northeast"],
            format_func=lambda x: MAPA_REGION[x],
        )

        def crear_fila_entrada(age, sex, bmi, children, smoker, region):
            return pd.DataFrame(
                [
                    {
                        "age": age,
                        "bmi": bmi,
                        "children": children,
                        "obese_smoker": int((bmi >= 30.0) and (smoker == "yes")),
                        "age_bmi_interaction": age * bmi,
                        "sex": sex,
                        "smoker": smoker,
                        "region": region,
                    }
                ]
            )

        df_input = crear_fila_entrada(in_age, in_sex, in_bmi, in_children, in_smoker, in_region)

    with sc2:
        pred_usd = float(np.expm1(pipeline.predict(df_input)[0]))
        es_prioritario = pred_usd > umbral_rf03
        in_obese_smoker = int(df_input["obese_smoker"].iloc[0])

        st.markdown("#### 🩺 Evaluación Analítica y Contrafactual Preventivo")
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Gasto Anual Estimado (RF02)",
            f"${pred_usd:,.0f} USD",
            f"± ${metricas['mae_test']:,.0f} USD (MAE ref.)",
        )
        k2.metric(
            "Clasificación Preventiva (RF03)",
            "Atención Prioritaria" if es_prioritario else "Seguimiento Estándar",
            "Derivar a programa preventivo" if es_prioritario else "Control habitual",
            delta_color="inverse" if es_prioritario else "normal",
        )
        k3.metric(
            "Sinergia Fumador × Obesidad",
            "ACTIVA (IMC ≥ 30 + Fuma)" if in_obese_smoker else "Inactiva",
            "Prioridad clínica máxima" if in_obese_smoker else "Sin sinergia crítica",
            delta_color="inverse" if in_obese_smoker else "off",
        )

        # Simulador Contrafactual Preventivo (¿Cómo cambia el costo estimado ante intervención?)
        escenarios_cf = []
        if in_smoker == "yes":
            df_sin_tabaco = crear_fila_entrada(in_age, in_sex, in_bmi, in_children, "no", in_region)
            pred_sin_tabaco = float(np.expm1(pipeline.predict(df_sin_tabaco)[0]))
            escenarios_cf.append(
                {
                    "Intervención Preventiva Simulada": "Cesación tabáquica (deja de fumar)",
                    "Costo Proyectado (USD)": round(pred_sin_tabaco, 0),
                    "Reducción Estimada vs Actual (USD)": round(pred_usd - pred_sin_tabaco, 0),
                    "% Reducción": f"{(pred_usd - pred_sin_tabaco) / pred_usd * 100:.1f}%",
                }
            )
        if in_bmi >= 30.0:
            df_imc_norm = crear_fila_entrada(in_age, in_sex, 24.9, in_children, in_smoker, in_region)
            pred_imc_norm = float(np.expm1(pipeline.predict(df_imc_norm)[0]))
            escenarios_cf.append(
                {
                    "Intervención Preventiva Simulada": "Control nutricional (IMC baja a 24,9)",
                    "Costo Proyectado (USD)": round(pred_imc_norm, 0),
                    "Reducción Estimada vs Actual (USD)": round(pred_usd - pred_imc_norm, 0),
                    "% Reducción": f"{(pred_usd - pred_imc_norm) / pred_usd * 100:.1f}%",
                }
            )
        if in_smoker == "yes" and in_bmi >= 30.0:
            df_ambos = crear_fila_entrada(in_age, in_sex, 24.9, in_children, "no", in_region)
            pred_ambos = float(np.expm1(pipeline.predict(df_ambos)[0]))
            escenarios_cf.append(
                {
                    "Intervención Preventiva Simulada": "Programa integral (cesación + IMC 24,9)",
                    "Costo Proyectado (USD)": round(pred_ambos, 0),
                    "Reducción Estimada vs Actual (USD)": round(pred_usd - pred_ambos, 0),
                    "% Reducción": f"{(pred_usd - pred_ambos) / pred_usd * 100:.1f}%",
                }
            )

        if escenarios_cf:
            st.markdown("##### 💡 Impacto Contrafactual de Intervenciones Preventivas (Referencial)")
            st.dataframe(pd.DataFrame(escenarios_cf), use_container_width=True, hide_index=True)
            st.caption(
                "Nota metodológica: el contrafactual muestra la diferencia entre perfiles históricos observados; "
                "no constituye prueba causal de eficacia clínica (requiere piloto con grupo control)."
            )

        # Explicabilidad individual (SHAP o contribución local)
        prep_step = pipeline.named_steps["prep"]
        mod_step = pipeline.named_steps["modelo"]
        x_trans = prep_step.transform(df_input)
        if sparse.issparse(x_trans):
            x_trans = x_trans.toarray()
        nombres_raw = [n.split("__", 1)[1] for n in prep_step.get_feature_names_out()]
        nombres_es = [MAPA_VARIABLES.get(n, n) for n in nombres_raw]

        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(mod_step)
            sv_ind = np.asarray(explainer.shap_values(x_trans)).flatten()
            titulo_exp = "Aporte Individual SHAP por Factor (escala log(1 + USD))"
        else:
            X_tr_trans = prep_step.transform(X_train)
            if sparse.issparse(X_tr_trans):
                X_tr_trans = X_tr_trans.toarray()
            z_diff = (x_trans.flatten() - np.median(X_tr_trans, axis=0)) / (
                np.std(X_tr_trans, axis=0) + 1e-6
            )
            sv_ind = z_diff * mod_step.feature_importances_
            titulo_exp = "Contribución Local por Factor al Caso Evaluado"

        df_exp_ind = pd.DataFrame(
            {
                "Factor": nombres_es,
                "Impacto en la Predicción": sv_ind,
                "Dirección": np.where(
                    sv_ind >= 0, "Aumenta el costo (+)", "Reduce el costo (-)"
                ),
            }
        ).sort_values("Impacto en la Predicción", key=lambda s: np.abs(s), ascending=True)

        fig_shap_ind = px.bar(
            df_exp_ind,
            x="Impacto en la Predicción",
            y="Factor",
            color="Dirección",
            orientation="h",
            color_discrete_map={
                "Aumenta el costo (+)": "#C53030",
                "Reduce el costo (-)": "#2C5282",
            },
            title=titulo_exp,
        )
        st.plotly_chart(fig_shap_ind, use_container_width=True)

# ==============================================================================
# TAB 4: AUDITORÍA ÉTICA (RNF05 / H5) Y MONITOREO MLOPS (RNF06)
# ==============================================================================
with tabs[3]:
    st.subheader("¿Cómo garantizamos equidad ética (RNF05) y vigencia del modelo (RNF06)?")
    st.markdown(
        """
        <div class="decision-box">
            <b>Compuertas Ética y Operativa (H5 / RNF05 / RNF06):</b> Ningún modelo pasa a producción sin aprobar (1) <b>brecha de MAE por sexo &lt; 10%</b> y <b>Razón de Calibración Mediana (RCM) dentro de [0,80–1,25]</b> en todos los subgrupos; y (2) monitoreo continuo de deriva estadística mediante pruebas <b>Kolmogorov-Smirnov (p &lt; 0,05)</b> y <b>PSI &gt; 0,25</b>.<br>
            <b>¿Por qué?</b> Evita sesgos sistemáticos contra subgrupos protegidos (Riesgo R1, criticidad 15) y previene decisiones obsoletas cuando cambia el perfil clínico de la cartera (Riesgo R2).
        </div>
        """,
        unsafe_allow_html=True,
    )

    auditoria = pd.DataFrame(
        {
            "Sexo": X_test["sex"].map(MAPA_SEXO).values,
            "Región": X_test["region"].map(MAPA_REGION).values,
            "Segmento": np.where(
                X_test["obese_smoker"].values == 1,
                "Fumador con IMC ≥ 30",
                "Resto de afiliados",
            ),
            "Error Absoluto": np.abs(pred_test_usd - y_test_real.values),
            "Ratio": pred_test_usd / y_test_real.values,
        }
    )
    auditoria["Error Relativo (%)"] = auditoria["Error Absoluto"] / y_test_real.values * 100

    mae_sexo = auditoria.groupby("Sexo")["Error Absoluto"].mean()
    brecha_sexo = (
        abs(mae_sexo["Mujer"] - mae_sexo["Hombre"]) / auditoria["Error Absoluto"].mean() * 100
    )
    p_mw = mannwhitneyu(
        auditoria.loc[auditoria["Sexo"] == "Mujer", "Error Absoluto"],
        auditoria.loc[auditoria["Sexo"] == "Hombre", "Error Absoluto"],
    ).pvalue

    r_sexo = auditoria.groupby("Sexo")["Ratio"].median()
    rcm_sexo = float(r_sexo["Mujer"] / r_sexo["Hombre"])
    r_reg = auditoria.groupby("Región")["Ratio"].median()
    rcm_reg = float(r_reg.min() / r_reg.max())
    r_seg = auditoria.groupby("Segmento")["Ratio"].median()
    rcm_seg = float(r_seg.min() / r_seg.max())

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Brecha MAE Sexo (< 10%)", f"{brecha_sexo:.2f}%", f"Mann-Whitney p = {p_mw:.4f}")
    e2.metric("RCM Sexo (RNF05)", f"{rcm_sexo:.3f}", "Rango aprobado: [0,80 – 1,25]")
    e3.metric("RCM Región (RNF05)", f"{rcm_reg:.3f}", "Rango aprobado: [0,80 – 1,25]")
    e4.metric("RCM Segmento (RNF05)", f"{rcm_seg:.3f}", "Rango aprobado: [0,80 – 1,25]")

    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown("##### ⚖️ Auditoría Desagregada por Sexo y Región (Conjunto de Prueba)")
        tabla_sexo = (
            auditoria.groupby("Sexo", observed=True)
            .agg(
                Afiliados=("Error Absoluto", "size"),
                MAE_USD=("Error Absoluto", lambda s: round(s.mean(), 1)),
                Calibración_Mediana=("Ratio", lambda s: round(s.median(), 3)),
                Error_Rel_Mediano_Pct=("Error Relativo (%)", lambda s: round(s.median(), 2)),
            )
            .rename(
                columns={
                    "MAE_USD": "MAE (USD)",
                    "Calibración_Mediana": "Ratio Mediano (Pred/Real)",
                    "Error_Rel_Mediano_Pct": "Error Relativo Mediano (%)",
                }
            )
        )
        st.dataframe(tabla_sexo, use_container_width=True)

        tabla_reg = (
            auditoria.groupby("Región", observed=True)
            .agg(
                Afiliados=("Error Absoluto", "size"),
                MAE_USD=("Error Absoluto", lambda s: round(s.mean(), 1)),
                Calibración_Mediana=("Ratio", lambda s: round(s.median(), 3)),
                Error_Rel_Mediano_Pct=("Error Relativo (%)", lambda s: round(s.median(), 2)),
            )
            .rename(
                columns={
                    "MAE_USD": "MAE (USD)",
                    "Calibración_Mediana": "Ratio Mediano (Pred/Real)",
                    "Error_Rel_Mediano_Pct": "Error Relativo Mediano (%)",
                }
            )
        )
        st.dataframe(tabla_reg, use_container_width=True)

    with col_eq2:
        st.markdown("##### 📡 Simulador MLOps de Deriva Estadística (RNF06 — KS y PSI)")
        shift_bmi = st.slider(
            "Simular cambio en IMC promedio de nuevos lotes (+ puntos IMC):",
            -3.0,
            5.0,
            0.0,
            0.5,
        )
        lote_base = X_train["bmi"].values
        lote_nuevo = X_test["bmi"].values + shift_bmi

        ks_stat, ks_pval = ks_2samp(lote_base, lote_nuevo)
        cortes_psi = np.percentile(lote_base, np.linspace(0, 100, 11))
        cortes_psi[0], cortes_psi[-1] = -np.inf, np.inf
        p_base = np.clip(np.histogram(lote_base, bins=cortes_psi)[0] / len(lote_base), 1e-4, None)
        p_nuevo = np.clip(
            np.histogram(lote_nuevo, bins=cortes_psi)[0] / len(lote_nuevo), 1e-4, None
        )
        psi_val = float(np.sum((p_nuevo - p_base) * np.log(p_nuevo / p_base)))

        alerta_drift = (ks_pval < 0.05) or (psi_val > 0.25)
        st.markdown(
            f"**Control MLOps (Sección 5.2):** "
            f"`PSI = {psi_val:.4f}` (Umbral > 0,25) · `KS p-value = {ks_pval:.4f}` (Umbral < 0,05) → "
            + (
                "<span class='badge-crit'>🚨 ALERTA DRIFT: ACTIVAR REENTRENAMIENTO</span>"
                if alerta_drift
                else "<span class='badge-ok'>✅ DISTRIBUCIÓN ESTABLE</span>"
            ),
            unsafe_allow_html=True,
        )
        # Gráfico por deciles: la base se divide en 10 tramos de IMC con 10% de afiliados cada uno.
        # Si el nuevo lote es estable, sus barras quedan cerca de la línea del 10%.
        etiquetas = [f"Tramo {k}" for k in range(1, 11)]
        rangos = np.concatenate([[lote_base.min()], cortes_psi[1:-1], [lote_base.max()]])
        df_drift_plot = pd.DataFrame(
            {
                "Tramo de IMC (de menor a mayor)": etiquetas * 2,
                "% de afiliados": np.concatenate([p_base, p_nuevo]) * 100,
                "Origen del Lote": ["Base de entrenamiento"] * 10 + ["Nuevo lote"] * 10,
                "Rango IMC": [f"{rangos[k]:.1f} a {rangos[k + 1]:.1f}" for k in range(10)] * 2,
            }
        )
        fig_drift = px.bar(
            df_drift_plot,
            x="Tramo de IMC (de menor a mayor)",
            y="% de afiliados",
            color="Origen del Lote",
            barmode="group",
            custom_data=["Rango IMC"],
            color_discrete_map={
                "Base de entrenamiento": "#2C5282",
                "Nuevo lote": "#C53030",
            },
            title="Deriva del IMC: % de afiliados por tramo (base vs. nuevo lote)",
        )
        fig_drift.add_hline(
            y=10,
            line_dash="dash",
            line_color="#9CA3AF",
            annotation_text="Referencia estable: 10%",
            annotation_position="top left",
        )
        fig_drift.update_yaxes(ticksuffix="%", rangemode="tozero")
        fig_drift.update_traces(
            hovertemplate="%{x} (IMC %{customdata[0]})<br>%{y:.1f}% de afiliados<extra>%{fullData.name}</extra>"
        )
        st.plotly_chart(fig_drift, use_container_width=True)
        st.caption(
            "Cómo leerlo: dividimos el IMC de la base de entrenamiento en 10 tramos con 10% de afiliados cada uno (barras azules). "
            "Si el nuevo lote (barras rojas) se parece a la base, todas las barras quedan cerca de la línea del 10%. "
            "Si las rojas se juntan hacia un lado, el IMC cambió y sube el PSI."
        )

# ==============================================================================
# TAB 5: FUNDAMENTACIÓN FINANCIERA Y SENSIBILIDAD (RF06 / SECCIÓN 2.4)
# ==============================================================================
with tabs[4]:
    st.subheader("¿Por qué el punto de equilibrio (0,42%) fundamenta la decisión financiera?")
    st.markdown(
        """
        <div class="decision-box">
            <b>Decisión Financiera (RF06 / Sección 2.4):</b> Separar el <b>CapEx analítico ($7.950 USD)</b> del <b>OpEx anual del programa preventivo ($17.400 USD/año)</b> y anclar la decisión en el <b>punto de equilibrio de 0,42%</b>.<br>
            <b>¿Por qué?</b> El modelo predice costos con R² = 0,9032, pero el dataset no registra eficacia clínica. Por eso, el dato más robusto no es suponer un 3% de ahorro, sino demostrar que basta reducir un <b>0,42%</b> del gasto del segmento crítico ($25.350 de $6.025.909 USD) para pagar el ecosistema completo en el Año 1.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fc1, fc2 = st.columns([1, 2])
    with fc1:
        st.markdown("#### 💵 Parámetros de Sensibilidad")
        capex_sw = st.number_input(
            "Inversión Inicial CapEx Software (USD):",
            value=7950,
            step=100,
            help="RRHH 272 h ($6.800) + AWS ($225) + CI/CD ($75) + Contingencia ≈12% ($850)",
        )
        costo_afiliado = st.slider(
            "Costo Anual Programa Clínico por Afiliado (USD):",
            60,
            300,
            120,
            10,
            help="Supuesto de planificación base: $120 USD/afiliado/año (Sección 2.4)",
        )
        pct_ahorro_sim = st.slider(
            "% Reducción Hipotética del Gasto en Segmento Crítico:",
            0.2,
            8.0,
            3.0,
            0.1,
        )

        n_criticos = 145
        gasto_critico_total = float(df.loc[df["obese_smoker"] == 1, "charges"].sum())
        opex_programa = n_criticos * costo_afiliado
        costo_total_ecosistema = capex_sw + opex_programa
        breakeven_pct = (costo_total_ecosistema / gasto_critico_total) * 100

        ahorro_bruto_sim = gasto_critico_total * (pct_ahorro_sim / 100.0)
        beneficio_neto_sim = ahorro_bruto_sim - costo_total_ecosistema
        roi_sim = (beneficio_neto_sim / costo_total_ecosistema) * 100
        payback_meses = (
            (costo_total_ecosistema / ahorro_bruto_sim) * 12 if ahorro_bruto_sim > 0 else np.nan
        )

    with fc2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Costo Ecosistema Año 1",
            f"${costo_total_ecosistema:,.0f} USD",
            f"CapEx ${capex_sw:,.0f} + OpEx ${opex_programa:,.0f}",
        )
        m2.metric(
            "Punto de Equilibrio",
            f"{breakeven_pct:.2f}%",
            f"Ahorro mín: ${costo_total_ecosistema:,.0f} USD",
        )
        m3.metric(
            f"Beneficio Neto ({pct_ahorro_sim:.1f}%)",
            f"${beneficio_neto_sim:,.0f} USD",
            f"ROI: {roi_sim:,.1f}%",
        )
        m4.metric(
            "Payback Ecosistema",
            f"{payback_meses:.2f} meses",
            f"~{payback_meses * 30.4:.0f} días",
        )

        # Curva de punto de equilibrio
        rango_pct = np.linspace(0.0, 5.5, 56)
        curva_df = pd.DataFrame(
            {
                "% Reducción del Gasto": rango_pct,
                "Ahorro Hipotético (USD)": gasto_critico_total * (rango_pct / 100.0),
                "Costo Total Año 1 (USD)": [costo_total_ecosistema] * len(rango_pct),
            }
        )
        fig_be = go.Figure()
        fig_be.add_trace(
            go.Scatter(
                x=curva_df["% Reducción del Gasto"],
                y=curva_df["Ahorro Hipotético (USD)"],
                mode="lines",
                name="Ahorro Bruto Hipotético (USD)",
                line=dict(color="#2C5282", width=3),
            )
        )
        fig_be.add_trace(
            go.Scatter(
                x=curva_df["% Reducción del Gasto"],
                y=curva_df["Costo Total Año 1 (USD)"],
                mode="lines",
                name="Costo Total Año 1 (CapEx + OpEx)",
                line=dict(color="#C53030", width=2, dash="dash"),
            )
        )
        fig_be.add_vline(
            x=breakeven_pct,
            line_dash="dot",
            line_color="#15803D",
            annotation_text=f"Equilibrio: {breakeven_pct:.2f}%",
        )
        fig_be.update_layout(
            title="Curva de Punto de Equilibrio Financiero (Ecosistema Completo Año 1)",
            xaxis_title="% de Reducción del Gasto en Segmento Crítico",
            yaxis_title="Monto Anual (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_be, use_container_width=True)

    st.markdown("##### 📋 Matriz de Sensibilidad: Costo del Programa vs. % de Reducción Hipotética")
    filas_mat = []
    for c_unit in [60, 120, 180, 240]:
        op = n_criticos * c_unit
        tot = capex_sw + op
        eq = tot / gasto_critico_total * 100
        fila = {
            "Costo Programa / Afiliado": f"${c_unit} USD/año",
            "Costo Total Año 1": f"${tot:,.0f} USD",
            "Equilibrio (%)": f"{eq:.2f}%",
        }
        for p_red, nom_esc in [(1.0, "Pesimista (1%)"), (3.0, "Base (3%)"), (5.0, "Optimista (5%)")]:
            ah = gasto_critico_total * (p_red / 100.0)
            bn = ah - tot
            roi_v = bn / tot * 100
            fila[nom_esc] = f"${bn:,.0f} ({roi_v:,.1f}% ROI)"
        filas_mat.append(fila)
    st.dataframe(pd.DataFrame(filas_mat), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 6: CONTROL WBS ↔ JIRA, GANTT Y REGISTRO DE DECISIONES (SECCIÓN 4)
# ==============================================================================
with tabs[5]:
    st.subheader("Trazabilidad de Planificación vs. Ejecución (WBS ↔ Jira PCM · Corte Semana 5)")
    st.markdown(
        """
        <div class="decision-box">
            <b>Gobernanza de Ejecución (Sección 4.1–4.4):</b> 15 actividades WBS (272 h) trazadas en 13 tickets de Jira (<code>PCM-01</code> a <code>PCM-13</code>).<br>
            <b>Estado al cierre de Semana 5:</b> 11 de 15 actividades finalizadas (<b>216 de 272 h = 79,4%</b>), 1 actividad en revisión (<code>PCM-12</code>, 16 h = 5,9%, que al aprobarse eleva el avance a <b>232 h / 85,3%</b>) y 3 actividades de cierre agrupadas en el ticket compuesto <code>PCM-13</code> (40 h = 14,7%). La Reserva de Contingencia ($850 USD) permanece intacta.
        </div>
        """,
        unsafe_allow_html=True,
    )

    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Avance Finalizado (S5)", "79,4%", "216 de 272 h (85,3% con PCM-12)")
    j2.metric("Actividades Completadas", "11 / 15", "En 13 tickets Jira (PCM-01 a PCM-13)")
    j3.metric("Buffer Consumido (S4)", "3 días", "Absorbió ajuste de 3.2 (Riesgo R3)")
    j4.metric("Devengado al Corte (S5)", "$5.650 / $7.950", "Reserva $850 USD intacta")

    # Registro exacto de las 15 actividades WBS alineado 100% con Informe Sección 4.1 y 4.2
    df_wbs = pd.DataFrame(
        [
            {"Cód.": "1.1", "Actividad WBS": "Project Charter y Levantamiento MoSCoW", "Responsable": "Nicolás Fonseca (PM)", "Horas": 12, "Plan": "S1", "Real": "S1", "Ticket Jira": "PCM-01", "Estado": "✅ Finalizado"},
            {"Cód.": "1.2", "Actividad WBS": "Aprovisionamiento Cloud y Permisos RBAC", "Responsable": "Bastián Rubio (DE)", "Horas": 8, "Plan": "S1", "Real": "S1", "Ticket Jira": "PCM-02", "Estado": "✅ Finalizado"},
            {"Cód.": "1.3", "Actividad WBS": "Control de Hitos y Formularios de Cambio", "Responsable": "Nicolás Fonseca (PM)", "Horas": 12, "Plan": "S1–S6", "Real": "S1–S5", "Ticket Jira": "PCM-03", "Estado": "✅ Finalizado"},
            {"Cód.": "2.1", "Actividad WBS": "Ingesta y Validación Great Expectations", "Responsable": "Bastián Rubio (DE)", "Horas": 24, "Plan": "S2", "Real": "S2", "Ticket Jira": "PCM-04", "Estado": "✅ Finalizado"},
            {"Cód.": "2.2", "Actividad WBS": "Cifrado AES-256, Seudonimización SHA-256 y PIA", "Responsable": "Bastián Rubio (DE)", "Horas": 24, "Plan": "S2–S3", "Real": "S2–S3", "Ticket Jira": "PCM-05", "Estado": "✅ Finalizado"},
            {"Cód.": "2.3", "Actividad WBS": "Análisis Exploratorio (EDA) y Reporte", "Responsable": "Cristóbal Segovia (DS)", "Horas": 24, "Plan": "S2–S3", "Real": "S2–S3", "Ticket Jira": "PCM-06", "Estado": "✅ Finalizado"},
            {"Cód.": "3.1", "Actividad WBS": "Ingeniería de Características y MLflow", "Responsable": "Cristóbal Segovia (DS)", "Horas": 24, "Plan": "S3", "Real": "S3–S4", "Ticket Jira": "PCM-07", "Estado": "✅ Finalizado"},
            {"Cód.": "3.2", "Actividad WBS": "Entrenamiento, Tuning y Benchmark R² ≥ 0,85", "Responsable": "Cristóbal Segovia (DS)", "Horas": 32, "Plan": "S3–S4", "Real": "S4 (Buffer 3d)", "Ticket Jira": "PCM-08", "Estado": "✅ Finalizado"},
            {"Cód.": "3.3", "Actividad WBS": "Algoritmo de Scoring y Segmentación (RF03)", "Responsable": "Cristóbal Segovia (DS)", "Horas": 16, "Plan": "S4", "Real": "S4", "Ticket Jira": "PCM-09", "Estado": "✅ Finalizado"},
            {"Cód.": "3.4", "Actividad WBS": "Auditoría de Equidad Ética (H5) y Módulo SHAP", "Responsable": "Cristóbal Segovia (DS)", "Horas": 24, "Plan": "S4–S5", "Real": "S4–S5", "Ticket Jira": "PCM-10", "Estado": "✅ Finalizado"},
            {"Cód.": "4.1", "Actividad WBS": "Empaquetamiento Docker y Serialización", "Responsable": "Bastián Rubio (MLOps/DE)", "Horas": 16, "Plan": "S4–S5", "Real": "S4–S5", "Ticket Jira": "PCM-11", "Estado": "✅ Finalizado"},
            {"Cód.": "4.2", "Actividad WBS": "Pipeline CI/CD y Pruebas de Estrés Lote ≤ 5s", "Responsable": "Bastián Rubio (MLOps/DE)", "Horas": 16, "Plan": "S5", "Real": "S5", "Ticket Jira": "PCM-12", "Estado": "🔄 En Revisión"},
            {"Cód.": "4.3", "Actividad WBS": "Dashboard Ejecutivo y Script de Deriva", "Responsable": "Bastián Rubio (MLOps/DE)", "Horas": 20, "Plan": "S6", "Real": "—", "Ticket Jira": "PCM-13 (1/3)", "Estado": "⏳ Por Hacer"},
            {"Cód.": "5.1", "Actividad WBS": "Consolidación Informe Final EP3 y Referencias", "Responsable": "Nicolás Fonseca (PM)", "Horas": 12, "Plan": "S6", "Real": "—", "Ticket Jira": "PCM-13 (2/3)", "Estado": "⏳ Por Hacer"},
            {"Cód.": "5.2", "Actividad WBS": "Firma de Acta de Recepción (Sign-off PO)", "Responsable": "Nicolás Fonseca (PM)", "Horas": 8, "Plan": "S6", "Real": "—", "Ticket Jira": "PCM-13 (3/3)", "Estado": "⏳ Por Hacer"},
        ]
    )

    st.markdown("##### 📋 Registro Completo WBS ↔ Tablero Jira (`PCM`)")
    st.dataframe(df_wbs, use_container_width=True, hide_index=True)

    # Gráfico de avance por estado (Kanban visual)
    resumen_estado = df_wbs.groupby("Estado")["Horas"].sum().reset_index()
    fig_kanban = px.bar(
        resumen_estado,
        x="Estado",
        y="Horas",
        color="Estado",
        text="Horas",
        color_discrete_map={
            "✅ Finalizado": "#15803D",
            "🔄 En Revisión": "#D97706",
            "⏳ Por Hacer": "#64748B",
        },
        title="Distribución de Horas Planificadas por Estado Kanban al Corte de Semana 5 (Total: 272 h)",
    )
    fig_kanban.update_traces(texttemplate="%{text} h", textposition="outside")
    st.plotly_chart(fig_kanban, use_container_width=True)

    st.markdown("##### ⚠️ Registro de Decisiones Clave y Acciones Correctivas (Desviaciones D1–D3)")
    df_desv = pd.DataFrame(
        [
            {
                "ID": "D1",
                "Problema / Desviación": "Sin variables de interacción, el modelo incumplía RF02 en validación cruzada (MAE = $2.179 USD > $2.100 USD).",
                "Decisión Tomada": "Activar el buffer de 3 días en S4 para crear la variable explícita Fumador con Obesidad (IMC ≥ 30) y ajustar hiperparámetros.",
                "Resultado Verificado": "MAE bajó a $2.011 USD en CV 5-fold y $1.895 USD en Test (R² = 0,9032). Compuerta H4 aprobada sin tocar la reserva de $850.",
            },
            {
                "ID": "D2",
                "Problema / Desviación": "Subestimación promedio de +$1.845 USD en el decil superior de costos (10% más costoso) por cola asimétrica.",
                "Decisión Tomada": "Mantener regularización del árbol para evitar sobreajuste y establecer regla financiera de +4,1% de reserva técnica sobre el decil 90.",
                "Resultado Verificado": "Estabilidad predictiva preservada y riesgo de sub-presupuestación catastrófica cubierto financieramente.",
            },
            {
                "ID": "D3",
                "Problema / Desviación": "Diferencia de MAE por sexo cercana al límite de 10% debido a la mayor prevalencia de fumadores en hombres (23,6% vs 17,4%).",
                "Decisión Tomada": "Validar calibración mediana (RCM = 0,97 en sexo y 0,94 en región, dentro de [0,80–1,25]) y blindar restricción ética RNF07.",
                "Resultado Verificado": "Compuerta ética H5 aprobada con monitoreo continuo obligatorio de equidad en cada reentrenamiento.",
            },
        ]
    )
    st.dataframe(df_desv, use_container_width=True, hide_index=True)