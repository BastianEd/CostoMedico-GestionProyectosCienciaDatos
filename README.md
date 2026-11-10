# Análisis de costos médicos y bienestar del paciente

Proyecto académico de ciencia de datos para analizar qué características de los afiliados se relacionan con sus costos médicos anuales. El trabajo incluye exploración de datos, preparación de variables, comparación y evaluación de modelos predictivos, además de análisis ético, financiero y de riesgos.

El entregable técnico principal es el notebook [`Ev3.ipynb`](./Ev3.ipynb). El modelo estima `charges` (costos en USD) usando variables como edad, sexo, IMC, número de hijos, tabaquismo y región. Sus resultados sirven como análisis exploratorio y no como recomendación médica ni como sistema listo para producción.

## Contenido del proyecto

| Archivo | Descripción |
|---|---|
| `Ev3.ipynb` | Notebook principal: EDA, ingeniería de variables, entrenamiento, evaluación, explicabilidad y auditoría de equidad. También contiene análisis ético, financiero y de riesgos. |
| `insurance.csv` | Dataset de entrada utilizado por el notebook. |
| `modelo_costos_medicos.joblib` | Pipeline de modelo entrenado, guardado para su reutilización. |
| `metricas_modelo.json` | Resumen de hiperparámetros y métricas del modelo guardado. |
| `requirements.txt` | Dependencias Python para el análisis. |
| `Informe_Ev2_Duoc_corregido_2.docx` | Informe de factibilidad relacionado con el proyecto. |
| `EP5_Instrucciones y Pauta ENCARGO_Estudiante.pdf` | Instrucciones y pauta de evaluación del encargo. |
| `Intrucciones_Evaluacion.md` | Copia en Markdown de las instrucciones y la pauta. |

## Requisitos

- Python instalado.
- Las dependencias listadas en `requirements.txt`.
- Jupyter Notebook o JupyterLab para ejecutar el análisis de forma interactiva.

## Instalación y ejecución

Desde la carpeta del proyecto, en PowerShell:

```powershell
pip install -r requirements.txt
```

En Jupyter, abre `Ev3.ipynb` y ejecuta todas las celdas, preferentemente con **Restart Kernel and Run All Cells**. El archivo `insurance.csv` debe permanecer en la misma carpeta que el notebook.

Al finalizar, las últimas celdas guardan o actualizan `modelo_costos_medicos.joblib` y `metricas_modelo.json` en la carpeta de trabajo. El notebook incluye celdas de instalación y muestra versiones de algunas librerías para apoyar la reproducibilidad. Las versiones no están fijadas en `requirements.txt`, por lo que los resultados pueden variar entre entornos.

## Resultado registrado

El archivo de métricas disponible corresponde a un modelo **Random Forest** con 300 árboles, profundidad máxima 8, mínimo de 3 muestras por hoja y `max_features=0.5`. En la evaluación registrada:

- **MAE en test:** USD 1.894,7.
- **MAPE en test:** 16,65%.
- **R² en test:** 0,9032.
- **Validación cruzada de 5 particiones:** MAE promedio USD 2.011,42 (desviación estándar 143,54) y R² promedio 0,8575 (desviación estándar 0,0278).

Estas cifras describen la ejecución registrada en `metricas_modelo.json`; al volver a ejecutar el notebook, los valores pueden cambiar.

## Alcance y limitaciones

El notebook informa que, de 1.338 registros originales, se elimina un duplicado y se analizan 1.337 filas sin valores nulos. Las edades observadas van de 18 a 64 años, por lo que los resultados no representan a adultos mayores. Las asociaciones encontradas en los datos no implican causalidad, y las métricas por sí solas no bastan para tomar decisiones sobre afiliados. Consulte las secciones de ética y auditoría de equidad del notebook antes de interpretar el modelo.
