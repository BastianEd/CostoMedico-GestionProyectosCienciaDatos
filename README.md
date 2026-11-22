# Costos médicos y bienestar del paciente

Proyecto académico de ciencia de datos para analizar los costos médicos anuales y comparar modelos de estimación. El notebook puede ejecutarse en **Google Colab** o **VS Code**. El dashboard interactivo se ejecuta localmente desde **VS Code** con Streamlit.

## Integrantes

- Nicolás Fonseca Olivares
- Cristóbal Segovia Retamal
- Bastián Rubio Meneses

## Archivos principales

- `EP5_Implementacion_Costos_Medicos_v3.ipynb`: análisis, preparación de datos y evaluación de modelos.
- `app_dashboard_salud-v2.py`: dashboard de resultados y análisis preventivo.
- `insurance.csv`: dataset utilizado por el notebook y el dashboard.
- `requirements.txt`: dependencias de Python para el proyecto.

## Ejecutar en Google Colab

1. Abre Google Colab y carga `EP5_Implementacion_Costos_Medicos_v3.ipynb` desde tu computador.
2. Sube `insurance.csv` al entorno de Colab desde el panel **Archivos** (icono de carpeta). El archivo debe quedar en la carpeta de trabajo, junto al notebook.
3. En una celda, instala las dependencias del análisis:

   ```python
   %pip install pandas numpy matplotlib seaborn scipy scikit-learn shap joblib
   ```

4. Ejecuta todas las celdas con **Entorno de ejecución > Ejecutar todas**. Si Colab solicita reiniciar el entorno después de instalar, reinícialo y vuelve a ejecutar las celdas.

Los archivos que genera el notebook quedan en el almacenamiento temporal de la sesión de Colab. Descárgalos antes de cerrar o reiniciar el entorno si necesitas conservarlos.

## Ejecutar en VS Code

1. Abre en VS Code la carpeta del proyecto.
2. Instala las extensiones **Python** y **Jupyter** si aún no están instaladas.
3. Abre la terminal integrada y crea un entorno virtual:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Si PowerShell bloquea la activación del entorno, ejecuta `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` en esa terminal y vuelve a activar `.venv`.
4. En VS Code, selecciona el intérprete Python de `.venv` y abre `EP5_Implementacion_Costos_Medicos_v3.ipynb`.
5. Selecciona el kernel `.venv` y ejecuta todas las celdas con **Restart Kernel and Run All**.

Mantén `insurance.csv` en la carpeta principal del proyecto. El notebook genera `modelo_costos_medicos.joblib`, `metricas_modelo.json` y `reporte_segmentacion_preventiva.csv` en la carpeta de trabajo.

## Ejecutar el dashboard en VS Code

Con `.venv` activado y la terminal situada en la carpeta principal del proyecto, ejecuta:

```powershell
streamlit run app_dashboard_salud-v2.py
```

El dashboard se abrirá en el navegador, normalmente en `http://localhost:8501`. La aplicación carga `insurance.csv` y prepara los modelos al iniciar; la carga inicial puede tardar. Para detenerla, presiona `Ctrl+C` en la terminal.

## Dataset y uso de resultados

`insurance.csv` debe contener las columnas `age`, `sex`, `bmi`, `children`, `smoker`, `region` y `charges`. El procesamiento comprueba el esquema y los valores nulos, y elimina filas duplicadas.

Los resultados son exploratorios y pueden variar según los datos y las versiones de las bibliotecas. No demuestran causalidad ni constituyen recomendaciones médicas. No deben utilizarse para fijar tarifas ni excluir afiliados; consulta las secciones de ética y equidad del notebook antes de interpretarlos.
