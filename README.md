# Análisis de Data Drift en Series Temporales: Generación de Datos Sintéticos, Aumentación de Datos y Evaluación de Modelos Predictivos

Este proyecto ofrece un entorno completo para el **análisis y monitoreo de drift en series temporales**, complementado con herramientas de **generación de datos sintéticos**, **técnicas de aumentación** y **evaluación de modelos predictivos multivariantes**. 

## 🚀 Instalación

1. **Clona el repositorio**
```bash
git clone https://github.com/davidraamirez/trabajo_fin_grado.git
cd tu-repositorio
```

2. **Crea un entorno virtual e instala las dependencias:**
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Configuration
1. **Copia el archivo de entorno de ejemplo y configura tus variables personalizadas:**
```bash
cp .env.template .env
```
Edita el archivo .env con tus credenciales o parámetros necesarios, como claves de API, rutas de datos, etc.

2. **Iniciar la API local**
Arranca la API principal para que pueda ser utilizada por la aplicación en casos_uso/ u otras herramientas que dependan de ella:

```bash
fastapi dev main_api.py --port=8000
```
Esto levantará la API localmente en:
http://127.0.0.1:8000


## 💻 Interfaz de Usuario con Streamlit
Este proyecto cuenta con una interfaz web desarrollada en Streamlit, accesible localmente, que permite explorar todas las funcionalidades del sistema de forma visual y ordenada. La interfaz está dividida en distintas secciones, cada una enfocada en un aspecto clave del análisis de series temporales:

📊 **Página de Inicio: Reporte Estadístico**
Muestra un informe descriptivo y visual de las series temporales cargadas, incluyendo estadísticas básicas.

⚠️ **Detección de Drift**
Permite aplicar técnicas de detección de cambios en la distribución de los datos (drift), útiles para validar la consistencia de los modelos en el tiempo.

🧪 **Ampliación de Características**
Ofrece herramientas para aplicar técnicas de aumentación y enriquecimiento de variables sobre las series temporales, con el objetivo de mejorar el rendimiento predictivo de los modelos.

➕ **Variable Exógena**
Sección para crear e incorporar una variable exógena adicional, y comparar los resultados de predicción del modelo con y sin dicha variable, evaluando su impacto en la mejora del rendimiento.

▶️ **Ejecutar la interfaz**
Para iniciar la aplicación en tu navegador local:

```bash
streamlit run casos_uso/Report.py 
```
Esto abrirá automáticamente la interfaz en:
http://localhost:8501/


## 📁 Estructura del Proyecto
aumentacion/
Técnicas de aumentación de datos aplicadas a series temporales.

casos_uso/
Aplicación que prueba la API en los casos de uso de ampliación de características, creación de variables exógenas y detección de drift.

drift/
Modelos y métodos para la detección de drift en los datos.

ejemplos/
Ejemplos de series sintéticas, técnicas de aumentación y el dataset usado en el caso de uso de creación de variable exógena.

modelos_prediccion/
Contiene los modelos de predicción implementados.

series_sinteticas/
Modelos para la generación de series sintéticas.

main_api.py
Archivo principal de la API que centraliza el acceso a los modelos de Detección de Data Drift en Series Temporales, Generación de Datos Sintéticos, Aumentación de Datos y Evaluación de Modelos Predictivos.

