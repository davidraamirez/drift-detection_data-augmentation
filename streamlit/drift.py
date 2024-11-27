import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Título de la aplicación
st.title("Petición Datos")

# Entrada del usuario
api_urlDatos = "http://127.0.0.1:8000/Datos/distribucion/fin?inicio=1%2F1%2F2000&fin=1%2F1%2F2020&freq=M&distr=2&columna=valor&params=55&params=0.2"

try:
    # Hacer la llamada a la API
    responseDatos = requests.get(api_urlDatos)

    # Mostrar datos si la respuesta es exitosa
    if responseDatos.status_code == 200:
        data = responseDatos.content
        datos_csv = pd.io.common.BytesIO(data)
        df = pd.read_csv(datos_csv)
        st.dataframe(df)
            
    else:
        st.error(f"Error al consultar la API: {responseDatos.text}")
except Exception as e:
    st.error(f"Error: {str(e)}")
        
        
st.title("Graficar Datos")

# Entrada del usuario
api_urlPlot = "http://127.0.0.1:8000/Plot/distribuciones/fin?inicio=1%2F1%2F2000&fin=1%2F1%2F2020&freq=M&distr=2&columna=valor&params=55&params=0.2"

try:
    # Hacer la llamada a la API
    responsePlot = requests.get(api_urlPlot,stream=True)

        
    # Mostrar datos si la respuesta es exitosa
    if responsePlot.status_code == 200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responsePlot.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Imagen obtenida de la API") 
            
    else:
        st.error(f"Error al consultar la API: {responsePlot.text}")
except Exception as e:
    st.error(f"Error: {str(e)}")
        
st.title("Detectar Drift")

api_urlKS = "http://127.0.0.1:8000/Deteccion/KS?indice=Indice&threshold_ks=0.05&inicio=1"
api_urlJS = "http://127.0.0.1:8000/Deteccion/JS?indice=Indice&threshold_js=0.2&inicio=1"
api_urlPSI = "http://127.0.0.1:8000/Deteccion/PSI?indice=Indice&threshold_psi=2&num_bins=10&inicio=1"
api_urlPSIQ = "http://127.0.0.1:8000/Deteccion/PSI/Cuantiles?indice=Indice&threshold_psi=0.2&num_quantiles=10&inicio=1"
api_urlCUSUM = 'http://127.0.0.1:8000/Deteccion/CUSUM?indice=Indice&threshold_cusum=1.5&drift_cusum=0.5&inicio=1'
api_urlPH = 'http://127.0.0.1:8000/Deteccion/PH?indice=Indice&min_instances=30&delta=0.005&threshold=30&alpha=0.9999'

try:
    files = {'file': ('datos-distribucion-fin.csv', pd.io.common.BytesIO(data), 'text/csv')}
    # Hacer la llamada a la API
    responseKS = requests.post(api_urlKS,files=files)
    files = {'file': ('datos-distribucion-fin.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseJS = requests.post(api_urlJS,files=files)
    files = {'file': ('datos-distribucion-fin.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responsePSI = requests.post(api_urlPSI,files=files)
    files = {'file': ('datos-distribucion-fin.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responsePSIQ = requests.post(api_urlPSIQ,files=files)
    files = {'file': ('datos-distribucion-fin.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseCUSUM = requests.post(api_urlCUSUM,files=files)
    files = {'file': ('datos-distribucion-fin.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responsePH = requests.post(api_urlPH,files=files)
 
    # Mostrar datos si la respuesta es exitosa
    if responseKS.status_code == 200 and responseJS.status_code == 200 and responsePSI.status_code == 200 and responsePSIQ.status_code == 200 and responseCUSUM.status_code == 200 and responsePH.status_code == 200:
        st.text("Kolgomorov-Smirnov")
        st.json(responseKS.text)
        st.text("Jensen-Shannon")
        st.json(responseJS.text)
        st.text("Population Stability Index")
        st.json(responsePSI.text)
        st.text("Population Stability Index Quantiles")
        st.json(responsePSIQ.text)
        st.text("CUSUM")
        st.json(responseCUSUM.text)
        st.text("Page-Hinkley")
        st.json(responsePH.text)
    else:
        st.error(f"Error al consultar la API KS: "+str(responseKS.status_code)+", "+ {responseKS.text}+ 
                 ". Error al consultar la API JS: "+str(responseJS.status_code)+", "+ {responseJS.text}+
                 ". Error al consultar la API PSI: "+str(responsePSI.status_code)+", "+ {responsePSI.text}+
                 ". Error al consultar la API PSIQ: "+str(responsePSIQ.status_code)+", "+ {responsePSIQ.text}+
                 ". Error al consultar la API CUSUM: "+str(responseCUSUM.status_code)+", "+ {responseCUSUM.text}+
                 ". Error al consultar la API PH: "+str(responsePH.status_code)+", "+ {responsePH.text})
except Exception as e:
    st.error(f"Error: {str(e)}")