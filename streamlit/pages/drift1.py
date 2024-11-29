import json
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
        st.image(image, caption="Serie temporal") 
            
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
    
    driftKS = json.loads(responseKS.text)
    driftJS = json.loads(responseJS.text)
    driftPSI = json.loads(responsePSI.text)
    driftPSIQ = json.loads(responsePSIQ.text)
    driftCUSUM = json.loads(responseCUSUM.text)
    driftPH = json.loads(responsePH.text)
 
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
    
        if driftKS["Drift"]  == "No detectado" and driftJS["Drift"]  == "No detectado":
            st.write("No se han detectado cambios en la distribución entre la primera mitad de los datos y la segunda mitad de los datos, según las medidas de Kolmogorov-Smirnov y Jensen Shannon. Por ello, podemos intuir que la distribución de los datos se mantiene a lo largo del tiempo.")
        else :
            st.write("Se ha detectado un cambio en la distribución entre la primera mitad de los datos y la segunda mitad, gracias a las medidas de Kolmogorov-Smirnov y Jensen Shannon.")
            
        if driftPSI["Drift"]  == "No detectado" and driftPSIQ["Drift"]  == "No detectado":
            st.write("No se han detectado cambios en la distribución entre los primeros datos (80%) y los últimos datos (20%), según el Population Stability Index. Por ello, podemos intuir que la distribución de los datos se mantiene a lo largo del tiempo.")
        elif  driftPSI["Drift"]  == "Detectado":
            st.write("Solo se ha detectado un cambio en la distribución entre los primeros datos (80%) y los últimos datos (20%) con el Population Stability Index sin cuantiles. En el caso del uso de cuantiles, no se ha detectado un cambio. Esto indica que el cambio ha sido muy leve, por lo que se mantiene el modelo de los datos.")
        else :
            st.write("Se ha detectado un cambio en la distribución entre los primeros datos (80%) y los últimos datos (20%), gracias al Population Stability Index.")
        
        if driftCUSUM["Drift"] == "No detectado" and driftPH["Drift"]=="No detectado":
            st.write("No se han detectado grandes desviaciones de los datos respecto a la media a lo largo del tiempo, respecto a CUSUM y Page-Hinkley. Esto podría indicar una desviación del modelo o simplemente que el modelo sigue una distribución con cierta desviación respecto de la media.")
        else:
            st.write("Se han detectado desviaciones de los datos respecto a la media a lo largo del tiempo, gracias a CUSUM y Page-Hinkley. Esto podría indicar una desviación del modelo o simplemente que el modelo sigue una distribución con cierta desviación respecto de la media.")
            
    else:
        st.error(f"Error al consultar la API KS: "+str(responseKS.status_code)+", "+ {responseKS.text}+ 
                 ". Error al consultar la API JS: "+str(responseJS.status_code)+", "+ {responseJS.text}+
                 ". Error al consultar la API PSI: "+str(responsePSI.status_code)+", "+ {responsePSI.text}+
                 ". Error al consultar la API PSIQ: "+str(responsePSIQ.status_code)+", "+ {responsePSIQ.text}+
                 ". Error al consultar la API CUSUM: "+str(responseCUSUM.status_code)+", "+ {responseCUSUM.text}+
                 ". Error al consultar la API PH: "+str(responsePH.status_code)+", "+ {responsePH.text})
except Exception as e:
    st.error(f"Error: {str(e)}")
    
    