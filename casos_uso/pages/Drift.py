import io
import json
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Configurar la página
st.set_page_config(
    page_title="Drift",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Técnica de detección de drift")

st.header("Petición Datos")

uploaded_file = st.file_uploader("Cargar un archivo CSV", type=["csv"])

if uploaded_file is not None:
    
    col1,col2 =st.columns([1,2.05])

    # Convertir el archivo a un DataFrame de pandas
    df = pd.read_csv(uploaded_file)
    df.set_index(df.columns[0],inplace=True)
    with col1:       
        # Mostrar el DataFrame cargado
        st.write("Datos cargados:", df)

    with col2:  
        # Crear un gráfico de líneas usando pandas (esto utiliza Matplotlib por detrás)
        fig, ax = plt.subplots()  # Crear un objeto figure y un eje
        df.plot(ax=ax,title="Serie temporal",figsize=(13,5))  # Graficar en el eje creado

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)  # Usamos st.pyplot para mostrar el gráfico
        
            
    st.header("Detectar Drift")

    api_urlKS = "http://127.0.0.1:8000/Deteccion/KS?indice="+df.index.name+"&threshold_ks=0.05&inicio=1"
    api_urlJS = "http://127.0.0.1:8000/Deteccion/JS?indice="+df.index.name+"&threshold_js=0.2&inicio=1"
    api_urlPSI = "http://127.0.0.1:8000/Deteccion/PSI?indice="+df.index.name+"&threshold_psi=1&num_bins=10&inicio=1"
    api_urlPSIQ = "http://127.0.0.1:8000/Deteccion/PSI/Cuantiles?indice="+df.index.name+"&threshold_psi=0.5&num_quantiles=10&inicio=1"
    api_urlCUSUM = 'http://127.0.0.1:8000/Deteccion/CUSUM?indice='+df.index.name+'&threshold_cusum=1.5&drift_cusum=0.5&inicio=1'
    api_urlPH = 'http://127.0.0.1:8000/Deteccion/PH?indice='+df.index.name+'&min_instances=30&delta=0.005&threshold=10&alpha=0.9999'
    col3,col4 =st.columns(2)
    # Convertir el DataFrame a CSV para enviar en el POST
    csv_data = df.to_csv(index=df.index.name)
    try:
        files = {'file': ('datos-distribucion-fin.csv', io.StringIO(csv_data), 'text/csv')}
        # Hacer la llamada a la API
        responseKS = requests.post(api_urlKS,files=files)
        files = {'file': ('datos-distribucion-fin.csv', io.StringIO(csv_data), 'text/csv')}
        responseJS = requests.post(api_urlJS,files=files)
        files = {'file': ('datos-distribucion-fin.csv', io.StringIO(csv_data), 'text/csv')}
        responsePSI = requests.post(api_urlPSI,files=files)
        files = {'file': ('datos-distribucion-fin.csv', io.StringIO(csv_data), 'text/csv')}
        responsePSIQ = requests.post(api_urlPSIQ,files=files)
        files = {'file': ('datos-distribucion-fin.csv', io.StringIO(csv_data), 'text/csv')}
        responseCUSUM = requests.post(api_urlCUSUM,files=files)
        files = {'file': ('datos-distribucion-fin.csv', io.StringIO(csv_data), 'text/csv')}
        responsePH = requests.post(api_urlPH,files=files)
        
        driftKS = json.loads(responseKS.text)
        driftJS = json.loads(responseJS.text)
        driftPSI = json.loads(responsePSI.text)
        driftPSIQ = json.loads(responsePSIQ.text)
        driftCUSUM = json.loads(responseCUSUM.text)
        driftPH = json.loads(responsePH.text)
    
        # Mostrar datos si la respuesta es exitosa
        if responseKS.status_code == 200 and responseJS.status_code == 200 and responsePSI.status_code == 200 and responsePSIQ.status_code == 200 and responseCUSUM.status_code == 200 and responsePH.status_code == 200:
            with col3:
                st.text("Kolgomorov-Smirnov")
                st.json(responseKS.text)
            with col4:
                st.text("Jensen-Shannon")
                st.json(responseJS.text)
            with col3:
                st.text("Population Stability Index")
                st.json(responsePSI.text)
            with col4:
                st.text("Population Stability Index Quantiles")
                st.json(responsePSIQ.text)
            with col3:
                st.text("CUSUM")
                st.json(responseCUSUM.text)
            with col4:
                st.text("Page-Hinkley")
                st.json(responsePH.text)
        
        
            if driftKS["Drift"]  == "No detectado":
                st.write("No se han detectado cambios en la distribución entre la primera mitad de los datos y la segunda mitad de los datos, según las medidas de Kolmogorov-Smirnov. Por ello, podemos intuir que la distribución de los datos se mantiene a lo largo del tiempo.")
                for x in df.columns:
                    st.write("En el caso de la columna "+x+", hemos obtenido un p-value de "+ str(driftKS["Report"][x]["p_value"]) + " que es superior al valor umbral, 0.05.")

            else :
                st.write("Se ha detectado un cambio en la distribución entre la primera mitad de los datos y la segunda mitad, gracias a las medidas de Kolmogorov-Smirnov.")
                for x in df.columns:
                    if driftKS["Report"][x]["drift_status"]==True:
                        st.write("En el caso de la columna "+x+", hemos obtenido un p-value de "+ str(driftKS["Report"][x]["p_value"]) + " que es inferior al valor umbral, 0.05. Por tanto, detectamos drift en esta columna.")
                    else:
                        st.write("En el caso de la columna "+x+", hemos obtenido un p-value de "+ str(driftKS["Report"][x]["p_value"]) + " que es superior al valor umbral, 0.05. Por tanto, NO detectamos drift en esta columna.")

            if driftJS["Drift"]  == "No detectado":
                st.write("No se han detectado cambios en la distribución entre la primera mitad de los datos y la segunda mitad de los datos, según la medida de Jensen-Shannon. Por ello, podemos intuir que la distribución de los datos se mantiene a lo largo del tiempo.")
                for x in df.columns:
                    st.write("En el caso de la columna "+x+", hemos obtenido un valor de "+ str(driftJS["Report"][x]["Jensen-Shannon"]) + " que es inferior al valor umbral, 0.2.")

            else :
                st.write("Se ha detectado un cambio en la distribución entre la primera mitad de los datos y la segunda mitad, gracias a las medidas de Kolmogorov-Smirnov.")
                for x in df.columns:
                    if driftPSI["Report"][x]["drift_status"]==True:
                        st.write("En el caso de la columna "+x+", hemos obtenido un p-value de "+ str(driftJS["Report"][x]["Jensen-Shannon"]) + " que es superior al valor umbral, 0.2. Por tanto, detectamos drift en esta columna.")
                    else:
                        st.write("En el caso de la columna "+x+", hemos obtenido un p-value de "+ str(driftJS["Report"][x]["Jensen-Shannon"]) + " que es inferior al valor umbral, 0.2. Por tanto, NO detectamos drift en esta columna.")

            if driftPSI["Drift"]  == "No detectado":
                st.write("No se han detectado cambios en la distribución entre los primeros datos (80%) y los últimos datos (20%), según el Population Stability Index. Por ello, podemos intuir que la distribución de los datos se mantiene a lo largo del tiempo.")
                for x in df.columns:
                    st.write("En el caso de la columna "+x+", hemos obtenido un Population Stability Index de "+ str(driftPSI["Report"][x]["PSI"]) + " que es inferior al valor umbral, 1.")

            else :
                st.write("Se ha detectado un cambio en la distribución entre los primeros datos (80%) y los últimos datos (20%), gracias al Population Stability Index.")
                for x in df.columns:
                    if driftPSI["Report"][x]["drift_status"]==True:
                        st.write("En el caso de la columna "+x+", hemos obtenido un Population Stability Index de "+ str(driftPSI["Report"][x]["PSI"]) + " que es superior al valor umbral, 1. Por tanto, detectamos drift en esta columna.")
                    else:
                        st.write("En el caso de la columna "+x+", hemos obtenido un Population Stability Index de "+ str(driftPSI["Report"][x]["PSI"]) + " que es inferior al valor umbral, 1. Por tanto, NO detectamos drift en esta columna.")
           
            if driftPSIQ["Drift"]  == "No detectado":
                st.write("No se han detectado cambios en la distribución entre los primeros datos (80%) y los últimos datos (20%), según el Population Stability Index con cuantiles. Por ello, podemos intuir que la distribución de los datos se mantiene a lo largo del tiempo.")
                for x in df.columns:
                    st.write("En el caso de la columna "+x+", hemos obtenido un Population Stability Index de "+ str(driftPSIQ["Report"][x]["PSI"]) + " que es inferior al valor umbral, 1.")

            else :
                st.write("Se ha detectado un cambio en la distribución entre los primeros datos (80%) y los últimos datos (20%), gracias al Population Stability Index con cuantiles.")
                for x in df.columns:
                    if driftPSIQ["Report"][x]["drift_status"]==True:
                        st.write("En el caso de la columna "+x+", hemos obtenido un Population Stability Index con cuantiles de "+ str(driftPSIQ["Report"][x]["PSI"]) + " que es superior al valor umbral, 1. Por tanto, detectamos drift en esta columna.")
                    else:
                        st.write("En el caso de la columna "+x+", hemos obtenido un Population Stability Index con cuantiles de "+ str(driftPSIQ["Report"][x]["PSI"]) + " que es inferior al valor umbral, 1. Por tanto, NO detectamos drift en esta columna.")

            if driftCUSUM["Drift"] == "No detectado":
                st.write("No se han detectado grandes desviaciones de los datos respecto a la media a lo largo del tiempo, respecto a CUSUM con valor umbral 1.5. Esto podría indicar una desviación del modelo o simplemente que el modelo sigue una distribución con cierta desviación respecto de la media.")
            else:
                st.write("Se han detectado desviaciones de los datos respecto a la media a lo largo del tiempo, gracias a CUSUM con valor umbral 1.5. Esto podría indicar una desviación del modelo o simplemente que el modelo sigue una distribución con cierta desviación respecto de la media.")
                
            if driftPH["Drift"] == "No detectado":
                st.write("No se han detectado grandes desviaciones de los datos respecto a la media a lo largo del tiempo, respecto a Page-Hinkley con valor umbral 10. Esto podría indicar una desviación del modelo o simplemente que el modelo sigue una distribución con cierta desviación respecto de la media.")
            else:
                st.write("Se han detectado desviaciones de los datos respecto a la media a lo largo del tiempo, gracias a Page-Hinkley con valor umbral 10. Esto podría indicar una desviación del modelo o simplemente que el modelo sigue una distribución con cierta desviación respecto de la media.")
            
        else:
            st.error(f"Error al consultar la API KS: "+str(responseKS.status_code)+", "+ {responseKS.text}+ 
                    ". Error al consultar la API JS: "+str(responseJS.status_code)+", "+ {responseJS.text}+
                    ". Error al consultar la API PSI: "+str(responsePSI.status_code)+", "+ {responsePSI.text}+
                    ". Error al consultar la API PSIQ: "+str(responsePSIQ.status_code)+", "+ {responsePSIQ.text}+
                    ". Error al consultar la API CUSUM: "+str(responseCUSUM.status_code)+", "+ {responseCUSUM.text}+
                    ". Error al consultar la API PH: "+str(responsePH.status_code)+", "+ {responsePH.text})
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        
