import json
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

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
    
    
st.title("Técnicas de Interpolación")
num_float = st.number_input(label="Número de datos a interpolar")
num = int(num_float)
st.header("Técnicas estadísticas:")

st.subheader("Media")
api_urlMedia = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice=Indice"
api_urlMedia2 = "http://127.0.0.1:8000/Plot/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice=Indice"

try:
    files = {'file': ('Media.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseMedia= requests.post(api_urlMedia,files=files)
    files = {'file': ('Media.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseMedia2 = requests.post(api_urlMedia2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseMedia2.status_code == 200 and responseMedia.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseMedia2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con media") 
        datos_media = responseMedia.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_media))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseMedia.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Mediana")
api_urlMediana = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=2&num="+str(num)+"&freq=M&indice=Indice"
api_urlMediana2 = "http://127.0.0.1:8000/Plot/Aumentar/Estadistica?tipo=2&num="+str(num)+"&freq=M&indice=Indice"

try:
    files = {'file': ('Mediana.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseMediana= requests.post(api_urlMediana,files=files)
    files = {'file': ('Mediana.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseMediana2 = requests.post(api_urlMediana2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseMediana2.status_code == 200 and responseMediana.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseMediana2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con mediana") 
        datos_mediana = responseMediana.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_mediana))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseMediana.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Moda")
api_urlModa = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=3&num="+str(num)+"&freq=M&indice=Indice"
api_urlModa2 = "http://127.0.0.1:8000/Plot/Aumentar/Estadistica?tipo=3&num="+str(num)+"&freq=M&indice=Indice"

try:
    files = {'file': ('Moda.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseModa= requests.post(api_urlModa,files=files)
    files = {'file': ('Moda.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseModa2 = requests.post(api_urlModa2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseModa2.status_code == 200 and responseModa.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseModa2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con moda") 
        datos_moda = responseModa.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_moda))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseModa.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Normal")
api_urlNormal = "http://127.0.0.1:8000/Aumentar/Normal?size="+str(num)+"&freq=M&indice=Indice"
api_urlNormal2 = "http://127.0.0.1:8000/Plot/Aumentar/Normal?size="+str(num)+"&freq=M&indice=Indice"
try:
    files = {'file': ('Normal.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseNormal= requests.post(api_urlNormal,files=files)
    files = {'file': ('Normal.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseNormal2 = requests.post(api_urlNormal2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseNormal2.status_code == 200 and responseNormal.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseNormal2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con normal") 
        datos_normal = responseNormal.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_normal))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseNormal.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Lognormal")
api_urlLognormal = "http://127.0.0.1:8000/Aumentar/Lognormal?size="+str(num)+"&freq=M&indice=Indice"
api_urlLognormal2 = "http://127.0.0.1:8000/Plot/Aumentar/Lognormal?size="+str(num)+"&freq=M&indice=Indice"
try:
    files = {'file': ('Lognormal.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseLognormal= requests.post(api_urlLognormal,files=files)
    files = {'file': ('Lognormal.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseLognormal2 = requests.post(api_urlLognormal2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseLognormal2.status_code == 200 and responseLognormal.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseLognormal2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con lognormal") 
        datos_lognormal = responseNormal.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_lognormal))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseLognormal.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Box-Muller")
api_urlMuller = "http://127.0.0.1:8000/Aumentar/Muller?size="+str(num)+"&freq=M&indice=Indice"
api_urlMuller2 = "http://127.0.0.1:8000/Plot/Aumentar/Muller?size="+str(num)+"&freq=M&indice=Indice"
try:
    files = {'file': ('Muller.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseMuller= requests.post(api_urlMuller,files=files)
    files = {'file': ('Muller.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseMuller2 = requests.post(api_urlMuller2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseMuller2.status_code == 200 and responseMuller.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseMuller2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con box-muller") 
        datos_muller = responseMuller.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_muller))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseMuller.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
