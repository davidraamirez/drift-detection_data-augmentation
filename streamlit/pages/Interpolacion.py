import json
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose

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
        datos_lognormal = responseLognormal.content
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
    
st.header("Modelos predictivos")
st.subheader("Sarimax")
api_urlSarimax= "http://127.0.0.1:8000/Datos/Sarimax?indice=Indice&freq=M&size="+str(num)
api_urlSarimax2 = "http://127.0.0.1:8000/Plot/Datos/Sarimax?indice=Indice&freq=M&size="+str(num)
try:
    files = {'file': ('Sarimax.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseSarimax= requests.post(api_urlSarimax,files=files)
    files = {'file': ('Sarimax.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseSarimax2 = requests.post(api_urlSarimax2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseSarimax2.status_code == 200 and responseSarimax.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseSarimax2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con modelo Sarimax") 
        datos_sarimax = responseSarimax.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_sarimax))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseSarimax.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Forecaster Autoregresivos Random Forest")
api_urlAutoreg= "http://127.0.0.1:8000/Datos/ForecasterAutoreg?indice=Indice&freq=M&size="+str(num)
api_urlAutoreg2 = "http://127.0.0.1:8000/Plot/Datos/ForecasterAutoreg?indice=Indice&freq=M&size="+str(num)
try:
    files = {'file': ('AutoregRf.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseAutoreg= requests.post(api_urlAutoreg,files=files)
    files = {'file': ('AutoregRF.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseAutoreg2 = requests.post(api_urlAutoreg2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseAutoreg2.status_code == 200 and responseAutoreg.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseAutoreg2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con Modelo Autorregresivo Random Forest") 
        datos_autoreg = responseAutoreg.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_autoreg))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseAutoreg.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Forecaster Autoregresivos Ridge")
api_urlRidge= "http://127.0.0.1:8000/Datos/AutoregRidge?indice=Indice&freq=M&size="+str(num)
api_urlRidge2 = "http://127.0.0.1:8000/Plot/Datos/AutoregRidge?indice=Indice&freq=M&size="+str(num)
try:
    files = {'file': ('AutoregRidge.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseRidge= requests.post(api_urlRidge,files=files)
    files = {'file': ('AutoregRidge.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseRidge2 = requests.post(api_urlRidge2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseRidge2.status_code == 200 and responseRidge.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseRidge2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con Modelo Autorregresivo Ridge") 
        datos_ridge = responseRidge.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_ridge))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseRidge.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Prophet")
api_urlProphet= "http://127.0.0.1:8000/Datos/Prophet?indice=Indice&freq=M&size="+str(num)
api_urlProphet2 = "http://127.0.0.1:8000/Plot/Datos/Prophet?indice=Indice&freq=M&size="+str(num)
try:
    files = {'file': ('Prophet.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseProphet= requests.post(api_urlProphet,files=files)
    files = {'file': ('Prophet.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseProphet2 = requests.post(api_urlProphet2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseProphet2.status_code == 200 and responseProphet.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseProphet2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada con Modelo Prophet") 
        datos_prophet = responseProphet.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_prophet))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseProphet.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.header("Técnicas que introducen ruido")
st.subheader("Bootstraping")
api_urlBootstrap = "http://127.0.0.1:8000/Aumentar/Sampling?size="+str(num)+"&freq=M&indice=Indice"
api_urlBootstrap2 = "http://127.0.0.1:8000/Plot/Aumentar/Sampling?size="+str(num)+"&freq=M&indice=Indice"
try:
    files = {'file': ('Bootstrap.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseBootstrap= requests.post(api_urlBootstrap,files=files)
    files = {'file': ('Bootstrap.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseBootstrap2 = requests.post(api_urlBootstrap2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseBootstrap2.status_code == 200 and responseBootstrap.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseBootstrap2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada mediante barajo y añadir ruido") 
        datos_bootstrap = responseBootstrap.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_bootstrap))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseBootstrap.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.subheader("Ruido Harmónico")
api_urlHarmonico = "http://127.0.0.1:8000/Aumentar/Harmonico?freq=M&indice=Indice&size="+str(num)
api_urlHarmonico2 = "http://127.0.0.1:8000/Plot/Aumentar/Harmonico?freq=M&indice=Indice&size="+str(num)
try:
    files = {'file': ('Harmonico.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseHarmonico= requests.post(api_urlHarmonico,files=files)
    files = {'file': ('Harmonico.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseHarmonico2 = requests.post(api_urlHarmonico2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseHarmonico2.status_code == 200 and responseHarmonico.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseHarmonico2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada obtenido al añadir ruido harmónico") 
        datos_harm = responseHarmonico.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_harm))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseHarmonico.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")


    
st.subheader("Combinación Lineal")
api_urlCl = "http://127.0.0.1:8000/Aumentar/Comb_lineal?freq=M&size="+str(num)+"&indice=Indice&window_size=5"
api_urlCl2 = "http://127.0.0.1:8000/Plot/Aumentar/Comb_lineal?freq=M&size="+str(num)+"&indice=Indice&window_size=5"
try:
    files = {'file': ('Comb_lineal.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseCl= requests.post(api_urlCl,files=files)
    files = {'file': ('Comb_lineal.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseCl2 = requests.post(api_urlCl2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseCl2.status_code == 200 and responseCl.status_code ==200:
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseCl2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada mediante combinación lineal de los últimos 5 valores") 
        datos_cl = responseCl.content
        df = pd.read_csv(pd.io.common.BytesIO(datos_cl))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseCl.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    
st.header("Técnicas que se basan en descomponer la serie")
st.subheader("Descomposición aditiva en tendencia y estacionalidad")
api_urlDescomp = "http://127.0.0.1:8000/Aumentar/Descomponer?indice=Indice&freq=M&size="+str(num)+"&tipo=additive"
api_urlDescomp2 = "http://127.0.0.1:8000/Plot/Aumentar/Descomponer?indice=Indice&freq=M&size="+str(num)+"&tipo=additive"
try:
    files = {'file': ('Descomposicion.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseDesc= requests.post(api_urlDescomp,files=files)
    files = {'file': ('Descomposicion.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseDesc2 = requests.post(api_urlDescomp2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseDesc.status_code == 200 and responseDesc2.status_code ==200:
        datos_desc = responseDesc.content
        df1 =  pd.read_csv(pd.io.common.BytesIO(data))
        descomposicion = seasonal_decompose(df1[df1.columns[1]], model='additive', period=12)
        st.pyplot(descomposicion.plot())
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseDesc2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada mediante descomposición aditiva en tendencia y estacionalidad") 

        df = pd.read_csv(pd.io.common.BytesIO(datos_desc))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseDesc2.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")

    
st.subheader("Descomposición multiplicativa en tendencia y estacionalidad")
api_urlDescompM = "http://127.0.0.1:8000/Aumentar/Descomponer?indice=Indice&freq=M&size="+str(num)+"&tipo=multiplicative"
api_urlDescompM2 = "http://127.0.0.1:8000/Plot/Aumentar/Descomponer?indice=Indice&freq=M&size="+str(num)+"&tipo=multiplicative"
try:
    files = {'file': ('Descomposicion.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseDescM= requests.post(api_urlDescompM,files=files)
    files = {'file': ('Descomposicion.csv', pd.io.common.BytesIO(data), 'text/csv')}
    responseDescM2 = requests.post(api_urlDescompM2,files=files)
    # Mostrar datos si la respuesta es exitosa
    if responseDescM.status_code == 200 and responseDescM2.status_code ==200:
        datos_descM = responseDescM.content
        df1 =  pd.read_csv(pd.io.common.BytesIO(data))
        descomposicionM = seasonal_decompose(df1[df1.columns[1]], model='multiplicative', period=12)
        st.pyplot(descomposicionM.plot())
        # Leer el contenido de la imagen
        image = Image.open(BytesIO(responseDescM2.content))
        # Mostrar la imagen en la aplicación
        st.image(image, caption="Serie temporal aumentada mediante descomposición multiplicativa en tendencia y estacionalidad") 

        df = pd.read_csv(pd.io.common.BytesIO(datos_descM))
        st.dataframe(df)
    else:
        st.error(f"Error al consultar la API: {responseDescM2.text}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")