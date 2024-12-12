import io
import json

from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose

# Configurar la página
st.set_page_config(
    page_title="Interpolación",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Interpolación de los datos")
st.header("Petición Datos")
uploaded_file = st.file_uploader("Cargar un archivo CSV", type=["csv"])

if uploaded_file is not None:
    
    col1,col2 = st.columns([1,2.05])
    # Convertir el archivo a un DataFrame de pandas
    df = pd.read_csv(uploaded_file)
    df.set_index(df.columns[0],inplace=True)
    with col1:       
        # Mostrar el DataFrame cargado
        st.write("Datos cargados:", df)

    with col2:  
        st.subheader("Graficar Datos")
        # Crear un gráfico de líneas usando pandas (esto utiliza Matplotlib por detrás)
        fig, ax = plt.subplots()  # Crear un objeto figure y un eje
        df.plot(ax=ax,title="Serie temporal",figsize=(13,5))  # Graficar en el eje creado

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)  # Usamos st.pyplot para mostrar el gráfico
        
    st.header("Técnicas de Interpolación")
    num_float = st.number_input(label="Número de datos a interpolar",value=5)
    num = int(num_float)
    df_train=df[:df.shape[0]-num]
    col3,col4 = st.columns(2)
    with col3:
        st.write("Datos de entrenamiento",df_train)
    df_test=df[df.shape[0]-num:]
    with col4:
        st.write("Datos de testeo",df_test)

    # Convertir el DataFrame a CSV para enviar en el POST
    csv_data = df_train.to_csv(index=df.index.name)
    
    st.subheader("Técnicas estadísticas:")

    st.text("Media")
    api_urlMedia = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice="+df.index.name
    api_urlMedia2 = "http://127.0.0.1:8000/Plot/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice="+df.index.name

    try:
        col5,col6 = st.columns([1,3])
        files = {'file': ('Media.csv', io.StringIO(csv_data), 'text/csv')}
        responseMedia= requests.post(api_urlMedia,files=files)
        files = {'file': ('Media.csv', io.StringIO(csv_data), 'text/csv')}
        responseMedia2 = requests.post(api_urlMedia2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseMedia2.status_code == 200 and responseMedia.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseMedia2.content))
            # Mostrar la imagen en la aplicación
            datos_media = responseMedia.content
            df_media = pd.read_csv(pd.io.common.BytesIO(datos_media),index_col="Indice")
            with col5:
                st.dataframe(df_media)
            with col6:
                st.image(image, caption="Serie temporal aumentada con media") 
        else:
            st.error(f"Error al consultar la API: {responseMedia.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Mediana")
    api_urlMediana = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=2&num="+str(num)+"&freq=M&indice="+df.index.name
    api_urlMediana2 = "http://127.0.0.1:8000/Plot/Aumentar/Estadistica?tipo=2&num="+str(num)+"&freq=M&indice="+df.index.name

    try:
        col7,col8 = st.columns([1,3])
        files = {'file': ('Mediana.csv', io.StringIO(csv_data), 'text/csv')}
        responseMediana= requests.post(api_urlMediana,files=files)
        files = {'file': ('Mediana.csv', io.StringIO(csv_data), 'text/csv')}
        responseMediana2 = requests.post(api_urlMediana2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseMediana2.status_code == 200 and responseMediana.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseMediana2.content))
            
            datos_mediana = responseMediana.content
            df_mediana = pd.read_csv(pd.io.common.BytesIO(datos_mediana),index_col="Indice")
            with col7:
                st.dataframe(df_mediana)
            with col8:
                st.image(image, caption="Serie temporal aumentada con mediana")
        else:
            st.error(f"Error al consultar la API: {responseMediana.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Moda")
    api_urlModa = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=3&num="+str(num)+"&freq=M&indice="+df.index.name
    api_urlModa2 = "http://127.0.0.1:8000/Plot/Aumentar/Estadistica?tipo=3&num="+str(num)+"&freq=M&indice="+df.index.name

    try:
        col9,col10 = st.columns([1,3])
        files = {'file': ('Moda.csv', io.StringIO(csv_data), 'text/csv')}
        responseModa= requests.post(api_urlModa,files=files)
        files = {'file': ('Moda.csv', io.StringIO(csv_data), 'text/csv')}
        responseModa2 = requests.post(api_urlModa2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseModa2.status_code == 200 and responseModa.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseModa2.content))
             
            datos_moda = responseModa.content
            df_moda = pd.read_csv(pd.io.common.BytesIO(datos_moda),index_col="Indice")
            with col9:
                st.dataframe(df_moda)
            with col10:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada con moda")
                
        else:
            st.error(f"Error al consultar la API: {responseModa.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Normal")
    api_urlNormal = "http://127.0.0.1:8000/Aumentar/Normal?size="+str(num)+"&freq=M&indice="+df.index.name
    api_urlNormal2 = "http://127.0.0.1:8000/Plot/Aumentar/Normal?size="+str(num)+"&freq=M&indice="+df.index.name
    try:
        col11,col12 = st.columns([1,3])
        files = {'file': ('Normal.csv', io.StringIO(csv_data), 'text/csv')}
        responseNormal= requests.post(api_urlNormal,files=files)
        files = {'file': ('Normal.csv', io.StringIO(csv_data), 'text/csv')}
        responseNormal2 = requests.post(api_urlNormal2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseNormal2.status_code == 200 and responseNormal.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseNormal2.content))
            datos_normal = responseNormal.content
            df_normal= pd.read_csv(pd.io.common.BytesIO(datos_normal),index_col="Indice")
            with col11:
                st.dataframe(df_normal)
            with col12:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada con normal")
                
        else:
            st.error(f"Error al consultar la API: {responseNormal.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        
    st.text("Box-Muller")
    api_urlMuller = "http://127.0.0.1:8000/Aumentar/Muller?size="+str(num)+"&freq=M&indice="+df.index.name
    api_urlMuller2 = "http://127.0.0.1:8000/Plot/Aumentar/Muller?size="+str(num)+"&freq=M&indice="+df.index.name
    try:
        col13,col14 = st.columns([1,3])
        files = {'file': ('Muller.csv', io.StringIO(csv_data), 'text/csv')}
        responseMuller= requests.post(api_urlMuller,files=files)
        files = {'file': ('Muller.csv', io.StringIO(csv_data), 'text/csv')}
        responseMuller2 = requests.post(api_urlMuller2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseMuller2.status_code == 200 and responseMuller.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseMuller2.content)) 
            datos_muller = responseMuller.content
            df_muller = pd.read_csv(pd.io.common.BytesIO(datos_muller),index_col="Indice")
            with col13:
                st.dataframe(df_muller)
            with col14:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada con box-muller") 
        else:
            st.error(f"Error al consultar la API: {responseMuller.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.subheader("Modelos predictivos")
    st.text("Sarimax")
    api_urlSarimax= "http://127.0.0.1:8000/Datos/Sarimax?indice="+df.index.name+"&freq=M&size="+str(num)
    api_urlSarimax2 = "http://127.0.0.1:8000/Plot/Datos/Sarimax?indice="+df.index.name+"&freq=M&size="+str(num)
    try:
        col15,col16 = st.columns([1,3])
        files = {'file': ('Sarimax.csv', io.StringIO(csv_data), 'text/csv')}
        responseSarimax= requests.post(api_urlSarimax,files=files)
        files = {'file': ('Sarimax.csv', io.StringIO(csv_data), 'text/csv')}
        responseSarimax2 = requests.post(api_urlSarimax2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseSarimax2.status_code == 200 and responseSarimax.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseSarimax2.content))
           
            datos_sarimax = responseSarimax.content
            df_sarimax = pd.read_csv(pd.io.common.BytesIO(datos_sarimax),index_col="Indice")
            with col15:
                st.dataframe(df_sarimax)
            with col16:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada con modelo Sarimax")
        else:
            st.error(f"Error al consultar la API: {responseSarimax.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Forecaster Autoregresivos Random Forest")
    api_urlAutoreg= "http://127.0.0.1:8000/Datos/ForecasterRF?indice="+df.index.name+"&freq=M&size="+str(num)
    api_urlAutoreg2 = "http://127.0.0.1:8000/Plot/Datos/ForecasterRF?indice="+df.index.name+"&freq=M&size="+str(num)
    try:
        col17,col18 = st.columns([1,3])
        files = {'file': ('AutoregRf.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoreg= requests.post(api_urlAutoreg,files=files)
        files = {'file': ('AutoregRF.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoreg2 = requests.post(api_urlAutoreg2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseAutoreg2.status_code == 200 and responseAutoreg.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseAutoreg2.content))
           
            datos_autoreg = responseAutoreg.content
            df_RF = pd.read_csv(pd.io.common.BytesIO(datos_autoreg),index_col="Indice")
            with col17:
                st.dataframe(df_RF)
            with col18:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada con Modelo Autorregresivo Random Forest") 
        else:
            st.error(f"Error al consultar la API: {responseAutoreg.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Forecaster Autoregresivos Ridge")
    api_urlRidge= "http://127.0.0.1:8000/Datos/AutoregRidge?indice="+df.index.name+"&freq=M&size="+str(num)
    api_urlRidge2 = "http://127.0.0.1:8000/Plot/Datos/AutoregRidge?indice="+df.index.name+"&freq=M&size="+str(num)
    try:
        col19,col20 = st.columns([1,3])
    
        files = {'file': ('AutoregRidge.csv', io.StringIO(csv_data), 'text/csv')}
        responseRidge= requests.post(api_urlRidge,files=files)
        files = {'file': ('AutoregRidge.csv', io.StringIO(csv_data), 'text/csv')}
        responseRidge2 = requests.post(api_urlRidge2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseRidge2.status_code == 200 and responseRidge.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseRidge2.content))
             
            datos_ridge = responseRidge.content
            df_ridge = pd.read_csv(pd.io.common.BytesIO(datos_ridge),index_col="Indice")
            with col19:
                st.dataframe(df_ridge)
            with col20:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada con Modelo Autorregresivo Ridge")
                
        else:
            st.error(f"Error al consultar la API: {responseRidge.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Prophet")
    api_urlProphet= "http://127.0.0.1:8000/Datos/Prophet?indice="+df.index.name+"&freq=M&size="+str(num)
    api_urlProphet2 = "http://127.0.0.1:8000/Plot/Datos/Prophet?indice="+df.index.name+"&freq=M&size="+str(num)
    try:
        col21,col22 = st.columns([1,3])
        files = {'file': ('Prophet.csv', io.StringIO(csv_data), 'text/csv')}
        responseProphet= requests.post(api_urlProphet,files=files)
        files = {'file': ('Prophet.csv', io.StringIO(csv_data), 'text/csv')}
        responseProphet2 = requests.post(api_urlProphet2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseProphet2.status_code == 200 and responseProphet.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseProphet2.content))
            
            datos_prophet = responseProphet.content
            df_prophet = pd.read_csv(pd.io.common.BytesIO(datos_prophet),index_col="Indice")
            with col21:
                st.dataframe(df_prophet)
            with col22:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada con Modelo Prophet") 
        else:
            st.error(f"Error al consultar la API: {responseProphet.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.subheader("Técnicas que introducen ruido")
    st.text("Bootstraping")
    api_urlBootstrap = "http://127.0.0.1:8000/Aumentar/Sampling?size="+str(num)+"&freq=M&indice="+df.index.name
    api_urlBootstrap2 = "http://127.0.0.1:8000/Plot/Aumentar/Sampling?size="+str(num)+"&freq=M&indice="+df.index.name
    try:
        col23,col24 = st.columns([1,3])
        files = {'file': ('Bootstrap.csv', io.StringIO(csv_data), 'text/csv')}
        responseBootstrap= requests.post(api_urlBootstrap,files=files)
        files = {'file': ('Bootstrap.csv', io.StringIO(csv_data), 'text/csv')}
        responseBootstrap2 = requests.post(api_urlBootstrap2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseBootstrap2.status_code == 200 and responseBootstrap.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseBootstrap2.content))
            
            datos_bootstrap = responseBootstrap.content
            df_bootstrap = pd.read_csv(pd.io.common.BytesIO(datos_bootstrap),index_col="Indice")
            with col23:
                st.dataframe(df_bootstrap)
            with col24:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada mediante barajo y añadir ruido")
        else:
            st.error(f"Error al consultar la API: {responseBootstrap.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Ruido Harmónico")
    api_urlHarmonico = "http://127.0.0.1:8000/Aumentar/Harmonico?freq=M&indice="+df.index.name+"&size="+str(num)
    api_urlHarmonico2 = "http://127.0.0.1:8000/Plot/Aumentar/Harmonico?freq=M&indice="+df.index.name+"&size="+str(num)
    try:
        col25,col26 = st.columns([1,3])
        files = {'file': ('Harmonico.csv', io.StringIO(csv_data), 'text/csv')}
        responseHarmonico= requests.post(api_urlHarmonico,files=files)
        files = {'file': ('Harmonico.csv', io.StringIO(csv_data), 'text/csv')}
        responseHarmonico2 = requests.post(api_urlHarmonico2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseHarmonico2.status_code == 200 and responseHarmonico.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseHarmonico2.content)) 
            datos_harm = responseHarmonico.content
            df_harm = pd.read_csv(pd.io.common.BytesIO(datos_harm),index_col="Indice")
            with col25:
                st.dataframe(df_harm)
            with col26:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada obtenido al añadir ruido harmónico")
                
        else:
            st.error(f"Error al consultar la API: {responseHarmonico.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

        
    st.text("Combinación Lineal")
    api_urlCl = "http://127.0.0.1:8000/Aumentar/Comb_lineal?freq=M&size="+str(num)+"&indice="+df.index.name+"&window_size=5"
    api_urlCl2 = "http://127.0.0.1:8000/Plot/Aumentar/Comb_lineal?freq=M&size="+str(num)+"&indice="+df.index.name+"&window_size=5"
    try:
        col27,col28= st.columns([1,3])

        files = {'file': ('Comb_lineal.csv', io.StringIO(csv_data), 'text/csv')}
        responseCl= requests.post(api_urlCl,files=files)
        files = {'file': ('Comb_lineal.csv', io.StringIO(csv_data), 'text/csv')}
        responseCl2 = requests.post(api_urlCl2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseCl2.status_code == 200 and responseCl.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseCl2.content))
            datos_cl = responseCl.content
            df_cl = pd.read_csv(pd.io.common.BytesIO(datos_cl),index_col="Indice")
            with col27:
                st.dataframe(df_cl)
            with col28:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada mediante combinación lineal de los últimos 5 valores") 
        else:
            st.error(f"Error al consultar la API: {responseCl.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.subheader("Técnicas que se basan en descomponer la serie")
    st.text("Descomposición aditiva en tendencia y estacionalidad")
    api_urlDescomp = "http://127.0.0.1:8000/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=additive"
    api_urlDescomp2 = "http://127.0.0.1:8000/Plot/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=additive"
    try:
        col52,col53 =st.columns([1,1])

        col29,col30= st.columns([1,3])
        files = {'file': ('Descomposicion.csv', io.StringIO(csv_data), 'text/csv')}
        responseDesc= requests.post(api_urlDescomp,files=files)
        files = {'file': ('Descomposicion.csv', io.StringIO(csv_data), 'text/csv')}
        responseDesc2 = requests.post(api_urlDescomp2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseDesc.status_code == 200 and responseDesc2.status_code ==200:
            datos_desc = responseDesc.content
            df1 =  pd.read_csv(io.StringIO(csv_data))
            for i in range(1,len(df.columns)+1):
                if i%2==1:
                    with col52:
                        descomposicion = seasonal_decompose(df1[df1.columns[i]], model='additive', period=12)
                        st.pyplot(descomposicion.plot())
                else :
                    with col53:
                        descomposicion = seasonal_decompose(df1[df1.columns[i]], model='additive', period=12)
                        st.pyplot(descomposicion.plot())
            df_desc = pd.read_csv(pd.io.common.BytesIO(datos_desc),index_col="Indice")
            with col29:
                st.dataframe(df_desc)
            with col30:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada mediante descomposición aditiva en tendencia y estacionalidad") 
                
        else:
            st.error(f"Error al consultar la API: {responseDesc2.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

        
    st.text("Descomposición multiplicativa en tendencia y estacionalidad")
    api_urlDescompM = "http://127.0.0.1:8000/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=multiplicative"
    api_urlDescompM2 = "http://127.0.0.1:8000/Plot/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=multiplicative"
    try:
        col50,col51 =st.columns([1,1])
        col31,col32= st.columns([1,3])
        
        files = {'file': ('Descomposicion.csv', io.StringIO(csv_data), 'text/csv')}
        responseDescM= requests.post(api_urlDescompM,files=files)
        files = {'file': ('Descomposicion.csv', io.StringIO(csv_data), 'text/csv')}
        responseDescM2 = requests.post(api_urlDescompM2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseDescM.status_code == 200 and responseDescM2.status_code ==200:
            datos_descM = responseDescM.content
            df1 =  pd.read_csv(io.StringIO(csv_data))
            for i in range(1,len(df.columns)+1):
                if i%2 ==1:
                    with col50:
                        descomposicionM = seasonal_decompose(df1[df1.columns[i]], model='multiplicative', period=12)
                        st.pyplot(descomposicionM.plot())
                else:
                    with col51:
                        descomposicionM = seasonal_decompose(df1[df1.columns[i]], model='multiplicative', period=12)
                        st.pyplot(descomposicionM.plot())
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseDescM2.content))
            df_descM = pd.read_csv(pd.io.common.BytesIO(datos_descM),index_col="Indice")
            with col31:
                st.dataframe(df_descM)
            with col32:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada mediante descomposición multiplicativa en tendencia y estacionalidad") 
        else:
            st.error(f"Error al consultar la API: {responseDescM2.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.subheader("Técnicas de interpolación matemática")
    # Crear una lista de opciones
    options = ['linear', 'cubic', 'quadratic']

    # Crear el selectbox
    selected_option = st.selectbox('Seleccione un tipo de interpolacion:', options,index=0)

    st.text("Interpolación a partir de todos los datos")
    api_urlInterpol = "http://127.0.0.1:8000/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=normal&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    api_urlInterpol2 = "http://127.0.0.1:8000/Plot/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=normal&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    try:
        col33,col34= st.columns([1,3])
        files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
        responseInterpol= requests.post(api_urlInterpol,files=files)
        files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
        responseInterpol2 = requests.post(api_urlInterpol2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseInterpol.status_code == 200 and responseInterpol2.status_code ==200:
            datos_Interpol = responseInterpol.content
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseInterpol2.content))
            
            df_interpol = pd.read_csv(pd.io.common.BytesIO(datos_Interpol),index_col="Indice")
            with col33:
                st.dataframe(df_interpol)
            with col34:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada mediante interpolación "+selected_option +" de los datos") 
        else:
            st.error(f"Error al consultar la API: {responseInterpol2.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.text("Interpolación entre el máximo y el mínimo")
    api_urlInterpolM = "http://127.0.0.1:8000/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=min-max&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    api_urlInterpolM2 = "http://127.0.0.1:8000/Plot/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=min-max&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    try:
        col35,col36 = st.columns([1,3])
        files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
        responseInterpolM= requests.post(api_urlInterpolM,files=files)
        files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
        responseInterpolM2 = requests.post(api_urlInterpolM2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseInterpolM.status_code == 200 and responseInterpolM2.status_code ==200:
            datos_InterpolM = responseInterpolM.content
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseInterpolM2.content))
             
            df_interpolM = pd.read_csv(pd.io.common.BytesIO(datos_InterpolM),index_col="Indice")
            with col35:
                st.dataframe(df_interpolM)
            with col36:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal aumentada mediante interpolación "+selected_option +" de los datos entre el máximo y el mínimo")
                
        else:
            st.error(f"Error al consultar la API: {responseInterpolM2.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

    if selected_option=='linear' or selected_option=="cubic":
        st.text("Interpolación spline")
        api_urlInterpolS = "http://127.0.0.1:8000/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=spline&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
        api_urlInterpolS2 = "http://127.0.0.1:8000/Plot/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=spline&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
        try:
            col37,col38 = st.columns([1,3])
            files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
            responseInterpolS= requests.post(api_urlInterpolS,files=files)
            files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
            responseInterpolS2 = requests.post(api_urlInterpolS2,files=files)
            # Mostrar datos si la respuesta es exitosa
            if responseInterpolS.status_code == 200 and responseInterpolS2.status_code ==200:
                datos_InterpolS = responseInterpolS.content
                # Leer el contenido de la imagen
                image = Image.open(BytesIO(responseInterpolS2.content)) 
                df_interpolS = pd.read_csv(pd.io.common.BytesIO(datos_InterpolS),index_col="Indice")
                with col37:
                    st.dataframe(df_interpolS)
                with col38:
                    # Mostrar la imagen en la aplicación
                    st.image(image, caption="Serie temporal aumentada mediante interpolación spline "+selected_option )
            else:
                st.error(f"Error al consultar la API: {responseInterpolS2.text}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        

    st.header("Comparativa métodos")
    col39,col40 = st.columns([1,1])
    error_media = mean_squared_error(y_true = df_test.values,y_pred = df_media[df.shape[0]-num:].values)
    error_moda = mean_squared_error(y_true = df_test.values,y_pred = df_moda[df.shape[0]-num:].values)
    error_mediana = mean_squared_error(y_true = df_test.values,y_pred = df_mediana[df.shape[0]-num:].values)
    error_normal = mean_squared_error(y_true = df_test.values,y_pred = df_normal[df.shape[0]-num:].values)
    error_muller = mean_squared_error(y_true = df_test.values,y_pred = df_muller[df.shape[0]-num:].values)
    error_sarimax = mean_squared_error(y_true = df_test.values,y_pred = df_sarimax[df.shape[0]-num:].values)
    error_RF = mean_squared_error(y_true = df_test.values,y_pred = df_RF[df.shape[0]-num:].values)
    error_ridge = mean_squared_error(y_true = df_test.values,y_pred = df_ridge[df.shape[0]-num:].values)
    error_prophet = mean_squared_error(y_true = df_test.values,y_pred = df_prophet[df.shape[0]-num:].values)
    error_bootstrap = mean_squared_error(y_true = df_test.values,y_pred = df_bootstrap[df.shape[0]-num:].values)
    error_harm = mean_squared_error(y_true = df_test.values,y_pred = df_harm[df.shape[0]-num:].values)
    error_cl = mean_squared_error(y_true = df_test.values,y_pred = df_cl[df.shape[0]-num:].values)
    error_desc = mean_squared_error(y_true = df_test.values,y_pred = df_desc[df.shape[0]-num:].values)
    error_descM = mean_squared_error(y_true = df_test.values,y_pred = df_descM[df.shape[0]-num:].values)
    error_interpol = mean_squared_error(y_true = df_test.values,y_pred = df_interpol[df.shape[0]-num:].values)
    error_interpolM = mean_squared_error(y_true = df_test.values,y_pred = df_interpolM[df.shape[0]-num:].values)
    if selected_option!="quadratic":
        error_interpolS = mean_squared_error(y_true = df_test.values,y_pred = df_interpolS[df.shape[0]-num:].values)
        datos_error = {
            'Modelo': ['Media', 'Moda', 'Mediana', 'Normal','Muller','Sarimax','Random Forest','Ridge','Prophet','Bootstrap','Harmonico','Combinación Lineal','Descomposición aditiva','Descomposición multiplicativa','Interpolación', 'Interpolación min-max','Interpolación Spline'],
            'Error cuadrático medio': [error_media, error_moda, error_mediana, error_normal,error_muller,error_sarimax,error_RF,error_ridge,error_prophet,error_bootstrap,error_harm,error_cl,error_desc,error_descM,error_interpol, error_interpolM,error_interpolS],
        }
        # result = pd.DataFrame({
        #     'Valores Reales': df_test.values.reshape(-1),
        #     'Moda':df_moda[df.shape[0]-num:].values.reshape(-1),
        #     'Media':df_media[df.shape[0]-num:].values.reshape(-1),
        #     'Mediana': df_mediana[df.shape[0]-num:].values.reshape(-1),
        #     'Normal':df_normal[df.shape[0]-num:].values.reshape(-1),
        #     'Box Muller':df_muller[df.shape[0]-num:].values.reshape(-1),
        #     'Autorregresivos Sarimax': df_sarimax[df.shape[0]-num:].values.reshape(-1),
        #     'Forecaster Random Forest': df_RF[df.shape[0]-num:].values.reshape(-1),
        #     'Forecaster Ridge': df_ridge[df.shape[0]-num:].values.reshape(-1),
        #     'Prophet': df_prophet[df.shape[0]-num:].values.reshape(-1),
        #     'Bootstrap': df_bootstrap[df.shape[0]-num:].values.reshape(-1),
        #     'Harmonica':df_harm[df.shape[0]-num:].values.reshape(-1),
        #     'Combinación lineal':df_cl[df.shape[0]-num:].values.reshape(-1),
        #     'Descomposición aditivia':df_desc[df.shape[0]-num:].values.reshape(-1),
        #     'Descomposición multiplicativa':df_descM[df.shape[0]-num:].values.reshape(-1),
        #     'Interpolación':df_interpol[df.shape[0]-num:].values.reshape(-1),
        #     'Interpolación entre máximo y mínimo':df_interpolM[df.shape[0]-num:].values.reshape(-1),
        #     'Interpolación spline' : df_interpolS[df.shape[0]-num:].values.reshape(-1)

        # })
    else:
        datos_error = {
            'Modelo': ['Media', 'Moda', 'Mediana', 'Normal','Muller','Sarimax','Random Forest','Ridge','Prophet','Bootstrap','Harmonico','Combinación Lineal','Descomposición aditiva','Descomposición multiplicativa','Interpolación', 'Interpolación min-max'],
            'Error cuadrático medio': [error_media, error_moda, error_mediana, error_normal,error_muller,error_sarimax,error_RF,error_ridge,error_prophet,error_bootstrap,error_harm,error_cl,error_desc,error_descM,error_interpol, error_interpolM],
        }
        # result = pd.DataFrame({
        #     'Valores Reales': df_test.values.reshape(-1),
        #     'Moda':df_moda[df.shape[0]-num:].values.reshape(-1),
        #     'Media':df_media[df.shape[0]-num:].values.reshape(-1),
        #     'Mediana': df_mediana[df.shape[0]-num:].values.reshape(-1),
        #     'Normal':df_normal[df.shape[0]-num:].values.reshape(-1),
        #     'Box Muller':df_muller[df.shape[0]-num:].values.reshape(-1),
        #     'Autorregresivos Sarimax': df_sarimax[df.shape[0]-num:].values.reshape(-1),
        #     'Forecaster Random Forest': df_RF[df.shape[0]-num:].values.reshape(-1),
        #     'Forecaster Ridge': df_ridge[df.shape[0]-num:].values.reshape(-1),
        #     'Prophet': df_prophet[df.shape[0]-num:].values.reshape(-1),
        #     'Bootstrap': df_bootstrap[df.shape[0]-num:].values.reshape(-1),
        #     'Harmonica':df_harm[df.shape[0]-num:].values.reshape(-1),
        #     'Combinación lineal':df_cl[df.shape[0]-num:].values.reshape(-1),
        #     'Descomposición aditivia':df_desc[df.shape[0]-num:].values.reshape(-1),
        #     'Descomposición multiplicativa':df_descM[df.shape[0]-num:].values.reshape(-1),
        #     'Interpolación':df_interpol[df.shape[0]-num:].values.reshape(-1),
        #     'Interpolación entre máximo y mínimo':df_interpolM[df.shape[0]-num:].values.reshape(-1),
        # })
    with col39:
        df_error = pd.DataFrame(datos_error)
        st.dataframe(df_error)
   
    bottom_3 = df_error.nlargest(3, 'Error cuadrático medio')
    top_3 = df_error.nsmallest(3,'Error cuadrático medio')
    with col40:
        st.write("Las tres peores técnicas para la interpolación son: ")
        st.write(bottom_3['Modelo'].values)
        
        st.write("Las tres mejores técnicas para la interpolación son: ")
        st.write(top_3['Modelo'].values)
    
    #result.index=df_test.index
    #fig, ax = plt.subplots()  # Crear un objeto figure y un eje
    #result.plot(ax=ax,title="Serie temporal",figsize=(25,10))  # Graficar en el eje creado

    # Mostrar el gráfico en Streamlit
    #st.pyplot(fig)  # Usamos st.pyplot para mostrar el gráfico

    st.header("Conclusiones")