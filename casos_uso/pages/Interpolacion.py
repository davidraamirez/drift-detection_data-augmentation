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

num_float = st.number_input(label="Número de datos a interpolar",value=12)
num = int(num_float)

st.header("Petición Datos")
uploaded_file = st.file_uploader("Cargar un archivo CSV", type=["csv"])


if uploaded_file is not None :
    
    col1,col2 = st.columns([1,2.05])
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
        
    st.header("Técnicas de Interpolación")
    
    df_train=df[:df.shape[0]-num]
    col3,col4 = st.columns(2)
    with col3:
        st.write("Datos de entrenamiento",df_train)
    df_test=df[df.shape[0]-num:]
    with col4:
        st.write("Datos de testeo",df_test)

    # Convertir el DataFrame a CSV para enviar en el POST
    csv_data = df_train.to_csv(index=df.index.name)

    api_urlMedia = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice="+df.index.name
    try:
        files = {'file': ('Media.csv', io.StringIO(csv_data), 'text/csv')}
        responseMedia= requests.post(api_urlMedia,files=files)
        if responseMedia.status_code ==200:
            datos_media = responseMedia.content
            df_media = pd.read_csv(pd.io.common.BytesIO(datos_media),index_col="Indice") 
        else:
            st.error(f"Error al consultar la API: {responseMedia.text}")  
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlMediana = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=2&num="+str(num)+"&freq=M&indice="+df.index.name
    try:
        files = {'file': ('Mediana.csv', io.StringIO(csv_data), 'text/csv')}
        responseMediana= requests.post(api_urlMediana,files=files)
        if responseMediana.status_code ==200:
            datos_mediana = responseMediana.content
            df_mediana = pd.read_csv(pd.io.common.BytesIO(datos_mediana),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseMediana.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlModa = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=3&num="+str(num)+"&freq=M&indice="+df.index.name
    try:
        col9,col10 = st.columns([1,3])
        files = {'file': ('Moda.csv', io.StringIO(csv_data), 'text/csv')}
        responseModa= requests.post(api_urlModa,files=files)
        if responseModa.status_code ==200:
            datos_moda = responseModa.content
            df_moda = pd.read_csv(pd.io.common.BytesIO(datos_moda),index_col="Indice")               
        else:
            st.error(f"Error al consultar la API: {responseModa.text}")        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlNormal = "http://127.0.0.1:8000/Aumentar/Normal?size="+str(num)+"&freq=M&indice="+df.index.name
    try:
        files = {'file': ('Normal.csv', io.StringIO(csv_data), 'text/csv')}
        responseNormal= requests.post(api_urlNormal,files=files)
        if  responseNormal.status_code ==200:
            datos_normal = responseNormal.content
            df_normal= pd.read_csv(pd.io.common.BytesIO(datos_normal),index_col="Indice")      
        else:
            st.error(f"Error al consultar la API: {responseNormal.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlMuller = "http://127.0.0.1:8000/Aumentar/Muller?size="+str(num)+"&freq=M&indice="+df.index.name
    try:
        files = {'file': ('Muller.csv', io.StringIO(csv_data), 'text/csv')}
        responseMuller= requests.post(api_urlMuller,files=files)
        if  responseMuller.status_code ==200:
            datos_muller = responseMuller.content
            df_muller = pd.read_csv(pd.io.common.BytesIO(datos_muller),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseMuller.text}")  
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlSarimax= "http://127.0.0.1:8000/Datos/Sarimax?indice="+df.index.name+"&freq=M&size="+str(num)
    try:
        files = {'file': ('Sarimax.csv', io.StringIO(csv_data), 'text/csv')}
        responseSarimax= requests.post(api_urlSarimax,files=files)
        if responseSarimax.status_code ==200:
            datos_sarimax = responseSarimax.content
            df_sarimax = pd.read_csv(pd.io.common.BytesIO(datos_sarimax),index_col="Indice")   
        else:
            st.error(f"Error al consultar la API: {responseSarimax.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlAutoreg= "http://127.0.0.1:8000/Datos/ForecasterRF?indice="+df.index.name+"&freq=M&size="+str(num)
    try:
        files = {'file': ('AutoregRF.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoreg= requests.post(api_urlAutoreg,files=files)
        if responseAutoreg.status_code ==200:
            datos_autoreg = responseAutoreg.content
            df_RF = pd.read_csv(pd.io.common.BytesIO(datos_autoreg),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseAutoreg.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlRidge= "http://127.0.0.1:8000/Datos/AutoregRidge?indice="+df.index.name+"&freq=M&size="+str(num)
    try:    
        files = {'file': ('AutoregRidge.csv', io.StringIO(csv_data), 'text/csv')}
        responseRidge= requests.post(api_urlRidge,files=files)
        if responseRidge.status_code ==200:
        
            datos_ridge = responseRidge.content
            df_ridge = pd.read_csv(pd.io.common.BytesIO(datos_ridge),index_col="Indice")
               
        else:
            st.error(f"Error al consultar la API: {responseRidge.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlProphet= "http://127.0.0.1:8000/Datos/Prophet?indice="+df.index.name+"&freq=M&size="+str(num)
    try:
        files = {'file': ('Prophet.csv', io.StringIO(csv_data), 'text/csv')}
        responseProphet= requests.post(api_urlProphet,files=files)
        if responseProphet.status_code ==200:
            datos_prophet = responseProphet.content
            df_prophet = pd.read_csv(pd.io.common.BytesIO(datos_prophet),index_col="Indice")  
        else:
            st.error(f"Error al consultar la API: {responseProphet.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlBootstrap = "http://127.0.0.1:8000/Aumentar/Sampling?size="+str(num)+"&freq=M&indice="+df.index.name
    try:
        col23,col24 = st.columns([1,3])
        files = {'file': ('Bootstrap.csv', io.StringIO(csv_data), 'text/csv')}
        responseBootstrap= requests.post(api_urlBootstrap,files=files)
        if  responseBootstrap.status_code ==200:
            datos_bootstrap = responseBootstrap.content
            df_bootstrap = pd.read_csv(pd.io.common.BytesIO(datos_bootstrap),index_col="Indice")    
        else:
            st.error(f"Error al consultar la API: {responseBootstrap.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlHarmonico = "http://127.0.0.1:8000/Aumentar/Harmonico?freq=M&indice="+df.index.name+"&size="+str(num)
    try:
        files = {'file': ('Harmonico.csv', io.StringIO(csv_data), 'text/csv')}
        responseHarmonico= requests.post(api_urlHarmonico,files=files)
        if responseHarmonico.status_code ==200:
            datos_harm = responseHarmonico.content
            df_harm = pd.read_csv(pd.io.common.BytesIO(datos_harm),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseHarmonico.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlCl = "http://127.0.0.1:8000/Aumentar/Comb_lineal?freq=M&size="+str(num)+"&indice="+df.index.name+"&window_size=5"
    try:
        files = {'file': ('Comb_lineal.csv', io.StringIO(csv_data), 'text/csv')}
        responseCl= requests.post(api_urlCl,files=files)
        if  responseCl.status_code ==200:
            datos_cl = responseCl.content
            df_cl = pd.read_csv(pd.io.common.BytesIO(datos_cl),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseCl.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlDescomp = "http://127.0.0.1:8000/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=additive"
    try:
        files = {'file': ('Descomposicion.csv', io.StringIO(csv_data), 'text/csv')}
        responseDesc= requests.post(api_urlDescomp,files=files)
        if responseDesc.status_code == 200:
            datos_desc = responseDesc.content
            df_desc = pd.read_csv(pd.io.common.BytesIO(datos_desc),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseDesc.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlDescompM = "http://127.0.0.1:8000/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=multiplicative"
    try:
        files = {'file': ('Descomposicion.csv', io.StringIO(csv_data), 'text/csv')}
        responseDescM= requests.post(api_urlDescompM,files=files)
        if responseDescM.status_code == 200 :
            datos_descM = responseDescM.content
            df_descM = pd.read_csv(pd.io.common.BytesIO(datos_descM),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseDescM.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

    selected_option = 'linear'

    api_urlInterpol = "http://127.0.0.1:8000/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=normal&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    try:
        files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
        responseInterpol= requests.post(api_urlInterpol,files=files)
        if responseInterpol.status_code == 200 :
            datos_Interpol = responseInterpol.content
            df_interpol = pd.read_csv(pd.io.common.BytesIO(datos_Interpol),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseInterpol.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlInterpolM = "http://127.0.0.1:8000/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=min-max&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    try:
        files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
        responseInterpolM= requests.post(api_urlInterpolM,files=files)
        if responseInterpolM.status_code == 200 :
            datos_InterpolM = responseInterpolM.content
            df_interpolM = pd.read_csv(pd.io.common.BytesIO(datos_InterpolM),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseInterpolM.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

    
    api_urlInterpolS = "http://127.0.0.1:8000/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=spline&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    try:
        files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
        responseInterpolS= requests.post(api_urlInterpolS,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseInterpolS.status_code == 200 :
            datos_InterpolS = responseInterpolS.content
            df_interpolS = pd.read_csv(pd.io.common.BytesIO(datos_InterpolS),index_col="Indice")    
        else:
            st.error(f"Error al consultar la API: {responseInterpolS.text}")
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
    error_interpolS = mean_squared_error(y_true = df_test.values,y_pred = df_interpolS[df.shape[0]-num:].values)
    datos_error = {
        'Modelo': ['Media', 'Moda', 'Mediana', 'Normal','Muller','Sarimax','Random Forest','Ridge','Prophet','Bootstrap','Harmonico','Combinación Lineal','Descomposición aditiva','Descomposición multiplicativa','Interpolación', 'Interpolación min-max','Interpolación Spline'],
        'Error cuadrático medio': [error_media, error_moda, error_mediana, error_normal,error_muller,error_sarimax,error_RF,error_ridge,error_prophet,error_bootstrap,error_harm,error_cl,error_desc,error_descM,error_interpol, error_interpolM,error_interpolS],
    }
    for x in df.columns:
        result = pd.DataFrame({
            'Valores Reales': df_test[x].values.reshape(-1),
            'Moda':df_moda[x][df.shape[0]-num:].values.reshape(-1),
            'Media':df_media[x][df.shape[0]-num:].values.reshape(-1),
            'Mediana': df_mediana[x][df.shape[0]-num:].values.reshape(-1),
            'Normal':df_normal[x][df.shape[0]-num:].values.reshape(-1),
            'Box Muller':df_muller[x][df.shape[0]-num:].values.reshape(-1),
            'Autorregresivos Sarimax': df_sarimax[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Random Forest': df_RF[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Ridge': df_ridge[x][df.shape[0]-num:].values.reshape(-1),
            'Prophet': df_prophet[x][df.shape[0]-num:].values.reshape(-1),
            'Bootstrap': df_bootstrap[x][df.shape[0]-num:].values.reshape(-1),
            'Harmonica':df_harm[x][df.shape[0]-num:].values.reshape(-1),
            'Combinación lineal':df_cl[x][df.shape[0]-num:].values.reshape(-1),
            'Descomposición aditiva':df_desc[x][df.shape[0]-num:].values.reshape(-1),
            'Descomposición multiplicativa':df_descM[x][df.shape[0]-num:].values.reshape(-1),
            'Interpolación':df_interpol[x][df.shape[0]-num:].values.reshape(-1),
            'Interpolación entre máximo y mínimo':df_interpolM[x][df.shape[0]-num:].values.reshape(-1),
            'Interpolación spline' : df_interpolS[x][df.shape[0]-num:].values.reshape(-1)

        })
    
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
    
    top_1 = df_error.nsmallest(1,'Error cuadrático medio')
    top = top_1['Modelo'].values[0]
    st.subheader("CSV con los datos interpolados de forma óptima")
    if 'Media'== top:
        datos_df = df_media
    elif 'Moda' == top:
        datos_df = df_moda
    elif 'Mediana' == top:
        datos_df = df_mediana
    elif 'Normal' == top:
        datos_df = df_normal
    elif 'Muller' == top:
        datos_df = df_muller
    elif 'Sarimax' == top:
        datos_df = df_sarimax
    elif 'Random Forest' == top:
        datos_df = df_RF
    elif 'Ridge' == top:
        datos_df = df_ridge
    elif 'Prophet' == top:
        datos_df = df_prophet
    elif 'Bootstrap' == top:
        datos_df = df_bootstrap
    elif 'Harmonico' == top:
        datos_df = df_harm
    elif 'Combinación Lineal' == top:
        datos_df = df_cl
    elif 'Descomposición aditiva' == top:
        datos_df = df_desc
    elif 'Descomposición multiplicativa' == top:
        datos_df = df_descM 
    elif 'Interpolación' == top:
        datos_df = df_interpol
    elif 'Interpolación min-max' == top:
        datos_df = df_interpolM
    elif 'Interpolación Spline' == top:
        datos_df = df_interpolS
        
    csv_datos = datos_df.to_csv(index=df.columns[0])
    
    st.download_button(
        label="Descargar CSV con datos interpolados mediante "+top,
        data=csv_datos,
        file_name=top+'.csv',
        mime='text/csv',
        help="Haz clic para descargar el archivo CSV con los datos."
    )