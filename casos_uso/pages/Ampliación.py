import io
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
import os

load_dotenv()
url_api = os.getenv("URL_API")

# Configurar la página
st.set_page_config(
    page_title="Ampliación",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ampliación de las características")

num_float = st.number_input(label="Número de datos a generar",value=12)
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
        
    st.header("Técnicas de Ampliación")
    
    df_train=df[:df.shape[0]-num]
    col3,col4 = st.columns(2)
    with col3:
        st.write("Datos de entrenamiento",df_train)
    df_test=df[df.shape[0]-num:]
    with col4:
        st.write("Datos de testeo",df_test)

    # Convertir el DataFrame a CSV para enviar en el POST
    csv_data = df_train.to_csv(index=df.index.name)

    api_urlMedia = "http://"+url_api+"/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice="+df.index.name
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
        
    api_urlMediana = "http://"+url_api+"/Aumentar/Estadistica?tipo=2&num="+str(num)+"&freq=M&indice="+df.index.name
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
        
    api_urlModa = "http://"+url_api+"/Aumentar/Estadistica?tipo=3&num="+str(num)+"&freq=M&indice="+df.index.name
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
        
    api_urlNormal = "http://"+url_api+"/Aumentar/Normal?size="+str(num)+"&freq=M&indice="+df.index.name
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
        
    api_urlMuller = "http://"+url_api+"/Aumentar/Muller?size="+str(num)+"&freq=M&indice="+df.index.name
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
        
    api_urlSarimax= "http://"+url_api+"/Datos/Sarimax?indice="+df.index.name+"&freq=M&size="+str(num)
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
        
    api_urlAutoregRidge= "http://"+url_api+"/Datos/ForecasterAutoreg?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=RIDGE"
    try:
        files = {'file': ('AutoregRidge.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoregRidge= requests.post(api_urlAutoregRidge,files=files)
        if responseAutoregRidge.status_code ==200:
            datos_autoreg_ridge = responseAutoregRidge.content
            df_autoreg_ridge = pd.read_csv(pd.io.common.BytesIO(datos_autoreg_ridge),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseAutoregRidge.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    api_urlAutoregDT= "http://"+url_api+"/Datos/ForecasterAutoreg?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=DT"
    try:
        files = {'file': ('AutoregDT.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoregDT= requests.post(api_urlAutoregDT,files=files)
        if responseAutoregDT.status_code ==200:
            datos_autoreg_DT = responseAutoregDT.content
            df_autoreg_DT = pd.read_csv(pd.io.common.BytesIO(datos_autoreg_DT),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseAutoregDT.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlAutoregRF= "http://"+url_api+"/Datos/ForecasterAutoreg?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=RF"
    try:
        files = {'file': ('AutoregRF.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoregRF = requests.post(api_urlAutoregRF,files=files)
        if responseAutoregRF.status_code ==200:
            datos_autoreg_RF = responseAutoregRF.content
            df_autoreg_RF = pd.read_csv(pd.io.common.BytesIO(datos_autoreg_RF),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseAutoregRF.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    api_urlAutoregGB= "http://"+url_api+"/Datos/ForecasterAutoreg?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=GB"
    try:
        files = {'file': ('AutoregGB.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoregGB= requests.post(api_urlAutoregGB,files=files)
        if responseAutoregGB.status_code ==200:
            datos_autoreg_GB = responseAutoregGB.content
            df_autoreg_GB = pd.read_csv(pd.io.common.BytesIO(datos_autoreg_GB),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseAutoregGB.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
  
    api_urlAutoregET= "http://"+url_api+"/Datos/ForecasterAutoreg?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=ET"
    try:
        files = {'file': ('AutoregET.csv', io.StringIO(csv_data), 'text/csv')}
        responseAutoregET= requests.post(api_urlAutoregET,files=files)
        if responseAutoregET.status_code ==200:
            datos_autoreg_ET = responseAutoregET.content
            df_autoreg_ET = pd.read_csv(pd.io.common.BytesIO(datos_autoreg_ET),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseAutoregET.text}") 
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        
    api_urlDirectRidge= "http://"+url_api+"/Datos/AutoregDirect?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=RIDGE"
    try:    
        files = {'file': ('AutoreDirectRidge.csv', io.StringIO(csv_data), 'text/csv')}
        responseDirectRidge= requests.post(api_urlDirectRidge,files=files)
        if responseDirectRidge.status_code ==200:
            datos_direct_ridge = responseDirectRidge.content
            df_direct_ridge = pd.read_csv(pd.io.common.BytesIO(datos_direct_ridge),index_col="Indice") 
        else:
            st.error(f"Error al consultar la API: {responseDirectRidge.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    api_urlDirectDT= "http://"+url_api+"/Datos/AutoregDirect?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=DT"
    try:    
        files = {'file': ('AutoreDirectDT.csv', io.StringIO(csv_data), 'text/csv')}
        responseDirectDT= requests.post(api_urlDirectDT,files=files)
        if responseDirectDT.status_code ==200:
            datos_direct_DT = responseDirectDT.content
            df_direct_DT = pd.read_csv(pd.io.common.BytesIO(datos_direct_DT),index_col="Indice") 
        else:
            st.error(f"Error al consultar la API: {responseDirectDT.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlDirectRF= "http://"+url_api+"/Datos/AutoregDirect?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=RF"
    try:    
        files = {'file': ('AutoreDirectRF.csv', io.StringIO(csv_data), 'text/csv')}
        responseDirectRF= requests.post(api_urlDirectRF,files=files)
        if responseDirectRF.status_code ==200:
            datos_direct_RF = responseDirectRF.content
            df_direct_RF = pd.read_csv(pd.io.common.BytesIO(datos_direct_RF),index_col="Indice") 
        else:
            st.error(f"Error al consultar la API: {responseDirectRF.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlDirectGB= "http://"+url_api+"/Datos/AutoregDirect?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=GB"
    try:    
        files = {'file': ('AutoreDirectGB.csv', io.StringIO(csv_data), 'text/csv')}
        responseDirectGB= requests.post(api_urlDirectGB,files=files)
        if responseDirectGB.status_code ==200:
            datos_direct_GB = responseDirectGB.content
            df_direct_GB = pd.read_csv(pd.io.common.BytesIO(datos_direct_GB),index_col="Indice") 
        else:
            st.error(f"Error al consultar la API: {responseDirectGB.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlDirectET= "http://"+url_api+"/Datos/AutoregDirect?indice="+df.index.name+"&freq=M&size="+str(num)+"&regresor=ET"
    try:    
        files = {'file': ('AutoreDirectET.csv', io.StringIO(csv_data), 'text/csv')}
        responseDirectET= requests.post(api_urlDirectET,files=files)
        if responseDirectET.status_code ==200:
            datos_direct_ET = responseDirectET.content
            df_direct_ET = pd.read_csv(pd.io.common.BytesIO(datos_direct_ET),index_col="Indice") 
        else:
            st.error(f"Error al consultar la API: {responseDirectET.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        
    api_urlProphet= "http://"+url_api+"/Datos/Prophet?indice="+df.index.name+"&freq=M&size="+str(num)
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
        
    api_urlBootstrap = "http://"+url_api+"/Aumentar/Sampling?size="+str(num)+"&freq=M&indice="+df.index.name
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
        
    api_urlArmonico = "http://"+url_api+"/Aumentar/Armonico?freq=M&indice="+df.index.name+"&size="+str(num)
    try:
        files = {'file': ('Armonico.csv', io.StringIO(csv_data), 'text/csv')}
        responseArmonico= requests.post(api_urlArmonico,files=files)
        if responseArmonico.status_code ==200:
            datos_Arm = responseArmonico.content
            df_Arm = pd.read_csv(pd.io.common.BytesIO(datos_Arm),index_col="Indice")
        else:
            st.error(f"Error al consultar la API: {responseArmonico.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    api_urlCl = "http://"+url_api+"/Aumentar/Comb_lineal?freq=M&size="+str(num)+"&indice="+df.index.name+"&window_size=5"
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
        
    api_urlDescomp = "http://"+url_api+"/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=additive"
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
        
    api_urlDescompM = "http://"+url_api+"/Aumentar/Descomponer?indice="+df.index.name+"&freq=M&size="+str(num)+"&tipo=multiplicative"
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

    # selected_option = 'linear'

    # api_urlInterpol = "http://"+url_api+"/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=normal&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    # try:
    #     files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
    #     responseInterpol= requests.post(api_urlInterpol,files=files)
    #     if responseInterpol.status_code == 200 :
    #         datos_Interpol = responseInterpol.content
    #         df_interpol = pd.read_csv(pd.io.common.BytesIO(datos_Interpol),index_col="Indice")
    #     else:
    #         st.error(f"Error al consultar la API: {responseInterpol.text}")
            
    # except Exception as e:
    #     st.error(f"Error: {str(e)}")
        
    # api_urlInterpolM = "http://"+url_api+"/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=min-max&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    # try:
    #     files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
    #     responseInterpolM= requests.post(api_urlInterpolM,files=files)
    #     if responseInterpolM.status_code == 200 :
    #         datos_InterpolM = responseInterpolM.content
    #         df_interpolM = pd.read_csv(pd.io.common.BytesIO(datos_InterpolM),index_col="Indice")
    #     else:
    #         st.error(f"Error al consultar la API: {responseInterpolM.text}")
            
    # except Exception as e:
    #     st.error(f"Error: {str(e)}")

    
    # api_urlInterpolS = "http://"+url_api+"/Aumentar/Interpolacion?tipo_interpolacion="+selected_option+"&tipo_array=spline&num="+str(num)+"&freq=M&indice="+df.index.name+"&s=1"
    # try:
    #     files = {'file': ('Interpol.csv', io.StringIO(csv_data), 'text/csv')}
    #     responseInterpolS= requests.post(api_urlInterpolS,files=files)
    #     # Mostrar datos si la respuesta es exitosa
    #     if responseInterpolS.status_code == 200 :
    #         datos_InterpolS = responseInterpolS.content
    #         df_interpolS = pd.read_csv(pd.io.common.BytesIO(datos_InterpolS),index_col="Indice")    
    #     else:
    #         st.error(f"Error al consultar la API: {responseInterpolS.text}")
    # except Exception as e:
    #     st.error(f"Error: {str(e)}")

    st.header("Comparativa métodos")
    col39,col40 = st.columns([1,1])
    error_media = mean_squared_error(y_true = df_test.values,y_pred = df_media[df.shape[0]-num:].values)
    error_moda = mean_squared_error(y_true = df_test.values,y_pred = df_moda[df.shape[0]-num:].values)
    error_mediana = mean_squared_error(y_true = df_test.values,y_pred = df_mediana[df.shape[0]-num:].values)
    error_normal = mean_squared_error(y_true = df_test.values,y_pred = df_normal[df.shape[0]-num:].values)
    error_muller = mean_squared_error(y_true = df_test.values,y_pred = df_muller[df.shape[0]-num:].values)
    error_sarimax = mean_squared_error(y_true = df_test.values,y_pred = df_sarimax[df.shape[0]-num:].values)
    error_autoreg_Ridge = mean_squared_error(y_true = df_test.values,y_pred = df_autoreg_ridge[df.shape[0]-num:].values)
    error_autoreg_DT = mean_squared_error(y_true = df_test.values,y_pred = df_autoreg_DT[df.shape[0]-num:].values)
    error_autoreg_RF = mean_squared_error(y_true = df_test.values,y_pred = df_autoreg_RF[df.shape[0]-num:].values)
    error_autoreg_GB = mean_squared_error(y_true = df_test.values,y_pred = df_autoreg_GB[df.shape[0]-num:].values)
    error_autoreg_ET = mean_squared_error(y_true = df_test.values,y_pred = df_autoreg_ET[df.shape[0]-num:].values)
    error_direct_Ridge = mean_squared_error(y_true = df_test.values,y_pred = df_direct_ridge[df.shape[0]-num:].values)
    error_direct_DT = mean_squared_error(y_true = df_test.values,y_pred = df_direct_DT[df.shape[0]-num:].values)
    error_direct_RF = mean_squared_error(y_true = df_test.values,y_pred = df_direct_RF[df.shape[0]-num:].values)
    error_direct_GB = mean_squared_error(y_true = df_test.values,y_pred = df_direct_GB[df.shape[0]-num:].values)
    error_direct_ET = mean_squared_error(y_true = df_test.values,y_pred = df_direct_ET[df.shape[0]-num:].values)
    error_prophet = mean_squared_error(y_true = df_test.values,y_pred = df_prophet[df.shape[0]-num:].values)
    error_bootstrap = mean_squared_error(y_true = df_test.values,y_pred = df_bootstrap[df.shape[0]-num:].values)
    error_Arm = mean_squared_error(y_true = df_test.values,y_pred = df_Arm[df.shape[0]-num:].values)
    error_cl = mean_squared_error(y_true = df_test.values,y_pred = df_cl[df.shape[0]-num:].values)
    error_desc = mean_squared_error(y_true = df_test.values,y_pred = df_desc[df.shape[0]-num:].values)
    error_descM = mean_squared_error(y_true = df_test.values,y_pred = df_descM[df.shape[0]-num:].values)
    # error_interpol = mean_squared_error(y_true = df_test.values,y_pred = df_interpol[df.shape[0]-num:].values)
    # error_interpolM = mean_squared_error(y_true = df_test.values,y_pred = df_interpolM[df.shape[0]-num:].values)
    # error_interpolS = mean_squared_error(y_true = df_test.values,y_pred = df_interpolS[df.shape[0]-num:].values)
    datos_error = {
        'Modelo': ['Media', 'Moda', 'Mediana', 'Normal','Muller','Sarimax','Forecaster con regresor lineal con penalización Ridge','Forecaster con regresor árbol de decisión','Forecaster con regresor Random Forest', 'Forecaster con regresor Gradient Boosting','Forecaster con regresor Extra Tree','Forecaster directo con regresor lineal con penalización Ridge','Forecaster directo con regresor árbol de decisión','Forecaster directo con regresor Random Forest', 'Forecaster directo con regresor Gradient Boosting','Forecaster directo con regresor Extra Tree','Prophet','Bootstrap','Ruido armónico','Combinación Lineal','Descomposición aditiva','Descomposición multiplicativa'],
        'Error cuadrático medio': [error_media, error_moda, error_mediana, error_normal,error_muller,error_sarimax,error_autoreg_Ridge,error_autoreg_DT,error_autoreg_RF,error_autoreg_GB,error_autoreg_ET,error_direct_Ridge,error_direct_DT,error_direct_RF,error_direct_GB,error_direct_ET,error_prophet,error_bootstrap,error_Arm,error_cl,error_desc,error_descM],
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
            'Forecaster Ridge': df_autoreg_ridge[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Decision Tree': df_autoreg_DT[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Random Forest': df_autoreg_RF[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Gradient Boosting': df_autoreg_GB[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Extra Tree': df_autoreg_ET[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Direct Ridge': df_direct_ridge[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Direct Decision Tree': df_direct_DT[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Direct Random Forest': df_direct_RF[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Direct Gradient Boosting': df_direct_GB[x][df.shape[0]-num:].values.reshape(-1),
            'Forecaster Direct Extra Tree': df_direct_ET[x][df.shape[0]-num:].values.reshape(-1),
            'Prophet': df_prophet[x][df.shape[0]-num:].values.reshape(-1),
            'Bootstrap': df_bootstrap[x][df.shape[0]-num:].values.reshape(-1),
            'Ruido armónico':df_Arm[x][df.shape[0]-num:].values.reshape(-1),
            'Combinación lineal':df_cl[x][df.shape[0]-num:].values.reshape(-1),
            'Descomposición aditiva':df_desc[x][df.shape[0]-num:].values.reshape(-1),
            'Descomposición multiplicativa':df_descM[x][df.shape[0]-num:].values.reshape(-1)
            # 'Interpolación':df_interpol[x][df.shape[0]-num:].values.reshape(-1),
            # 'Interpolación entre máximo y mínimo':df_interpolM[x][df.shape[0]-num:].values.reshape(-1),
            # 'Interpolación spline' : df_interpolS[x][df.shape[0]-num:].values.reshape(-1)

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
    elif 'Forecaster con regresor lineal con penalización Ridge' == top:
        datos_df = df_autoreg_ridge
    elif 'Forecaster con regresor árbol de decisión' == top:
        datos_df = df_autoreg_DT
    elif 'Forecaster con regresor Random Forest' == top:
        datos_df = df_autoreg_RF
    elif 'Forecaster con regresor Gradient Boosting' == top:
        datos_df = df_autoreg_GB
    elif 'Forecaster con regresor Extra Tree' == top:
        datos_df = df_autoreg_ET
    elif 'Forecaster directo con regresor lineal con penalización Ridge' == top:
        datos_df = df_direct_ridge
    elif 'Forecaster directo con regresor árbol de decisión' == top:
        datos_df = df_direct_DT
    elif 'Forecaster directo con regresor Random Forest' == top:
        datos_df = df_direct_RF
    elif 'Forecaster directo con regresor Gradient Boosting' == top:
        datos_df = df_direct_GB
    elif 'Forecaster directo con regresor Extra Tree' == top:
        datos_df = df_direct_ET
    elif 'Prophet' == top:
        datos_df = df_prophet
    elif 'Bootstrap' == top:
        datos_df = df_bootstrap
    elif 'Armonico' == top:
        datos_df = df_Arm
    elif 'Combinación Lineal' == top:
        datos_df = df_cl
    elif 'Descomposición aditiva' == top:
        datos_df = df_desc
    elif 'Descomposición multiplicativa' == top:
        datos_df = df_descM 
    # elif 'Interpolación' == top:
    #     datos_df = df_interpol
    # elif 'Interpolación min-max' == top:
    #     datos_df = df_interpolM
    # elif 'Interpolación Spline' == top:
    #     datos_df = df_interpolS
        
    csv_datos = datos_df.to_csv(index=df.columns[0])
    
    st.download_button(
        label="Descargar CSV con datos aumentados mediante "+top,
        data=csv_datos,
        file_name=top+'.csv',
        mime='text/csv',
        help="Haz clic para descargar el archivo CSV con los datos."
    )