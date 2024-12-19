import io
from PIL import Image
from io import BytesIO
import requests
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Configurar la página
st.set_page_config(
    page_title="Aumentación",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Creación variable exógena")
st.header("Petición Datos")
uploaded_file = st.file_uploader("Cargar un archivo CSV", type=["csv"])

if uploaded_file is not None:
    
    col1,col2 = st.columns([1.65,2.05])
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
        
    csv_data = df.to_csv(index=df.index.name)
    st.subheader("Principal Component Analysis")
    api_urlPCA = "http://127.0.0.1:8000/Variables/PCA?indice="+df.index.name+"&columna=PCA"
    api_urlPCA2 = "http://127.0.0.1:8000/Plot/Variables/PCA?indice="+df.index.name+"&columna=PCA"

    try:
        col3,col4 = st.columns([2,3])
        files = {'file': ('PCA.csv', io.StringIO(csv_data), 'text/csv')}
        responsePCA= requests.post(api_urlPCA,files=files)
        files = {'file': ('PCA.csv', io.StringIO(csv_data), 'text/csv')}
        responsePCA2 = requests.post(api_urlPCA2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responsePCA2.status_code == 200 and responsePCA.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responsePCA2.content))
            datos_PCA = responsePCA.content
            df_PCA = pd.read_csv(pd.io.common.BytesIO(datos_PCA),index_col="Indice")
            with col3:
                st.dataframe(df_PCA)
            with col4:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal con aumento de variable a través de PCA") 
            st.text("Creación de una nueva variable PCA que se ha construido mediante estandarización y aplicando Principal Component Analysis sobre las variables.")    
        else:
            st.error(f"Error al consultar la API: {responsePCA.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    df.freq='M'
    df_PCA.freq='M'
    target_column = 'Air Quality Numeric'  # Columna a predecir
    exog_columns = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas']  # Variables exógenas

    # Dividir los datos en conjunto de entrenamiento y prueba
    train_size = int(df.shape[0] * 0.8)  # 80% entrenamiento, 20% prueba
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Separar variables objetivo y exógenas
    y_train = train_data[target_column]
    y_test = test_data[target_column]
    exog_train = train_data[exog_columns]
    exog_test = test_data[exog_columns]
    
    # Parámetros iniciales p, d, q, P, D, Q, s
    p, d, q = 1, 1, 1  # Parámetros no estacionales
    P, D, Q, s = 1, 1, 1, 12  # Parámetros estacionales (suponiendo datos mensuales, ajusta "s" según corresponda)


    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Ajustar el modelo
    sarimax_model = model.fit(disp=False)

    # Predicciones
    pred_train = sarimax_model.predict(start=0, end=len(y_train)-1, exog=exog_train,)
    pred_test = sarimax_model.predict(start=len(y_train), end=df.shape[0]-1, exog=exog_test)

    # Evaluar el modelo
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    pred_test.index=y_test.index
    df_test= pd.DataFrame({'Valores reales': y_test,
                             'Predicciones': pred_test})
    
    target_column2 = 'Air Quality Numeric'  # Columna a predecir
    exog_columns2 = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas','PCA']  # Variables exógenas

    # Dividir los datos en conjunto de entrenamiento y prueba
    train_size2 = int(df_PCA.shape[0] * 0.8)  # 80% entrenamiento, 20% prueba
    train_data2 = df_PCA[:train_size2]
    test_data2 = df_PCA[train_size2:]

    # Separar variables objetivo y exógenas
    y_train2 = train_data2[target_column2]
    y_test2 = test_data2[target_column2]
    exog_train2 = train_data2[exog_columns2]
    exog_test2 = test_data2[exog_columns2]

    # Parámetros iniciales p, d, q, P, D, Q, s
    p, d, q = 1, 1, 1  # Parámetros no estacionales
    P, D, Q, s = 1, 1, 1, 12  # Parámetros estacionales (suponiendo datos mensuales, ajusta "s" según corresponda)

    model2 = SARIMAX(
        y_train2,
        exog=exog_train2,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Ajustar el modelo
    sarimax_model2 = model2.fit(disp=False)
    
    # Predicciones
    pred_train2 = sarimax_model2.predict(start=0, end=len(y_train2)-1, exog=exog_train2)
    pred_test2 = sarimax_model2.predict(start=len(y_train2), end=df.shape[0]-1, exog=exog_test2)

    # Evaluar el modelo
    train_rmse2 = np.sqrt(mean_squared_error(y_train2, pred_train2))
    test_rmse2 = np.sqrt(mean_squared_error(y_test2, pred_test2))
    pred_test2.index=y_test2.index
    df_test2 = pd.DataFrame({'Valores reales': y_test2,
                             'Predicciones': pred_test2})
    
    
    st.header("Modelo Sarimax")
    
    
    st.text("Aplicamos el modelo Sarimax para predecir la variable objetivo 'Air Quality Numeric'")
    st.subheader("Caso 1: Realizamos la predicción usando el dataset pasado")
    col3,col4 = st.columns(2)
    with col3:
        st.write("Datos de entrenamiento",train_data)
    with col4:
        st.write("Datos de testeo",test_data)
    st.text("Comparativa datos de testeo y datos predecidos:")
    # Crear un gráfico de líneas usando pandas (esto utiliza Matplotlib por detrás)
    fig, ax = plt.subplots()  # Crear un objeto figure y un eje
    df_test.plot(ax=ax,title="Air Quality",figsize=(13,5))  # Graficar en el eje creado
    st.pyplot(fig)  # Usamos st.pyplot para mostrar el gráfico

    st.text("Error cuadrático medio: "+ str(test_rmse))
    st.subheader("Caso 2: Realizamos la predicción usando el dataset aumentado")
    col5,col6 = st.columns(2)
    with col5:
        st.write("Datos de entrenamiento",train_data2)
    with col6:
        st.write("Datos de testeo",test_data2)
    st.text("Comparativa datos de testeo y datos predecidos:")
    # Crear un gráfico de líneas usando pandas (esto utiliza Matplotlib por detrás)
    fig, ax = plt.subplots()  # Crear un objeto figure y un eje
    df_test2.plot(ax=ax,title="Air Quality",figsize=(13,5))  # Graficar en el eje creado
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)  # Usamos st.pyplot para mostrar el gráfico
    st.text("Error cuadrático medio: "+ str(test_rmse2))
    
    df_test3 = pd.DataFrame({'Valores reales': y_test2,
                             'Predicciones datos no aumentados': pred_test2,
                             'Predicciones datos aumentados': pred_test})
    st.subheader("Comparativa de la mejora")
    fig, ax = plt.subplots()  # Crear un objeto figure y un eje
    df_test3.plot(ax=ax,title="Air Quality",figsize=(13,5))  # Graficar en el eje creado
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)  # Usamos st.pyplot para mostrar el gráfico
    