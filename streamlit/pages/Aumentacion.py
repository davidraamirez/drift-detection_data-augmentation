import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt

st.title("Petición Datos")
uploaded_file = st.file_uploader("Cargar un archivo CSV", type=["csv"])

if uploaded_file is not None:
    
    # Convertir el archivo a un DataFrame de pandas
    df = pd.read_csv(uploaded_file)
    df.set_index(df.columns[0],inplace=True)

    # Mostrar el DataFrame cargado
    st.write("Datos cargados:", df)

    st.title("Graficar Datos")
    # Crear un gráfico de líneas usando pandas (esto utiliza Matplotlib por detrás)
    fig, ax = plt.subplots()  # Crear un objeto figure y un eje
    df.plot(ax=ax,title="Serie temporal",figsize=(13,5))  # Graficar en el eje creado

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)  # Usamos st.pyplot para mostrar el gráfico
    
    st.title("Técnicas para generar una variable exógena")
    
    # 3. Convertir el DataFrame a CSV para enviar en el POST
    csv_data = df.to_csv(index=df.index.name)
    
    st.header("Modelos de regresión:")

    st.subheader("Media")
    api_urlMedia = "http://127.0.0.1:8000/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice="+df.index.name
    api_urlMedia2 = "http://127.0.0.1:8000/Plot/Aumentar/Estadistica?tipo=1&num="+str(num)+"&freq=M&indice="+df.index.name

    try:
        files = {'file': ('Media.csv', io.StringIO(csv_data), 'text/csv')}
        responseMedia= requests.post(api_urlMedia,files=files)
        files = {'file': ('Media.csv', io.StringIO(csv_data), 'text/csv')}
        responseMedia2 = requests.post(api_urlMedia2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseMedia2.status_code == 200 and responseMedia.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseMedia2.content))
            # Mostrar la imagen en la aplicación
            st.image(image, caption="Serie temporal aumentada con media") 
            datos_media = responseMedia.content
            df_media = pd.read_csv(pd.io.common.BytesIO(datos_media),index_col="Indice")
            st.dataframe(df_media)
        else:
            st.error(f"Error al consultar la API: {responseMedia.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")