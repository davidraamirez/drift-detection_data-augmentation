import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Configurar la página
st.set_page_config(
    page_title="Aplicación",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título principal
st.title("Aplicación de comprobación de drift, interpolación de datos y creación de variables exógenas")

# Descripción breve de la aplicación
st.markdown(
    """
    ### Bienvenido/a 👋
    Esta aplicación te permite:
    - Detectar *drift* en conjuntos de datos.
    - Interpolar datos faltantes para un análisis más completo.
    - Crear variables exógenas personalizadas para tus modelos.
    
    Usa el menú de la izquierda para navegar entre las funcionalidades disponibles.
    """,
    unsafe_allow_html=True,
)

st.header("Reporte estadístico de los datos:")
uploaded_file = st.file_uploader("Cargar un archivo CSV con los datos", type=["csv"])

if uploaded_file is not None: 
    
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
    

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    df.hist(figsize=(10, 8))
    plt.show()
    
    df_infor= pd.DataFrame({'Tipos':df.dtypes, 
                            'Valores únicos':df.nunique(), 
                            'Outliers':((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum(),
                            'Valores Nulos': df.isnull().sum(),
                            'Porcentajes Nulos': (df.isnull().sum() / len(df)) * 100})
    
    st.text("Reporte estadístico")
    df_informacion = pd.concat([df.describe(),df_infor.T],axis=0)
    st.table(df_informacion)
    
    colh1,colh2 = st.columns([1,1])
    i=0
    for x in df.columns :
        fig, ax = plt.subplots()
        if i%2==0:
            with colh1:
                ax.hist(df[x], bins=30, color='blue', alpha=0.7)
                ax.set_title('Histograma '+x)
                ax.set_xlabel(x)
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)
        else:
            with colh2:
                ax.hist(df[x], bins=30, color='red', alpha=0.7)
                ax.set_title('Histograma '+x)
                ax.set_xlabel(x)
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)
        
        i = i+1

    
