import io
from PIL import Image
from io import BytesIO
import requests
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt

# Configurar la página
st.set_page_config(
    page_title="Aumentación",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

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
    
    st.header("Principal Component Analysis")
    api_urlPCA = "http://127.0.0.1:8000/Variables/PCA?indice="+df.index.name+"&columna=PCA"
    api_urlPCA2 = "http://127.0.0.1:8000/Plot/Variables/PCA?indice="+df.index.name+"&columna=PCA"

    try:
        files = {'file': ('PCA.csv', io.StringIO(csv_data), 'text/csv')}
        responsePCA= requests.post(api_urlPCA,files=files)
        files = {'file': ('PCA.csv', io.StringIO(csv_data), 'text/csv')}
        responsePCA2 = requests.post(api_urlPCA2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responsePCA2.status_code == 200 and responsePCA.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responsePCA2.content))
            # Mostrar la imagen en la aplicación
            st.image(image, caption="Serie temporal con aumento de variable a través de PCA") 
            datos_PCA = responsePCA.content
            df_PCA = pd.read_csv(pd.io.common.BytesIO(datos_PCA),index_col="Indice")
            st.dataframe(df_PCA)
        else:
            st.error(f"Error al consultar la API: {responsePCA.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.header("Correlacion")
    api_urlCorr = "http://127.0.0.1:8000/Variables/Correlacion?indice="+df.index.name+"&columna=Corr"
    api_urlCorr2 = "http://127.0.0.1:8000/Plot/Variables/Correlacion?indice="+df.index.name+"&columna=Corr"

    try:
        files = {'file': ('Corr.csv', io.StringIO(csv_data), 'text/csv')}
        responseCorr= requests.post(api_urlCorr,files=files)
        files = {'file': ('Corr.csv', io.StringIO(csv_data), 'text/csv')}
        responseCorr2 = requests.post(api_urlCorr2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseCorr2.status_code == 200 and responseCorr.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseCorr2.content))
            # Mostrar la imagen en la aplicación
            st.image(image, caption="Serie temporal con aumento de variable a través de la matriz de correlación.") 
            datos_Corr = responseCorr.content
            df_Corr = pd.read_csv(pd.io.common.BytesIO(datos_Corr),index_col="Indice")
            st.dataframe(df_Corr)
        else:
            st.error(f"Error al consultar la API: {responseCorr.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.header("Covarianza")
    api_urlCov = "http://127.0.0.1:8000/Variables/Covarianza?indice="+df.index.name+"&columna=Cov"
    api_urlCov2 = "http://127.0.0.1:8000/Plot/Variables/Covarianza?indice="+df.index.name+"&columna=Cov"

    try:
        files = {'file': ('Cov.csv', io.StringIO(csv_data), 'text/csv')}
        responseCov= requests.post(api_urlCov,files=files)
        files = {'file': ('Cov.csv', io.StringIO(csv_data), 'text/csv')}
        responseCov2 = requests.post(api_urlCov2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseCov2.status_code == 200 and responseCov.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseCov2.content))
            # Mostrar la imagen en la aplicación
            st.image(image, caption="Serie temporal con aumento de variable a través de la matriz de covarianza.") 
            datos_Cov = responseCov.content
            df_Cov = pd.read_csv(pd.io.common.BytesIO(datos_Cov),index_col="Indice")
            st.dataframe(df_Cov)
        else:
            st.error(f"Error al consultar la API: {responseCov.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Crear una lista de opciones
    options = ['Lineal','Polinomica2','Polinomica3', 'Polinomica4','Exponencial','Exponencial2','Log','Raiz', 'Seno','Coseno','Tangente','Absoluto','Truncar','Log10','Log1p','Log2','Exp1','Ceil']     

    # Crear el selectbox
    selected_option = st.selectbox('Seleccione un tipo de función:', options,index=0)
    
    st.header("Funcional")
    api_urlFunc = "http://127.0.0.1:8000/Variables/Funcional?funciones="+selected_option+"&indice="+df.index.name+"&columna="+selected_option
    api_urlFunc2 = "http://127.0.0.1:8000/Plot/Variables/Funcional?funciones="+selected_option+"&indice="+df.index.name+"&columna="+selected_option

    try:
        files = {'file': ('Func.csv', io.StringIO(csv_data), 'text/csv')}
        responseFunc= requests.post(api_urlFunc,files=files)
        files = {'file': ('Func.csv', io.StringIO(csv_data), 'text/csv')}
        responseFunc2 = requests.post(api_urlFunc2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseFunc2.status_code == 200 and responseFunc.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseFunc2.content))
            # Mostrar la imagen en la aplicación
            st.image(image, caption="Serie temporal con aumento de variable a través de una función.") 
            datos_Func = responseFunc.content
            df_Func = pd.read_csv(pd.io.common.BytesIO(datos_Func),index_col="Indice")
            st.dataframe(df_Func)
        else:
            st.error(f"Error al consultar la API: {responseFunc.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.header("Condicional")
    

    
    comparacion = [None,'Entre dos columnas','Entre una columna y un valor']
    sel_comp = st.selectbox('Seleccione un tipo de comparación:',comparacion)
    
    # Crear una lista de opciones
    modos = ['Menor=','Mayor=','Menor', 'Mayor','Igual']
        
    # Crear el selectbox
    selected_modos = st.selectbox('Seleccione el modo de comparación:', modos)
    
     # Crear una lista de opciones
    funciones = ['Lineal','Polinomica2','Polinomica3', 'Polinomica4','Exponencial','Exponencial2','Log','Raiz', 'Seno','Coseno','Tangente','Absoluto','Truncar','Log10','Log1p','Log2','Exp1','Ceil']     

    # Crear el selectbox
    selected_funcion = st.multiselect('Seleccione dos tipos de función:',funciones,max_selections=2)
    
    if sel_comp == 'Entre dos columnas':
        columnas = st.multiselect('Seleccione las columnas',df.columns,max_selections=2)
        if len(columnas)==2 and len(selected_funcion)==2:
            c = str(df.columns.get_loc(columnas[0]))+selected_modos[0:5]+'es'+selected_modos[5:]+str(df.columns.get_loc(columnas[1]))
            cond = [c,'default']
            api_urlCond = "http://127.0.0.1:8000/Variables/Condicional?indice="+df.index.name+"&columna=condicional"
            api_urlCond2 = "http://127.0.0.1:8000/Plot/Variables/Condicional?indice="+df.index.name+"&columna=condicional"

            try:
                files = {'funciones': (None,",".join(selected_funcion)),
                        'condiciones':(None,",".join(cond)),
                        'file': ('Cond.csv', io.StringIO(csv_data), 'text/csv')}
                responseCond= requests.post(api_urlCond,files=files)
                files = {'funciones': (None,",".join(selected_funcion)),
                        'condiciones':(None,",".join(cond)),
                        'file': ('Cond.csv', io.StringIO(csv_data), 'text/csv')}
                responseCond2 = requests.post(api_urlCond2,files=files)
                # Mostrar datos si la respuesta es exitosa
                if responseCond2.status_code == 200 and responseCond.status_code ==200:
                    # Leer el contenido de la imagen
                    image = Image.open(BytesIO(responseCond2.content))
                    # Mostrar la imagen en la aplicación
                    st.image(image, caption="Serie temporal con aumento de variable a través de una comparativa entre los valores de dos columnas distintas.") 
                    datos_Cond = responseCond.content
                    df_Cond = pd.read_csv(pd.io.common.BytesIO(datos_Cond),index_col="Indice")
                    st.dataframe(df_Cond)
                else:
                    st.error(f"Error al consultar la API: {responseCond.text}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    elif sel_comp == 'Entre una columna y un valor':
        columna = st.selectbox('Seleccione la columna:',df.columns,index=0)
        num = st.number_input(label="Valor con el que se compara",value=0)
        c = str(df.columns.get_loc(columna))+selected_modos[0:5]+str(num)
        cond = [c,'default']
        
        api_urlCond = "http://127.0.0.1:8000/Variables/Condicional?indice="+df.index.name+"&columna=condicional"
        api_urlCond2 = "http://127.0.0.1:8000/Plot/Variables/Condicional?indice="+df.index.name+"&columna=condicional"

        try:
            files = {'funciones': (None,",".join(selected_funcion)),
                     'condiciones':(None,",".join(cond)),
                     'file': ('Cond.csv', io.StringIO(csv_data), 'text/csv')}
            responseCond= requests.post(api_urlCond,files=files)
            files = {'funciones': (None,",".join(selected_funcion)),
                     'condiciones':(None,",".join(cond)),
                     'file': ('Cond.csv', io.StringIO(csv_data), 'text/csv')}
            responseCond2 = requests.post(api_urlCond2,files=files)
            # Mostrar datos si la respuesta es exitosa
            if responseCond2.status_code == 200 and responseCond.status_code ==200:
                # Leer el contenido de la imagen
                image = Image.open(BytesIO(responseCond2.content))
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal con aumento de variable a través de una comparativa entre los valores de una columna y un valor.") 
                datos_Cond = responseCond.content
                df_Cond = pd.read_csv(pd.io.common.BytesIO(datos_Cond),index_col="Indice")
                st.dataframe(df_Cond)
            else:
                st.error(f"Error al consultar la API: {responseCond.text}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        

    
    st.header("Resultados previos a la aumentación")

    st.header("Resultados posteriores a la aumentación")

    st.header("Conclusiones")