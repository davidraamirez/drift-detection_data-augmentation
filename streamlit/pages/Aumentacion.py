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
    
    st.header("Técnicas para generar una variable exógena")
    # 3. Convertir el DataFrame a CSV para enviar en el POST
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
        
    st.subheader("Correlacion")
    api_urlCorr = "http://127.0.0.1:8000/Variables/Correlacion?indice="+df.index.name+"&columna=Corr"
    api_urlCorr2 = "http://127.0.0.1:8000/Plot/Variables/Correlacion?indice="+df.index.name+"&columna=Corr"

    try:
        col5,col6 = st.columns([2,3])
        files = {'file': ('Corr.csv', io.StringIO(csv_data), 'text/csv')}
        responseCorr= requests.post(api_urlCorr,files=files)
        files = {'file': ('Corr.csv', io.StringIO(csv_data), 'text/csv')}
        responseCorr2 = requests.post(api_urlCorr2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseCorr2.status_code == 200 and responseCorr.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseCorr2.content)) 
            datos_Corr = responseCorr.content
            df_Corr = pd.read_csv(pd.io.common.BytesIO(datos_Corr),index_col="Indice")
            with col5:
                st.dataframe(df_Corr)
            with col6:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal con aumento de variable a través de la matriz de correlación.")
            st.text("Creación de una nueva variable Corr que se ha construido aplicando la matriz de correlación sobre las variables.")    
        else:
            st.error(f"Error al consultar la API: {responseCorr.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.subheader("Covarianza")
    api_urlCov = "http://127.0.0.1:8000/Variables/Covarianza?indice="+df.index.name+"&columna=Cov"
    api_urlCov2 = "http://127.0.0.1:8000/Plot/Variables/Covarianza?indice="+df.index.name+"&columna=Cov"

    try:
        col7,col8 = st.columns([2,3])
        files = {'file': ('Cov.csv', io.StringIO(csv_data), 'text/csv')}
        responseCov= requests.post(api_urlCov,files=files)
        files = {'file': ('Cov.csv', io.StringIO(csv_data), 'text/csv')}
        responseCov2 = requests.post(api_urlCov2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseCov2.status_code == 200 and responseCov.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseCov2.content))
            datos_Cov = responseCov.content
            df_Cov = pd.read_csv(pd.io.common.BytesIO(datos_Cov),index_col="Indice")
            with col7:
                st.dataframe(df_Cov)
            with col8:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal con aumento de variable a través de la matriz de covarianza.")
            st.text("Creación de una nueva variable Cov que se ha construido aplicando la matriz de covarianza sobre las variables.")    
        else:
            st.error(f"Error al consultar la API: {responseCov.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    
    
    st.subheader("Funcional")
    # Crear una lista de opciones
    options = ['Lineal','Polinomica2','Polinomica3', 'Polinomica4','Exponencial','Exponencial2','Log','Raiz', 'Seno','Coseno','Tangente','Absoluto','Truncar','Log10','Log1p','Log2','Exp1','Ceil']     

    # Crear el selectbox
    selected_option = st.selectbox('Seleccione un tipo de función:', options,index=0)
    api_urlFunc = "http://127.0.0.1:8000/Variables/Funcional?funciones="+selected_option+"&indice="+df.index.name+"&columna="+selected_option
    api_urlFunc2 = "http://127.0.0.1:8000/Plot/Variables/Funcional?funciones="+selected_option+"&indice="+df.index.name+"&columna="+selected_option
    
    try:
        col9,col10 = st.columns([2,3])
        files = {'file': ('Func.csv', io.StringIO(csv_data), 'text/csv')}
        responseFunc= requests.post(api_urlFunc,files=files)
        files = {'file': ('Func.csv', io.StringIO(csv_data), 'text/csv')}
        responseFunc2 = requests.post(api_urlFunc2,files=files)
        # Mostrar datos si la respuesta es exitosa
        if responseFunc2.status_code == 200 and responseFunc.status_code ==200:
            # Leer el contenido de la imagen
            image = Image.open(BytesIO(responseFunc2.content))
            
            datos_Func = responseFunc.content
            df_Func = pd.read_csv(pd.io.common.BytesIO(datos_Func),index_col="Indice")
            with col9:
                st.dataframe(df_Func)
            with col10:
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal con aumento de variable a través de una función.") 
            if selected_option =='Lineal':
                rel = 'relación lineal'
            elif selected_option =='Polinomica2':
                rel = 'relación polinómica de segundo grado'
            elif selected_option == 'Polinomica3':
                rel ='relación polinómica de tercer grado'
            elif selected_option == 'Polinomica4':
                rel = 'relación polinómica de cuarto grado'
            elif selected_option == 'Exponencial':
                rel = 'relación exponencial sobre una relación lineal del resto de variables.'
            elif selected_option == 'Exponencial':
                rel ='relación exponencial de base 2 sobre una relación lineal del resto de variables.'
            elif selected_option == 'Log':
                rel = 'relación logarítmica sobre una relación lineal del resto de variables.'
            elif selected_option == 'Raiz':
                rel = 'relación radical sobre una relación lineal del resto de variables.'
            elif selected_option == 'Seno':
                rel = 'relación senoidal sobre una relación lineal del resto de variables.'
            elif selected_option == 'Coseno':
                rel = 'relación cosenoidal sobre una relación lineal del resto de variables.'
            elif selected_option == 'Tangente':
                rel = 'relación tangencial sobre una relación lineal del resto de variables.'
            elif selected_option == 'Absoluto':
                rel = 'función valor absoluto sobre una relación lineal del resto de variables.'
            elif selected_option == 'Truncar':
                rel = 'función truncar sobre una relación lineal del resto de variables.'
            elif selected_option == 'Log10':
                rel = 'relación logarítmica en base 10 sobre una relación lineal del resto de variables.'
            elif selected_option == 'Log1p':
                rel = 'relación logarítmica'
            elif selected_option == 'Log2':
                rel = 'relación logarítmica en base 2 sobre una relación lineal del resto de variables.'
            elif selected_option == 'Ceil':
                rel = 'función parte entera sobre una relación lineal del resto de variables.'
  
            st.text("Creación de una nueva variable "+selected_option +" que se ha construido aplicando una " + rel)    

        else:
            st.error(f"Error al consultar la API: {responseFunc.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    st.subheader("Condicional")
    
    col11,col12 = st.columns([2,3])
    
    with col11:
        
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
    
        if len(selected_funcion)==2:
            if selected_funcion[0] =='Lineal':
                rel0 = 'relación lineal'
            elif selected_funcion[0] =='Polinomica2':
                rel0 = 'relación polinómica de segundo grado'
            elif selected_funcion[0] == 'Polinomica3':
                rel0 ='relación polinómica de tercer grado'
            elif selected_funcion[0] == 'Polinomica4':
                rel0 = 'relación polinómica de cuarto grado'
            elif selected_funcion[0] == 'Exponencial':
                rel0 = 'relación exponencial sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Exponencial':
                rel0 ='relación exponencial de base 2 sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Log':
                rel0 = 'relación logarítmica sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Raiz':
                rel0 = 'relación radical sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Seno':
                rel0 = 'relación senoidal sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Coseno':
                rel0 = 'relación cosenoidal sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Tangente':
                rel0 = 'relación tangencial sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Absoluto':
                rel0 = 'función valor absoluto sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Truncar':
                rel0 = 'función truncar sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Log10':
                rel0 = 'relación logarítmica en base 10 sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Log1p':
                rel0 = 'relación logarítmica'
            elif selected_funcion[0] == 'Log2':
                rel0 = 'relación logarítmica en base 2 sobre una relación lineal del resto de variables.'
            elif selected_funcion[0] == 'Ceil':
                rel0 = 'función parte entera sobre una relación lineal del resto de variables.'
                            
            if selected_funcion[1] =='Lineal':
                rel1 = 'relación lineal'
            elif selected_funcion[1] =='Polinomica2':
                rel1 = 'relación polinómica de segundo grado'
            elif selected_funcion[1] == 'Polinomica3':
                rel1 ='relación polinómica de tercer grado'
            elif selected_funcion[1] == 'Polinomica4':
                rel1 = 'relación polinómica de cuarto grado'
            elif selected_funcion[1] == 'Exponencial':
                rel1 = 'relación exponencial sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Exponencial':
                rel1 ='relación exponencial de base 2 sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Log':
                rel1 = 'relación logarítmica sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Raiz':
                rel1 = 'relación radical sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Seno':
                rel1 = 'relación senoidal sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Coseno':
                rel1 = 'relación cosenoidal sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Tangente':
                rel1 = 'relación tangencial sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Absoluto':
                rel1 = 'función valor absoluto sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Truncar':
                rel1 = 'función truncar sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Log10':
                rel1 = 'relación logarítmica en base 10 sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Log1p':
                rel1 = 'relación logarítmica'
            elif selected_funcion[1] == 'Log2':
                rel1 = 'relación logarítmica en base 2 sobre una relación lineal del resto de variables.'
            elif selected_funcion[1] == 'Ceil':
                rel1 = 'función parte entera sobre una relación lineal del resto de variables.'
            
        modos = ['Menor=','Mayor=','Menor', 'Mayor','Igual']
        if selected_modos=='Menor=':
            comparar='menor o igual'
        elif selected_modos=='Mayor=':
            comparar='mayor o igual'
        elif selected_modos=='Menor':
            comparar='menor'
        elif selected_modos=='Mayor':
            comparar='mayor'
        elif selected_modos=='Igual':
            comparar='igual'
            
    if sel_comp == 'Entre dos columnas':
        with col11:
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
                    datos_Cond = responseCond.content
                    df_Cond = pd.read_csv(pd.io.common.BytesIO(datos_Cond),index_col="Indice")
                    with col12:
                        st.dataframe(df_Cond)
                    # Mostrar la imagen en la aplicación
                    st.image(image, caption="Serie temporal con aumento de variable a través de una comparativa entre los valores de dos columnas distintas.")
                    
                    st.text("Creación de una nueva variable, condicional, obtenida a través de una "+rel0 +" si "+ columnas[0] +" "+comparar+" que " + columnas[1] +" o una " +rel1+" en otro caso.")
                else:
                    st.error(f"Error al consultar la API: {responseCond.text}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    elif sel_comp == 'Entre una columna y un valor':
        with col11:
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
                datos_Cond = responseCond.content
                df_Cond = pd.read_csv(pd.io.common.BytesIO(datos_Cond),index_col="Indice")
                with col12:
                    st.dataframe(df_Cond)
                # Mostrar la imagen en la aplicación
                st.image(image, caption="Serie temporal con aumento de variable a través de una comparativa entre los valores de una columna y un valor,") 
                
                
                st.text("Creación de una nueva variable, condicional, obtenida a través de una "+rel0 +" si "+ columna +" "+comparar+" que " +str(num) +" o una " +rel1+" en otro caso.")            
            else:
                st.error(f"Error al consultar la API: {responseCond.text}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
     