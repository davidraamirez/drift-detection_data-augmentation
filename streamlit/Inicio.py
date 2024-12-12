import streamlit as st

import streamlit as st
from PIL import Image

# Configurar la página
st.set_page_config(
    page_title="Aplicación",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Agregar un banner o imagen de encabezado (opcional)
#st.image("https://via.placeholder.com/1200x300.png?text=Drift+Detection+App", use_column_width=True)

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