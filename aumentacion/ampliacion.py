import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline, CubicSpline
from statsmodels.tsa.seasonal import seasonal_decompose
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from skforecast.model_selection import grid_search_forecaster
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import Ridge
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

# Ampliación del índice
def series_periodos(inicio, periodos, freq): 
    """
    Genera una serie temporal de fechas con una frecuencia y duración especificadas.

    Args:
        inicio (str or pd.Timestamp): Fecha de inicio de la serie.
        periodos (int): Número total de periodos a generar.
        freq (str): Frecuencia de los periodos (por ejemplo, 'D', 'M', 'H').

    Returns:
        pd.DatetimeIndex: Serie de fechas generadas según los parámetros especificados.
    """
    serie = pd.date_range(start=inicio, periods=periodos, freq=freq)
    return serie

# Normal
def normal(df,freq,size):
    """
    Genera un nuevo DataFrame con datos aumentados, donde los datos adicionales
    se obtienen a partir de una distribución normal que utiliza la media y 
    desviación estándar de las columnas originales del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original con los datos base.
        freq (str): Frecuencia temporal de los datos (por ejemplo, 'D' para diario, 'M' para mensual).
        size (int): Número de datos adicionales a generar por columna.

    Returns:
        pd.DataFrame: DataFrame con los datos originales y los nuevos datos generados,
                      indexado con un rango temporal que incluye los periodos añadidos.
    """
    np.random.seed(1)
    indice=series_periodos(df.index[0],size+df.shape[0],freq)
    for x in df.columns:
        data = df[x]
        mean,std_dev = np.mean(data),np.std(data)
        data_augmented = np.random.normal(mean,std_dev,size=size)
        if x == df.columns[0]:
            df_normal=pd.DataFrame(data=np.concatenate((df[x].values,data_augmented)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((data,data_augmented)),index=indice,columns=[x])
            df_normal= df_normal.join(df_new, how="outer")
    return df_normal

# Técnica Box-Muller

def box_muller_transform(mean, std_dev, size=100):
    """
    Genera datos simulados a partir de una distribución normal utilizando
    el método de Box-Muller, con parámetros de media y desviación estándar 
    especificados.

    Este método utiliza dos variables aleatorias uniformes para generar 
    una variable aleatoria con distribución normal estándar, y luego escala 
    y desplaza los valores resultantes.

    Args:
        mean (float or np.ndarray): Media de la distribución normal.
        std_dev (float or np.ndarray): Desviación estándar de la distribución.
        size (int, optional): Número de muestras a generar. Por defecto es 100.

    Returns:
        np.ndarray: Array de datos generados con distribución normal de media `mean` y desviación `std_dev`.
    """
    u1, u2 = np.random.rand(size), np.random.rand(size)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + z1 * std_dev

def box_muller(df,freq,size):
    """
    Genera un DataFrame ampliado con datos adicionales simulados utilizando
    el método de Box-Muller. Los nuevos datos siguen una distribución normal
    con la misma media y desviación estándar que los datos originales.

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        freq (str): Frecuencia temporal del índice (por ejemplo, 'D', 'M').
        size (int): Número de nuevos datos a generar por columna.

    Returns:
        pd.DataFrame: Nuevo DataFrame con los datos originales y los datos
                      generados mediante Box-Muller, indexado con fechas extendidas.
    """
    np.random.seed(1)
    indice=series_periodos(df.index[0],size+df.shape[0],freq)
    for x in df.columns:
        data = df[x].values
        data_bm = box_muller_transform(data.mean(),data.std(),size)
        if x == df.columns[0]:
            df_bm=pd.DataFrame(data=np.concatenate((df[x].values,data_bm)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((df[x].values,data_bm)),index=indice,columns=[x])
            df_bm = df_bm.join(df_new, how="outer")
    return df_bm

# Duplicado y Perturbación

def duplicate_and_perturb(data, duplication_factor=0.3, perturbation_std=0.05):
    """
    Duplica aleatoriamente puntos de un array y les añade una pequeña perturbación
    gaussiana, simulando ruido. Apropiado para tareas de aumento de datos.

    Args:
        data (array-like): Datos originales (1D).
        duplication_factor (float): Probabilidad de duplicar cada punto con perturbación.
        perturbation_std (float): Desviación estándar del ruido a aplicar a los duplicados.

    Returns:
        np.ndarray: Array extendido con duplicados perturbados.
    """
    duplicated_data = []
    np.random.seed(8)
    for point in data:
        duplicated_data.append(point)
        if np.random.rand() < duplication_factor:
            duplicated_data.append(point + np.random.normal(0, perturbation_std))
    return np.array(duplicated_data)

def duplicados(df,freq,duplication_factor=0.3,perturbation_std=0.05):
    """
    Genera un DataFrame extendido aplicando una técnica de duplicación con
    perturbación gaussiana a los datos de cada columna. 

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        freq (str): Frecuencia para el nuevo índice temporal.
        duplication_factor (float): Probabilidad de duplicar un punto con perturbación.
        perturbation_std (float): Desviación estándar del ruido agregado a los duplicados.

    Returns:
        pd.DataFrame: DataFrame resultante con datos originales y duplicados perturbados,
                      indexado temporalmente según la frecuencia dada.
    """
    np.random.seed(1)
    for x in df.columns:
        data = df[x]
        data_dd=duplicate_and_perturb(data,duplication_factor,perturbation_std)
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(data_dd),freq)
            df_dd = pd.DataFrame(data=data_dd,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = data_dd,index=indice,columns=[x])
            df_dd = df_dd.join(df_new, how="outer")
            
    return df_dd

# Combinación lineal

def linear_combinations(data,num_datos, n_combinations):
    """
    Genera nuevos datos como combinaciones lineales aleatorias de los últimos
    `n_combinations` valores del array original. A cada combinación se le añade
    ruido gaussiano para simular variación.

    Args:
        data (array-like): Datos originales (1D).
        num_datos (int): Número total de nuevos datos a generar.
        n_combinations (int): Número de valores recientes a combinar linealmente.

    Returns:
        np.ndarray: Array extendido con los datos originales y los nuevos generados.
    """
    for _ in range(num_datos):
        datos = data[-n_combinations:]
        weights = np.random.rand(n_combinations)
        weights /= np.sum(weights)  # Normalizar pesos
        combination = np.dot(weights, datos)
        combination += np.random.normal(0,0.5)
        data=np.append(data,combination)
    return np.array(data)

def agregar_comb(df,freq,size,window_size):
    """
    Genera un nuevo DataFrame extendido mediante combinaciones lineales aleatorias
    de los últimos `window_size` valores por columna. Los nuevos datos simulan
    dependencias locales y añaden ruido para aumentar la variabilidad.

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        freq (str): Frecuencia temporal para el índice resultante (por ejemplo, 'D', 'M').
        size (int): Número de nuevos datos a generar por columna.
        window_size (int): Tamaño de la ventana de valores pasados usados en la combinación.

    Returns:
        pd.DataFrame: DataFrame con los datos originales y los nuevos generados,
                      indexado con fechas extendidas.
    """
    np.random.seed(1)
    for x in df.columns:
        data = df[x]
        datos = linear_combinations(data.values,size,window_size)
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_dl = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_dl = df_dl.join(df_new, how="outer")
    return df_dl

# Traslación y escalado

def traslacion(df,shift,freq):
    """
    Realiza un aumento de datos mediante un desplazamiento (traslación) constante
    sobre cada columna del DataFrame. Los nuevos datos se concatenan a los originales
    con un índice temporal extendido.

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        shift (float): Valor constante que se suma a cada dato (traslación).
        freq (str): Frecuencia del índice temporal para el DataFrame resultante.

    Returns:
        pd.DataFrame: DataFrame con los datos originales y desplazados, indexado
                      con fechas extendidas según la frecuencia especificada.
    """
    df_trasl =df.copy()
    for x in df_trasl.columns:
        data = df[x]
        data_augmented = df[x] + shift
        datos = np.concatenate((data.values,data_augmented))
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_trasl = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_trasl = df_trasl.join(df_new, how="outer")
    return df_trasl

def escalado(df,freq,factor):
    """
    Realiza un aumento de datos mediante el escalado (multiplicación) de los datos
    originales por un factor constante. Los datos generados se concatenan a los originales
    con un índice temporal extendido.

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        freq (str): Frecuencia del índice temporal para el DataFrame resultante.
        factor (float): Valor por el cual se multiplican los datos originales.

    Returns:
        pd.DataFrame: DataFrame con los datos originales y los escalados, indexado
                      con fechas extendidas según la frecuencia especificada.
    """
    df_esc =df.copy()
    for x in df_esc.columns:
        data = df[x]
        data_augmented = df[x]*factor
        datos = np.concatenate((data.values,data_augmented))
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_esc = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_esc= df_esc.join(df_new, how="outer")
    return df_esc

# Adición de ruido armónico:

def add_harmonic_noise(df,freq,size):
    """"
    Realiza un aumento de datos generando una versión extendida de cada columna del DataFrame
    mediante la adición de ruido armónico basado en la frecuencia dominante de la serie original.
    
    El ruido se genera a partir de una componente senoidal cuya frecuencia y amplitud provienen
    del análisis espectral (FFT) de los datos. Esta técnica permite simular oscilaciones realistas
    que preservan patrones de periodicidad presentes en la señal original.

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        freq (str): Frecuencia temporal para el nuevo índice del DataFrame resultante (ej. 'D', 'H').
        size (int): Número de muestras nuevas que se generarán y se sumarán a los datos existentes.

    Returns:
        pd.DataFrame: DataFrame extendido que incluye los datos originales y los generados con
                      ruido armónico, con un índice temporal consistente y extendido.
    """
    np.random.seed(1)
    df_harm = df.copy()
    for x in df_harm.columns:
        data = df[x]
        time = np.arange(size)
        # Aplicar FFT
        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data), d=(time[1] - time[0]))  # Frecuencias asociadas
        amplitudes = np.abs(fft_result)  # Magnitudes (amplitud)
        dominant_freq_idx = np.argmax(amplitudes)
        frequency = frequencies[dominant_freq_idx]
        amplitude = amplitudes[dominant_freq_idx]
        harmonic_noise = amplitude * np.sin(2 * np.pi * frequency * time)
        data_augmented = np.random.choice(data, size=size, replace=True) + harmonic_noise
        datos = np.concatenate((data.values,data_augmented))
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_harm = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_harm = df_harm.join(df_new, how="outer")
    
    return df_harm

# Transformaciones matemáticas:

def agregar_log(df):
    """
    Aplica la transformación logarítmica (log(1 + x)) a cada columna del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con datos numéricos positivos o cero.

    Returns:
        pd.DataFrame: Nuevo DataFrame con la transformación logarítmica aplicada.
    """
    df_o = df.copy()
    for x in df_o.columns:
        df_o[x] = np.log1p(df[x])
    return df_o

def agregar_sqrt(df):
    """
    Aplica la transformación de raíz cuadrada a cada columna del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con datos numéricos no negativos.

    Returns:
        pd.DataFrame: Nuevo DataFrame con la transformación raíz cuadrada aplicada.
    """
    df_o = df.copy()
    for x in df_o.columns:
        df_o[x] = np.sqrt(df[x])
    return df_o

def agregar_exp(df,factor):
    """
    Aplica la transformación exponencial a cada columna del DataFrame, escalando
    previamente los datos dividiéndolos por un factor.

    Args:
        df (pd.DataFrame): DataFrame con datos numéricos.
        factor (float): Valor para escalar los datos antes de aplicar exponencial.

    Returns:
        pd.DataFrame: Nuevo DataFrame con la transformación exponencial aplicada.
    """
    df_o = df.copy()
    for x in df_o.columns:
        df_o[x] = np.exp(df_o[x]/factor)
    return df_o 

def agregar_sin(df):
    """
    Aplica la función seno a cada columna del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con datos numéricos.

    Returns:
        pd.DataFrame: Nuevo DataFrame con la transformación seno aplicada.
    """
    df_o = df.copy()
    for x in df_o.columns:
        df_o[x] = np.sin(df[x])
    return df_o

def agregar_trig(df):
    """
    Aplica una transformación trigonométrica sumando seno y coseno a cada columna.

    Args:
        df (pd.DataFrame): DataFrame con datos numéricos.

    Returns:
        pd.DataFrame: Nuevo DataFrame con la transformación trigonométrica aplicada.
    """
    df_o = df.copy()
    for x in df_o.columns:
        df_o[x] = np.cos(df_o[x]) + np.sin(df_o[x])
    return df_o

def agregar_sigmoid(df):
    """
    Aplica la función sigmoide a cada columna del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con datos numéricos.

    Returns:
        pd.DataFrame: Nuevo DataFrame con la transformación sigmoide aplicada.
    """
    df_o = df.copy()
    for x in df.columns:
        df_o[x] = 1 / (1 + np.exp(-df_o[x]))
    return df_o
   
def agregar_matematica(df,freq,funcion,factor=1):
    """
    Genera un DataFrame extendido concatenando los datos originales con sus transformaciones
    matemáticas según la función especificada.

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        freq (str): Frecuencia temporal para el índice del DataFrame resultante.
        funcion (str): Nombre de la transformación matemática a aplicar. Puede ser:
                       'sqrt', 'log', 'exp', 'sin', 'cos', 'trig', 'sigmoide'.
        factor (float, opcional): Factor usado en la transformación 'exp'. Por defecto 1.

    Returns:
        pd.DataFrame: DataFrame con los datos originales y las transformaciones concatenadas,
                      indexado con fechas extendidas según la frecuencia especificada.
    """
    indice=series_periodos(df.index[0],2*df.shape[0],freq)
    for x in df.columns:
        data = df[x]
        if funcion == 'sqrt':
            transformed_data = np.sqrt(data)
        elif funcion == 'log':
            transformed_data = np.log1p(data)
        elif funcion == 'exp':
            transformed_data = np.exp(data/factor)
        elif funcion == 'sin':
            transformed_data = np.sin(data)
        elif funcion == 'cos':
            transformed_data = np.cos(data)
        elif funcion == 'trig':
            transformed_data = np.cos(data) + np.sin(data)
        elif funcion == 'sigmoide':
            transformed_data = 1 / (1 + np.exp(-data))

        if x == df.columns[0]:
            df_transf=pd.DataFrame(data=np.concatenate((data,transformed_data)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((data,transformed_data)),index=indice,columns=[x])
            df_transf= df_transf.join(df_new, how="outer")
    return df_transf

# Técnicas estadísticas: media, moda y mediana

def estadist(df,freq,num,tipo):
    """
    Genera datos adicionales en base a estadísticas descriptivas calculadas de las columnas
    originales del DataFrame. Se extiende la serie concatenando valores constantes iguales a
    la media, mediana o moda de cada columna, según el tipo seleccionado.

    Args:
        df (pd.DataFrame): DataFrame original con datos numéricos.
        freq (str): Frecuencia temporal para el índice del DataFrame resultante (ej. 'D', 'H').
        num (int): Número de nuevos datos a generar y concatenar a cada columna.
        tipo (int): Estadística a utilizar para generar los datos nuevos:
                    1 - Media
                    2 - Mediana
                    3 - Moda (primer valor modal)

    Returns:
        pd.DataFrame: DataFrame extendido que incluye los datos originales y los nuevos valores
                      constantes generados según la estadística seleccionada, con índice temporal
                      extendido.
    """
    indice=series_periodos(df.index[0],num+df.shape[0],freq)
    for x in df.columns:
        data = df[x]
        if tipo==1:
            transformed_data = np.zeros(num)+ data.mean()
        elif tipo==2:
            transformed_data = np.zeros(num) + data.median()
        elif tipo==3:
            transformed_data = np.zeros(num) + data.mode().iloc[0]

        if x == df.columns[0]:
            df_transf=pd.DataFrame(data=np.concatenate((data,transformed_data)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((data,transformed_data)),index=indice,columns=[x])
            df_transf= df_transf.join(df_new, how="outer")
            
    return df_transf

# Descomposición 

def descomp(df,size,freq,tipo):
    """
    Realiza un aumento de datos basado en la descomposición temporal de una serie,
    utilizando modelos aditivos o multiplicativos. A partir de la tendencia y la estacionalidad
    extraídas de los datos originales, genera una proyección de `size` pasos hacia el futuro.

    Args:
        df (pd.DataFrame): DataFrame original con una o más series temporales.
        size (int): Número de pasos a predecir y añadir al final de cada serie.
        freq (str): Frecuencia temporal del índice (por ejemplo, 'D', 'M', 'H').
        tipo (str): Tipo de modelo de descomposición a aplicar: `"additive"` o `"multiplicative"`.

    Returns:
        pd.DataFrame: DataFrame extendido con la serie original y los valores proyectados,
                      generados a partir de la extrapolación de la tendencia y la estacionalidad.
    """
    indice=series_periodos(df.index[0],size+df.shape[0],freq)

    for x in df.columns:
        data = df[x]
        # Descomposición de la serie
        if tipo=="additive":
            descomposicion = seasonal_decompose(data, model='additive', period=12)
        elif tipo=="multiplicative":
            descomposicion = seasonal_decompose(data, model='multiplicative', period=12)
            
        tendencia = descomposicion.trend
        estacionalidad = descomposicion.seasonal
        residuo = descomposicion.resid
        # Calcular la tasa de cambio promedio de la tendencia
        tendencia_valida = tendencia.dropna()
        cambios = tendencia_valida.diff().dropna()
        tasa_cambio_promedio = cambios.mean()

        # Extrapolar los valores de la tendencia
        n_pasos = size
        ultima_tendencia = tendencia_valida.iloc[-1]
        tendencia_futura = [ultima_tendencia + (i + 1) * tasa_cambio_promedio for i in range(n_pasos)]
        
        # Replicar los valores estacionales
        longitud_estacionalidad = 12  # Basado en la periodicidad detectada
        estacionalidad_extrapolada = np.tile(estacionalidad[-longitud_estacionalidad:], int(size/12)+1)[:size]
        if tipo=="additive":
            prediccion = tendencia_futura + estacionalidad_extrapolada
        elif tipo=="multiplicative":
            prediccion = tendencia_futura * estacionalidad_extrapolada
        if x == df.columns[0]:
            df_desc=pd.DataFrame(data=np.concatenate((data,prediccion)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((data,prediccion)),index=indice,columns=[x])
            df_desc= df_desc.join(df_new, how="outer")
    return df_desc

# Modelos predictivos
def prediccion_sarimax(datos,datos_train,columna,size):
    """
    Realiza una predicción de una serie temporal utilizando un modelo SARIMAX.
    Se lleva a cabo una búsqueda en rejilla (grid search) sobre los hiperparámetros del modelo
    para encontrar la mejor combinación de orden y estacionalidad. Posteriormente, se realiza
    backtesting para obtener las predicciones finales, con un número de pasos ajustado por el usuario.

    Args:
        datos (pd.DataFrame): DataFrame con la serie temporal completa.
        datos_train (pd.DataFrame): Subconjunto del DataFrame utilizado como conjunto de entrenamiento.
        columna (str): Nombre de la columna sobre la que se aplica el modelo.
        size (int): Número de pasos adicionales a predecir mediante backtesting.

    Returns:
        pd.Series: Predicciones generadas por el modelo SARIMAX ajustado con los mejores parámetros.
    """
    # Grid search
    forecaster = ForecasterSarimax(
                    regressor=Sarimax(
                                    order=(1, 1, 1), # Placeholder replaced in the grid search
                                    maxiter=500
                                )
                )

    param_grid = {
        'order': [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1 ,1 ,2), ( 2, 1, 2),(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1), (1 ,0 ,2), ( 2, 0, 2) ],
        'seasonal_order': [(0, 0, 0, 0), (0, 1, 0, 12), (1, 1, 1, 12)],
        'trend': [None]
    }

    resultados_grid = grid_search_sarimax(
                            forecaster            = forecaster,
                            y                     = datos[columna],
                            param_grid            = param_grid,
                            steps                 = 12,
                            refit                 = True,
                            metric                = 'mean_absolute_error',
                            initial_train_size    = int(len(datos_train)*0.8),
                            fixed_train_size      = False,
                            return_best           = False,
                            n_jobs                = 'auto',
                            suppress_warnings_fit = True,
                            verbose               = False,
                            show_progress         = True
                    )
    
    r=resultados_grid.index[0]

    # Predicciones de backtesting con el mejor modelo según el grid search
    # ==============================================================================
    forecaster_1 = ForecasterSarimax( regressor=Sarimax(order=resultados_grid.order[r], seasonal_order=resultados_grid.seasonal_order[r], maxiter=500),
                    )

    metrica_m1, predicciones_m1 = backtesting_sarimax(
                                            forecaster            = forecaster_1,
                                            y                     = datos[columna],
                                            initial_train_size    = int(len(datos_train)*0.8),
                                            steps                 = size+12,
                                            metric                = 'mean_absolute_error',
                                            refit                 = True,
                                            n_jobs                = "auto",
                                            suppress_warnings_fit = True,
                                            verbose               = False,
                                            show_progress         = True
                                        )

    
    return predicciones_m1

def prediccion_backtesting_forecasterAutoreg(datos_train,column,size,steps,param_grid,lags_grid,forecaster):
    """
    Realiza predicción de una serie temporal utilizando un modelo autoregresivo (ForecasterAutoreg),
    optimizando sus hiperparámetros mediante búsqueda en rejilla (grid search) y generando
    predicciones sobre un número definido de pasos.

    Args:
        datos_train (pd.DataFrame): Conjunto de entrenamiento con las series temporales.
        column (str): Nombre de la columna que contiene la serie objetivo.
        size (int): Número de pasos futuros que se desea predecir.
        steps (int): Número de pasos usados en la evaluación durante la validación (backtesting).
        param_grid (dict): Diccionario con los hiperparámetros del modelo a optimizar.
        lags_grid (list or dict): Conjunto de retardos (lags) a considerar en la búsqueda.
        forecaster (ForecasterAutoreg): Objeto forecaster predefinido para ajuste y predicción.

    Returns:
        np.ndarray or pd.Series: Predicciones generadas por el modelo optimizado.
    """
    resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos_train[column],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        refit              = False,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)*0.8),
                        fixed_train_size   = False,
                        return_best        = True,
                        n_jobs             = 'auto',
                        verbose            = False
                    )

    # Predicciones
    # ==============================================================================
    predicciones = forecaster.predict(steps=size)

    return predicciones

def predicciones_backtesting_forecasterAutoregDirect(datos_train,column,steps,param_grid,lags_grid,forecaster):
    """
    Entrena un modelo autoregresivo directo (ForecasterAutoregDirect) utilizando un regresor
    lineal con regularización Ridge. Optimiza los hiperparámetros mediante grid search
    y devuelve las predicciones finales del modelo entrenado.

    Args:
        datos_train (pd.DataFrame): Conjunto de entrenamiento con la serie temporal.
        column (str): Nombre de la columna objetivo a predecir.
        steps (int): Número de pasos a predecir en cada evaluación del grid search.
        param_grid (dict): Diccionario con hiperparámetros del regresor a optimizar.
        lags_grid (list or dict): Conjunto de lags (retardos) candidatos a usar.
        forecaster (ForecasterAutoregDirect): Objeto `ForecasterAutoregDirect` configurado.

    Returns:
        np.ndarray or pd.Series: Predicciones generadas por el modelo final ajustado.
    """
    resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos_train[column],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        refit              = False,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)*0.8),
                        fixed_train_size   = False,
                        return_best        = True,
                        n_jobs             = 'auto',
                        verbose            = False
                    )

    # Predicciones
    # ==============================================================================
    predicciones = forecaster.predict()

    return predicciones

def pred_prophet_prediccion(data_train,column,size,frequ):
    """
    Entrena un modelo de predicción Prophet con los datos de entrenamiento y devuelve
    las predicciones futuras para un número de pasos especificado.

    Args:
        data_train (pd.DataFrame): Conjunto de entrenamiento con un índice temporal.
        column (str): Nombre de la columna objetivo de la serie temporal.
        size (int): Número de pasos futuros a predecir.
        frequ (str): Frecuencia de los datos (ej. 'D', 'M', 'H') compatible con pandas.

    Returns:
        np.ndarray: Array con las predicciones generadas por el modelo Prophet.
    """
    data_train=data_train.reset_index()
    data_train.rename(columns={data_train.columns[0] : 'ds', column: 'y'}, inplace=True)
    model = Prophet()
    model.fit(data_train)
    
    future = model.make_future_dataframe(periods=size,freq=frequ)
    forecast=model.predict(future)
    
    y_pred=forecast['yhat'][len(data_train):].values
    
    return y_pred

# Interpolación 
def interpolacion_min_max(df,kind,num,freq):

    """
    Interpolación entre los valores mínimo y máximo de cada columna.

    Args:
        df (pd.DataFrame): DataFrame con series temporales.
        kind (str): Tipo de interpolación. Puede ser 'linear', 'cubic' o 'quadratic'.
        num (int): Número de nuevos puntos interpolados a generar.
        freq (str): Frecuencia temporal deseada (por ejemplo, 'M' para mensual).

    Returns:
        pd.DataFrame: Nuevo DataFrame con los valores originales y los nuevos puntos
                      interpolados entre el mínimo y el máximo de cada variable.

    Funcionamiento:
        - Reinicia el índice para tratarlo como numérico.
        - Para cada columna, identifica las posiciones del mínimo y máximo.
        - Aplica interpolación en ese rango.
        - Genera un nuevo índice temporal extendido.
        - Devuelve un DataFrame con los datos originales y los nuevos interpolados.
    """
    df=df.reset_index()
    indices=df.index.values
    indice=series_periodos(df[df.columns[0]][0],num+df.shape[0],freq)
    x = indices 
    for i in range(1,len(df.columns)):
        y = df[df.columns[i]]
        inicio = min(df[df.columns[i]].argmin(),df[df.columns[i]].argmax())
        fin = max(df[df.columns[i]].argmin(),df[df.columns[i]].argmax())
        f = interp1d(x, y, kind=kind) # kind ='linear' / 'cubic' / 'quadratic'
        x_new = np.linspace(inicio,fin, num=num)  # New x values
        y_new = f(x_new)  # Interpolated y values
        if i==1:
            df_int = pd.DataFrame(data=np.concatenate((y.values.reshape(-1),y_new)),index=indice,columns=[df.columns[i]])
        else :     
            df_n = pd.DataFrame(data=np.concatenate((y.values.reshape(-1),y_new)),index=indice,columns=[df.columns[i]])
            df_int= df_int.join(df_n, how="outer")
            
    return df_int

def interpolacion_normal(df,kind,num,freq):

    """
    Interpolación estándar sobre todo el rango de la serie.

    Args:
        df (pd.DataFrame): DataFrame con series temporales.
        kind (str): Tipo de interpolación ('linear', 'cubic', 'quadratic').
        num (int): Número de nuevos puntos interpolados a generar.
        freq (str): Frecuencia temporal deseada.

    Returns:
        pd.DataFrame: DataFrame original extendido con los nuevos puntos interpolados.

    Funcionamiento:
        - Reinicia el índice de las series temporales.
        - Aplica interpolación uniforme sobre toda la longitud de cada columna.
        - Combina los datos originales con los interpolados.
        - Construye un nuevo índice temporal para ajustar la extensión.
    """
    df=df.reset_index()
    indices=df.index.values
    indice=series_periodos(df[df.columns[0]][0],num+df.shape[0],freq)
    x = indices 
    for i in range(1,len(df.columns)):
        y = df[df.columns[i]]
        f = interp1d(x, y, kind=kind) # kind = 'linear' / 'cubic' / 'quadratic'
        x_new = np.linspace(0,df.shape[0]-1, num=num)  # New x values
        y_new = f(x_new)  # Interpolated y values
        if i==1:
            df_int = pd.DataFrame(data=np.concatenate((y.values.reshape(-1),y_new)),index=indice,columns=[df.columns[i]])
        else :     
            df_n = pd.DataFrame(data=np.concatenate((y.values.reshape(-1),y_new)),index=indice,columns=[df.columns[i]])
            df_int= df_int.join(df_n, how="outer")
            
    return df_int

def interpolate(data):
    """
    Calcula puntos intermedios entre valores consecutivos de una serie.

    Args:
        data (pd.Series): Serie temporal de entrada.

    Returns:
        np.ndarray: Array con los datos originales y puntos intermedios añadidos.

    Funcionamiento:
        Recorre la serie y añade, entre cada par de valores consecutivos, un valor
        intermedio que es su media aritmética.
    """
    interpolated_data = []
    for i in range(len(data) - 1):
        interpolated_data.append(data.iloc[i])
        interpolated_data.append((data.iloc[i] + data.iloc[i + 1]) / 2)  # Punto intermedio
    interpolated_data.append(data.iloc[-1])
    return np.array(interpolated_data)

def punto_medio(df,freq):   
    """
    Aplica interpolación por punto medio a todas las columnas del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con series temporales.
        freq (str): Frecuencia del índice temporal (por ejemplo, 'M' para mensual).

    Returns:
        pd.DataFrame: Nuevo DataFrame con puntos intermedios añadidos.

    Funcionamiento:
        Para cada columna, aplica `interpolate()`. Luego construye un nuevo índice
        temporal con el doble de puntos y genera el DataFrame extendido.
    """
    for x in df.columns:
        data = df[x]
        a = interpolate(data)
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(a),freq)
            df_pm = pd.DataFrame(data=a,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=a,index=indice,columns=[x])
            df_pm = df_pm.join(df_new, how="outer")
    return df_pm

def spline_interpolation_linear(data, num,s=1):
    """
    Aplica interpolación spline lineal con suavizado.

    Args:
        data (array-like): Serie de datos a interpolar.
        num (int): Número de puntos nuevos a generar.
        s (float): Parámetro de suavizado.

    Returns:
        np.ndarray: Datos interpolados con `num` nuevos valores.

    Funcionamiento:
        Ajusta un spline a los datos y lo evalúa en un rango extendido.
    """
    x = np.arange(len(data))
    spline = UnivariateSpline(x, data, s=s)
    x_new = np.linspace(0,len(data)-1, num=num)
    return spline(x_new)

def spline_interpolation_cubic(data, num):
    """
    Interpolación spline cúbica (ajuste exacto) sobre un conjunto de datos.

    Args:
        data (array-like): Serie de datos original.
        num (int): Número de puntos interpolados a generar.

    Returns:
        np.ndarray: Serie extendida con puntos generados mediante interpolación cúbica.

    Funcionamiento:
        - Se crea un spline cúbico exacto usando `CubicSpline` de SciPy.
        - Genera `num` nuevos puntos equiespaciados.
        - Calcula los valores interpolados con la curva cúbica obtenida.
    """

    x = np.arange(len(data))
    spline = CubicSpline(x,data)
    x_new = np.linspace(0,len(data)-1, num=num)
    return spline(x_new)

def interpolacion_spline(df,tipo,num,freq,s):

    """
    Aplica interpolación spline (lineal o cúbica) a cada columna de un DataFrame 
    y genera nuevos datos interpolados.

    Args:
        df (pd.DataFrame): DataFrame original con las series temporales.
        tipo (str): Tipo de interpolación a aplicar ('linear' o 'cubic').
        num (int): Número de nuevos puntos interpolados a generar.
        freq (str): Frecuencia temporal de los datos (por ejemplo, 'M', 'D', etc.).
        s (float): Parámetro de suavizado (solo aplicable si tipo='linear').

    Returns:
        pd.DataFrame: DataFrame extendido con los nuevos datos interpolados 
                      añadidos al final.

    Funcionamiento:
        - Para cada columna de `df`, se aplica interpolación spline (usando 
          `UnivariateSpline` si tipo='linear' o `CubicSpline` si tipo='cubic').
        - Se concatenan los valores originales y los interpolados.
        - Se genera un nuevo índice de fechas usando la función `series_periodos`.
        - Devuelve un nuevo DataFrame con los datos originales más los interpolados.
    """
    indice=series_periodos(df.index[0],num+df.shape[0],freq)
    for x in df.columns:
        y=df[x]
        if tipo=='linear': 
            y_new = spline_interpolation_linear(df[x],num,s)
        elif tipo=='cubic':
            y_new = spline_interpolation_cubic(df[x],num)
        if x==df.columns[0]:
            df_int = pd.DataFrame(data=np.concatenate((y.values.reshape(-1),y_new)),index=indice,columns=[x])
        else :     
            df_n = pd.DataFrame(data=np.concatenate((y.values.reshape(-1),y_new)),index=indice,columns=[x])
            df_int= df_int.join(df_n, how="outer")       
    return df_int
