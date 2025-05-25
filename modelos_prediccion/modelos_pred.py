from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from skforecast.model_selection import grid_search_forecaster
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

# SARIMAX

def prediccion_sarimax(datos,datos_train, columna,size):
    """
    Entrena un modelo SARIMAX optimizado mediante Grid Search y genera predicciones 
    para una serie temporal.

    Args:
        datos (pd.DataFrame): Conjunto de datos completo que contiene la serie temporal.
        datos_train (pd.DataFrame): Subconjunto de entrenamiento usado para ajustar el modelo.
        columna (str): Nombre de la columna que contiene los valores a modelar.
        size (int): Número de pasos (horizonte temporal) para los que se desea realizar la predicción.

    Returns:
        pd.Series: Serie con las predicciones generadas por el modelo SARIMAX.
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
                                            steps                 = size,
                                            metric                = 'mean_absolute_error',
                                            refit                 = True,
                                            n_jobs                = "auto",
                                            suppress_warnings_fit = True,
                                            verbose               = False,
                                            show_progress         = True
                                        )

    
    return predicciones_m1

def error_sarimax(datos_train,datos_test, columna):
    """
    Ajusta un modelo SARIMAX utilizando búsqueda de hiperparámetros (Grid Search) 
    y devuelve el error cuadrático medio (MSE) sobre el conjunto de prueba.

    Args:
        datos_train (pd.DataFrame): Conjunto de datos de entrenamiento. Debe incluir una columna con la serie temporal.
        datos_test (pd.DataFrame): Conjunto de datos de prueba. Debe tener la misma estructura que 'datos_train'.
        columna (str): Nombre de la columna con los valores de la serie temporal a modelar.

    Returns:
        float: Error cuadrático medio (MSE) entre los valores predichos y los reales en 'datos_test'.
    """
    # Grid search
    forecaster = ForecasterSarimax(
                    regressor=Sarimax(
                                    order=(1, 1, 1), # Placeholder replaced in the grid search
                                    maxiter=500
                                )
                )

    param_grid = {
        'order': [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1 ,1 ,2), ( 2, 1, 2),(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1), (1 ,0 ,2), (2, 0, 2) ],
        'seasonal_order': [(0, 0, 0, 0), (0, 1, 0, 12), (1, 1, 1, 12)],
        'trend': [None]
    }

    resultados_grid = grid_search_sarimax(
                            forecaster            = forecaster,
                            y                     = datos_train[columna],
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
                                            y                     = datos_train[columna],
                                            initial_train_size    = int(len(datos_train)*0.8),
                                            steps                 = len(datos_test),
                                            metric                = 'mean_absolute_error',
                                            refit                 = True,
                                            n_jobs                = "auto",
                                            suppress_warnings_fit = True,
                                            verbose               = False,
                                            show_progress         = True
                                        )

    
    return metrics.mean_squared_error(datos_test, predicciones_m1[:len(datos_test)])
    
# FORECASTER AUTORREGRESIVO

def prediccion_backtesting_forecasterAutoreg(datos_train,column,size,steps,param_grid,lags_grid,forecaster):
    """
    Entrena un modelo autorregresivo utilizando búsqueda de hiperparámetros (Grid Search)
    y genera predicciones para una serie temporal.

    Args:
        datos_train (pd.DataFrame): Conjunto de datos de entrenamiento con la serie temporal.
        column (str): Nombre de la columna que contiene la serie temporal a modelar.
        size (int): Número de pasos a predecir (horizonte temporal).
        steps (int): Número de pasos usados durante cada iteración del backtesting.
        param_grid (dict): Diccionario con los hiperparámetros a explorar durante el grid search del modelo.
        lags_grid (list or np.array): Lista o arreglo con los valores de lags a probar durante el grid search.
        forecaster (ForecasterAutoreg): Instancia del modelo `ForecasterAutoreg` de Skforecast.

    Returns:
        np.ndarray: Arreglo con las predicciones generadas por el modelo ajustado.
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

def error_backtesting_forecasterAutoreg(datos_train,datos_test,column,size,steps,param_grid,lags_grid,forecaster):
    """
    Ajusta un modelo autorregresivo utilizando búsqueda de hiperparámetros (Grid Search), 
    realiza predicciones y calcula el error cuadrático medio (MSE) sobre el conjunto de prueba.

    Args:
        datos_train (pd.DataFrame): Datos de entrenamiento que contienen la serie temporal.
        datos_test (pd.Series o pd.DataFrame): Datos de prueba para evaluación.
        column (str): Nombre de la columna con la serie temporal.
        size (int): Número de pasos a predecir (horizonte de predicción).
        steps (int): Número de pasos para cada iteración del backtesting.
        param_grid (dict): Diccionario con hiperparámetros a explorar en el grid search.
        lags_grid (list o np.array): Valores de lags a evaluar durante el grid search.
        forecaster (ForecasterAutoreg): Instancia del modelo `ForecasterAutoreg` de Skforecast.

    Returns:
        float: Valor del error cuadrático medio (MSE) entre las predicciones y los valores reales.
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
    
    # Error de test
    # ==============================================================================
    error_mse = mean_squared_error(
                    y_true = datos_test,
                    y_pred = predicciones
                )

    return error_mse

# FORECASTER AUTORREGRESIVO DIRECTO 

def predicciones_backtesting_forecasterAutoregDirect(datos_train,column,steps,param_grid,lags_grid,forecaster):
    """
    Entrena un modelo autorregresivo directo con regresor lineal Ridge utilizando búsqueda de hiperparámetros
    (Grid Search) y devuelve las predicciones generadas.

    Args:
        datos_train (pd.DataFrame): Datos de entrenamiento que contienen la serie temporal.
        column (str): Nombre de la columna con la serie temporal a modelar.
        steps (int): Número de pasos para cada iteración del backtesting.
        param_grid (dict): Diccionario con los hiperparámetros a explorar durante el grid search.
        lags_grid (list o np.array): Lista o arreglo con los valores de lags a probar durante el grid search.
        forecaster (ForecasterAutoregDirect): Instancia del modelo `ForecasterAutoregDirect`.

    Returns:
        np.ndarray o pd.Series: Predicciones generadas por el modelo ajustado.
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

def error_backtesting_forecasterAutoregDirect(datos_train,datos_test,steps,lags_grid,param_grid,forecaster):
    """
    Ajusta un modelo autorregresivo directo utilizando búsqueda de hiperparámetros (Grid Search), 
    genera predicciones y calcula el error cuadrático medio (MSE) sobre el conjunto de prueba.

    Args:
        datos_train (pd.DataFrame): Datos de entrenamiento con la serie temporal.
        datos_test (pd.Series o pd.DataFrame): Datos de prueba para evaluación.
        steps (int): Número de pasos para cada iteración del backtesting.
        lags_grid (list o np.array): Valores de lags a evaluar durante el grid search.
        param_grid (dict): Diccionario con hiperparámetros para explorar en el grid search.
        forecaster (ForecasterAutoregDirect): Instancia del modelo `ForecasterAutoregDirect`.

    Returns:
        float: Valor del error cuadrático medio (MSE) entre las predicciones y los valores reales.
    """
    resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos_train[datos_train.columns[0]],
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

    # Error de test
    # ==============================================================================
    error_mse = mean_squared_error(
                    y_true = datos_test,
                    y_pred = predicciones
                )

    return error_mse
    
# PROPHET

def pred_prophet_prediccion(data_train,column,size,frequ):
    """
    Entrena un modelo Prophet con los datos de entrenamiento y devuelve las predicciones futuras.

    Args:
        data_train (pd.DataFrame): DataFrame con datos históricos de entrenamiento.
        column (str): Nombre de la columna objetivo a predecir en data_train.
        size (int): Número de períodos futuros a predecir.
        frequ (str): Frecuencia temporal de los datos (e.g., 'D' para diario, 'M' para mensual).

    Returns:
        np.ndarray: Array con las predicciones para los próximos 'size' períodos.
    """
    data_train=data_train.reset_index()
    data_train.rename(columns={data_train.columns[0] : 'ds', column: 'y'}, inplace=True)
    model = Prophet()
    model.fit(data_train)
    
    future = model.make_future_dataframe(periods=size,freq=frequ)
    forecast=model.predict(future)
    
    y_pred=forecast['yhat'][len(data_train):].values
    
    return y_pred

def error_prophet_prediccion(data_train,data_test,frequ):
    """
    Entrena un modelo Prophet con los datos de entrenamiento, realiza predicciones para el periodo 
    correspondiente al conjunto de prueba y calcula el error cuadrático medio (MSE).

    Args:
        data_train (pd.DataFrame): Datos de entrenamiento. Se espera que tenga un índice temporal.
                                   La primera columna será convertida en 'ds' y la segunda en 'y'.
        data_test (pd.Series o pd.DataFrame): Datos de prueba para evaluar el modelo. Debe tener la misma estructura y frecuencia que data_train.
        frequ (str): Frecuencia de los datos. Valores comunes incluyen 'D' (diaria), 'M' (mensual), 'H' (horaria), etc.

    Returns:
        float: Error cuadrático medio (MSE) entre las predicciones del modelo Prophet y los valores reales del conjunto de prueba.
    """
    data_train=data_train.reset_index()
    data_train.rename(columns={data_train.columns[0] : 'ds', data_train.columns[1]: 'y'}, inplace=True)
    model = Prophet()
    model.fit(data_train)
    
    future = model.make_future_dataframe(periods=len(data_test),freq=frequ)
    forecast=model.predict(future)
    
    y_true=data_test.values
    y_pred=forecast['yhat'][len(data_train):].values
    
    mae = mean_squared_error(y_true,y_pred)
    return mae

# REGRESIÓN LINEAL

def pred_entrenar_linearReg(df,columns_predict):
    """
    Entrena un modelo de regresión lineal sobre un conjunto de datos y devuelve las predicciones
    realizadas sobre el 20% final del conjunto de datos (datos de testeo).

    Args:
        df (pd.DataFrame): Conjunto de datos con variables predictoras y la variable objetivo.
        columns_predict (str): Nombre de la columna objetivo (la variable a predecir).

    Returns:
        pd.DataFrame: DataFrame que contiene los datos de testeo junto con una columna adicional 
                      llamada 'Predicciones' con los valores predichos por el modelo.
    """
    modelo = LinearRegression()
    l = int(df.shape[0]*0.8)
    modelo.fit(X=df[:l].drop(columns=columns_predict),y=df[columns_predict][:l])
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = modelo.predict(df[l:].drop(columns=columns_predict))
    return df_pred

def error_entrenar_linearReg(df,columns_predict):
    """
    Entrena un modelo de regresión lineal sobre un conjunto de datos y calcula el error cuadrático medio (MSE)
    en el conjunto de testeo.

    Args:
        df (pd.DataFrame): Conjunto de datos que contiene las variables predictoras y la variable objetivo.
        columns_predict (str): Nombre de la columna a predecir (variable objetivo).

    Returns:
        float: Error cuadrático medio (MSE) calculado sobre el 20% final del conjunto de datos (datos de test).
    """
    modelo = LinearRegression()
    l = int(df.shape[0]*0.8)
    modelo.fit(X=df[:l].drop(columns=columns_predict),y=df[columns_predict][:l])
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = modelo.predict(df[l:].drop(columns=columns_predict))
    mse = mean_squared_error(df_pred['Predicciones'].values,df_pred[columns_predict].values)
    return mse 

# DECISION TREE

def pred_entrenar_TreeReg(df,columns_predict):
    """
    Entrena un modelo de regresión basado en árbol de decisión con búsqueda de hiperparámetros 
    mediante validación cruzada y devuelve las predicciones realizadas sobre el conjunto de testeo 
    (último 20% del conjunto de datos).

    Args:
        df (pd.DataFrame): Conjunto de datos que contiene las variables predictoras y la variable objetivo.
        columns_predict (str): Nombre de la columna objetivo que se desea predecir.

    Returns:
        pd.DataFrame: Subconjunto del DataFrame original correspondiente al 20% final, con una columna adicional 
                      llamada 'Predicciones' que contiene las predicciones generadas por el modelo entrenado.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = DecisionTreeRegressor(random_state=42)
    param_grid = {
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }  
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = DecisionTreeRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    
    return df_pred

def error_entrenar_TreeReg(df,columns_predict):
    """
    Entrena un modelo de regresión basado en árbol de decisión con búsqueda de hiperparámetros 
    mediante validación cruzada y devuelve el error cuadrático medio sobre los datos de testeo 
    (último 20% del dataset).

    Args:
        df (pd.DataFrame): Conjunto de datos que contiene las variables predictoras y la variable objetivo.
        columns_predict (str): Nombre de la columna objetivo que se desea predecir.

    Returns:
        float: Valor del error cuadrático medio (MSE) entre las predicciones del modelo y los valores reales 
               de la columna objetivo en el conjunto de testeo.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = DecisionTreeRegressor(random_state=42)
    param_grid = {
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }  
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = DecisionTreeRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    mse = mean_squared_error(df_pred['Predicciones'].values,df_pred[columns_predict].values)
    return mse

# RANDOM FOREST 

def pred_entrenar_RandomForestReg(df,columns_predict):
    """
    Entrena un modelo de regresión basado en Random Forest usando búsqueda de hiperparámetros 
    mediante validación cruzada y devuelve un DataFrame con las predicciones para el conjunto 
    de testeo (último 20% del conjunto de datos).

    Args:
        df (pd.DataFrame): DataFrame con las variables predictoras y la variable objetivo.
        columns_predict (str): Nombre de la columna objetivo a predecir.

    Returns:
        pd.DataFrame: Copia del subconjunto del DataFrame correspondiente al conjunto de testeo, 
                      con una nueva columna 'Predicciones' que contiene las predicciones del modelo optimizado.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [ 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
    } 
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = RandomForestRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    
    return df_pred

def error_entrenar_RandomForestReg(df,columns_predict):
    """
    Entrena un modelo de regresión basado en Random Forest utilizando búsqueda de hiperparámetros 
    por validación cruzada y devuelve el error cuadrático medio (MSE) sobre el conjunto de testeo 
    (último 20% del conjunto de datos).

    Args:
        df (pd.DataFrame): DataFrame que contiene las variables predictoras y la variable objetivo.
        columns_predict (str): Nombre de la columna objetivo que se desea predecir.

    Returns:
        float: Error cuadrático medio (MSE) calculado entre las predicciones del modelo optimizado 
               y los valores reales del conjunto de testeo.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [ 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
    } 
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = RandomForestRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    
    mse = mean_squared_error(df_pred['Predicciones'].values,df_pred[columns_predict].values)
    
    return mse

# GRADIENT BOOSTING

def pred_entrenar_GradientBoostReg(df,columns_predict):
    """
    Entrena un modelo de regresión basado en Gradient Boosting con búsqueda de hiperparámetros
    y devuelve un DataFrame con las predicciones para el conjunto de prueba.

    Args:
        df (pd.DataFrame): DataFrame con los datos completos (entrenamiento + prueba).
        columns_predict (list or str): Nombre(s) de la(s) columna(s) objetivo(s) a predecir.

    Returns:
        pd.DataFrame: DataFrame con los datos de prueba y una columna adicional 'Predicciones' con los valores predichos.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    } 
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = GradientBoostingRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    
    return df_pred

def error_entrenar_GradientBoostReg(df,columns_predict):
    """
    Entrena un modelo de regresión Gradient Boosting con búsqueda de hiperparámetros
    y calcula el error cuadrático medio (MSE) sobre el conjunto de prueba.

    Args:
        df (pd.DataFrame): DataFrame con los datos completos (entrenamiento + prueba).
        columns_predict (list or str): Nombre(s) de la(s) columna(s) objetivo(s) a predecir.

    Returns:
        float: Error cuadrático medio (MSE) entre las predicciones y los valores reales del conjunto de prueba.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    } 
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = GradientBoostingRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    
    mse = mean_squared_error(df_pred['Predicciones'].values,df_pred[columns_predict].values)
    
    return mse

# ExtraTree

def pred_entrenar_ExtraTreeReg(df,columns_predict):
    """
    Entrena un modelo Extra Trees Regressor optimizado mediante búsqueda de hiperparámetros
    y devuelve un DataFrame con las predicciones para el conjunto de prueba.

    Args:
        df (pd.DataFrame): DataFrame completo con datos de entrenamiento y prueba.
        columns_predict (list or str): Nombre(s) de la(s) columna(s) objetivo(s).

    Returns:
        pd.DataFrame: DataFrame del conjunto de prueba con una nueva columna 'Predicciones' 
                      con los valores predichos por el modelo.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = ExtraTreesRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [ 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
    } 
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = ExtraTreesRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    return df_pred

def error_entrenar_ExtraTreeReg(df,columns_predict):
    """
    Entrena un modelo Extra Trees Regressor con búsqueda de hiperparámetros y calcula
    el error cuadrático medio (MSE) en el conjunto de prueba.

    Args:
        df (pd.DataFrame): DataFrame completo con datos de entrenamiento y prueba.
        columns_predict (list or str): Nombre(s) de la(s) columna(s) objetivo(s).

    Returns:
        float: MSE entre las predicciones y los valores reales del conjunto de prueba.
    """
    l = int(df.shape[0]*0.8)
    X_train=df[:l].drop(columns=columns_predict)
    y_train=df[columns_predict][:l]
    
    modelo = ExtraTreesRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [ 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
    } 
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    optimized_model = ExtraTreesRegressor(**best_params, random_state=42) 
    optimized_model.fit(X_train, y_train)
    
    df_pred = df[l:].copy()
    df_pred['Predicciones'] = optimized_model.predict(df[l:].drop(columns=columns_predict))
    
    mse = mean_squared_error(df_pred['Predicciones'].values,df_pred[columns_predict].values)
    
    return mse
