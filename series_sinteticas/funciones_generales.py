import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  
from scipy.stats import binom,poisson,geom,hypergeom,uniform,expon, gamma, beta,chi2,t,pareto,lognorm
from random import randrange, random
import math


# Creación de índices 

def series_fin(inicio, fin, freq):
    """
    Genera una serie temporal entre dos fechas con la frecuencia especificada.

    A partir de una fecha de inicio y una fecha de fin, esta función devuelve una serie temporal (DatetimeIndex)
    utilizando una frecuencia definida. Es útil para generar secuencias temporales en análisis de series de tiempo.

    Frecuencias soportadas incluyen:
        - 'B' : días hábiles (business day)
        - 'D' : días calendario
        - 'W' : semanas
        - 'M' : fin de mes
        - 'Q' : fin de trimestre
        - 'Y' : fin de año
        - 'h' : horas
        - 'min' : minutos
        - 's' : segundos
        - 'ms' : milisegundos
        - 'us' : microsegundos
        - 'ns' : nanosegundos

    Args:
        inicio (str o pd.Timestamp): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD' o similar).
        fin (str o pd.Timestamp): Fecha de fin de la serie temporal.
        freq (str): Frecuencia de la serie temporal (ver opciones arriba).

    Returns:
        pd.DatetimeIndex: Serie de fechas entre `inicio` y `fin` con la frecuencia especificada.
    """
    serie = pd.date_range(start=inicio, end=fin, freq=freq)
    return serie

def series_periodos(inicio, periodos, freq):
    """
    Genera una serie temporal a partir de una fecha de inicio, una frecuencia y un número de periodos.

    Esta función crea una serie de fechas (DatetimeIndex) que inicia en la fecha indicada
    y se extiende por un número determinado de periodos, con la frecuencia especificada.

    Frecuencias soportadas incluyen:
        - 'B'   : días hábiles (business day)
        - 'D'   : días calendario
        - 'W'   : semanas
        - 'M'   : fin de mes
        - 'Q'   : fin de trimestre
        - 'Y'   : fin de año
        - 'h'   : horas
        - 'min' : minutos
        - 's'   : segundos
        - 'ms'  : milisegundos
        - 'us'  : microsegundos
        - 'ns'  : nanosegundos

    Args:
        inicio (str o pd.Timestamp): Fecha inicial de la serie temporal (por ejemplo, '2024-01-01').
        periodos (int): Número de fechas a generar en la serie.
        freq (str): Frecuencia con la que se generan los periodos (ver opciones arriba).

    Returns:
        pd.DatetimeIndex: Serie de fechas generadas a partir de `inicio`, con longitud `periodos` y frecuencia `freq`.
    """
    serie = pd.date_range(start=inicio, periods=periodos, freq=freq)
    return serie


# Series temporales generadas a partir de una distribución estadística específica
def crear_datos(distr,params,num_datos):
    """
    Genera un array de datos sintéticos a partir de una distribución o patrón específico.

    Args:
        distr (int): Código numérico que indica el tipo de distribución o patrón deseado. Las opciones son:
            1. Normal: (media, desviación típica)
            2. Binomial: (n, p) o (n, p, loc)
            3. Poisson: (mu) o (mu, loc)
            4. Geométrica: (p) o (p, loc)
            5. Hipergeométrica: (M, n, N) o (M, n, N, loc)
            6. Constante: (valor,)
            7. Uniforme: (), (loc,), o (loc, scale)
            8. Lognormal: (s,), (s, loc), o (s, loc, scale)
            9. Exponencial: (), (loc,), o (loc, scale)
            10. Gamma: (a,), (a, loc), o (a, loc, scale)
            11. Beta: (a, b), (a, b, loc), o (a, b, loc, scale)
            12. Chi-Cuadrado: (df,), (df, loc), o (df, loc, scale)
            13. T-Student: (t,), (t, loc), o (t, loc, scale)
            14. Pareto: (b,), (b, loc), o (b, loc, scale)
            15. Lineal descendente: (valor inicial, pendiente). El mínimo valor es 0.
            16. Lineal ascendente: (valor inicial, pendiente)
            17. Aleatorio entre dos valores: (mínimo, máximo)
        params (tuple): Parámetros necesarios para la distribución o patrón, descritos arriba. Los parámetros loc y scale son opcionales.
        num_datos (int): Número total de observaciones a generar.

    Returns:
        np.ndarray: Array de longitud `num_datos` con los datos generados según la distribución/patrón especificado.

    """    
    if distr == 1 :
        datos = np.random.normal(params[0],params[1],num_datos)
        
    elif distr ==2 :
        if len(params)==2:
            datos = binom.rvs(params[0],params[1],size=num_datos)
        elif len(params) == 3:
            datos = binom.rvs(params[0],params[1],params[2],size=num_datos)
            
    elif distr== 3 :
        if len(params)==1:
            datos = poisson.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = poisson.rvs(params[0],params[1],size=num_datos)
            
    elif distr == 4 :
        if len(params)==1:
            datos = geom.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = geom.rvs(params[0],params[1],size=num_datos)
            
    elif distr == 5:
        if len(params)==3:
            datos = hypergeom.rvs(params[0],params[1],params[2],size=num_datos)
        elif len(params) == 4:
            datos = hypergeom.rvs(params[0],params[1],params[2],params[3],size=num_datos)
            
    elif distr == 6: 
        datos = np.zeros(num_datos) + params[0]
        
    elif distr == 7:
        if len(params)==0:
            datos = uniform.rvs(size=num_datos)
        elif len(params)==1:
            datos = uniform.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = uniform.rvs(params[0],params[1],size=num_datos)
            
    elif distr == 8:
        if len(params)==1:
            datos = lognorm.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = lognorm.rvs(params[0],params[1],size=num_datos)
        elif len(params) == 3:
            datos = lognorm.rvs(params[0],params[1],params[2],size=num_datos)
            
    elif distr == 9: 
        if len(params)==0:
            datos = expon.rvs(size=num_datos)
        elif len(params)==1:
            datos = expon.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = expon.rvs(params[0],params[1],size=num_datos)
            
    elif distr == 10: 
        if len(params)==1:
            datos = gamma.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = gamma.rvs(params[0],params[1],size=num_datos)
        elif len(params) == 3:
            datos = gamma.rvs(params[0],params[1],params[2],size=num_datos)
            
    elif distr == 11: 
        if len(params)==2:
            datos = beta.rvs(params[0],params[1],size=num_datos)
        elif len(params) == 3:
            datos = beta.rvs(params[0],params[1],params[2],size=num_datos)
        elif len(params) == 4:
            datos = beta.rvs(params[0],params[1],params[2],params[3],size=num_datos)
            
    elif distr == 12: 
        if len(params)==1:
            datos = chi2.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = chi2.rvs(params[0],params[1],size=num_datos)
        elif len(params) == 3:
            datos = chi2.rvs(params[0],params[1],params[2],size=num_datos)
            
    elif distr == 13: 
        if len(params)==1:
            datos = t.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = t.rvs(params[0],params[1],size=num_datos)
        elif len(params) == 3:
            datos = t.rvs(params[0],params[1],params[2],size=num_datos)
            
    elif distr == 14: 
        if len(params)==1:
            datos = pareto.rvs(params[0],size=num_datos)
        elif len(params) == 2:
            datos = pareto.rvs(params[0],params[1],size=num_datos)
        elif len(params) == 3:
            datos = pareto.rvs(params[0],params[1],params[2],size=num_datos)
            
    elif distr == 15:
        datos = np.zeros(num_datos)
        datos[0]= params[0]
        i=1
        while datos[i-1]>0 and i<num_datos:
            datos[i] = datos[i-1] - params[1]
            i= i+1
            
    elif distr == 16:
        datos = np.zeros(num_datos)
        datos[0] = params[0]
        for i in range(1,num_datos):
            datos[i] = datos[i-1] + params[1]
    
    elif distr == 17:
        datos= np.zeros(num_datos)
        for i in range(0,num_datos):
            datos[i] = randrange(params[0],params[1])
        
    return datos
 
# Series temporales con comportamiento periódico
def datos_periodicos_amplitud(distr, params, num_datos, periodo):
    """
    Genera una serie temporal periódica repitiendo un patrón de datos con una amplitud determinada.

    Args:
        distr (int): Código de la distribución que seguirán los datos en cada período (ver definiciones previas).
        params (list): Lista de parámetros correspondientes a la distribución especificada.
        num_datos (int): Número total de datos a generar en la serie.
        periodo (int): Número de datos que conforman un solo período (amplitud del patrón repetido).

    Returns:
        np.ndarray: Array con los datos generados, de longitud igual a `num_datos`, formado por la repetición del patrón generado para un período.
    """
    num_periodos = int(num_datos / periodo)
    res = num_datos % periodo
    datos_base0 = crear_datos(distr, params, periodo)
    datos_base = datos_base0
    for _ in range(num_periodos - 1):
        datos_base = np.concatenate((datos_base0, datos_base))
    if res > 0:
        datos_base = np.concatenate((datos_base, datos_base0[:res]))
    return datos_base

def datos_periodicos_cantidad(distr,params,num_datos,num_periodos):
    """
    Genera una serie temporal periódica dividiendo el total de datos en un número fijo de períodos.

    Args:
        distr (int): Código de la distribución que seguirán los datos en cada período (ver definiciones previas).
        params (list): Lista de parámetros correspondientes a la distribución especificada.
        num_datos (int): Número total de datos a generar en la serie.
        num_periodos (int): Número total de períodos en los que se dividirá la serie.

    Returns:
        np.ndarray: Array con los datos generados, de longitud igual a `num_datos`, compuesto por la repetición de un patrón base generado para cada período.
    """
    periodo = int(num_datos/num_periodos)
    res = num_datos % num_periodos
    datos_base0 = crear_datos(distr,params,periodo)
    datos_base = datos_base0
    for _ in range(0,num_periodos-1):
        datos_base=np.concatenate((datos_base0,datos_base))
    if res>0:
        datos_base=np.concatenate((datos_base,datos_base0[:res]))
    return datos_base

# Series temporales de tendencia determinista

def tendencia_determinista_lineal (a,b,t,e=0):
    """
    Calcula el valor de una tendencia lineal con componente de error en el instante t.

    Args:
        a (float): Término independiente.
        b (float): Pendiente de la recta.
        t (int o float): Instante de tiempo.
        e (float, opcional): Término de error. Por defecto 0.

    Returns:
        float: Valor de la tendencia lineal en el tiempo t, incluyendo el error.
    """
    return a + b*t + e

def tendencia_determinista_polinómica(params,t,e=0):
    """
    Calcula el valor de una tendencia polinómica con componente de error en el instante t.

    Args:
        params (list): Lista de coeficientes del polinomio, en orden creciente de grado. 
                       Por ejemplo, [a0, a1, a2] representa a0 + a1*t + a2*t^2.
        t (int o float): Instante de tiempo.
        e (float, opcional): Término de error. Por defecto 0.

    Returns:
        float: Valor de la tendencia polinómica en el tiempo t, incluyendo el error.
    """
    res = params[0]
    for k in range(1,len(params)):
        res = res + params[k] * t**k
    return res + e

def tendencia_determinista_exponencial(a,b,t,e=0):
    """
    Calcula el valor de una tendencia exponencial con componente de error en el instante t.

    Args:
        a (float): Coeficiente del exponente.
        b (float): Multiplicador del tiempo dentro del exponente.
        t (int o float): Instante de tiempo.
        e (float, opcional): Término de error. Por defecto 0.

    Returns:
        float: Valor de la tendencia exponencial en el tiempo t.
    """
    return math.exp(a+b*t+e)

def tendencia_determinista_logaritmica(a,b,t,e=0):
    """
    Calcula el valor de una tendencia logarítmica con componente de error en el instante t.

    Args:
        a (float): Término independiente.
        b (float): Coeficiente del término logarítmico.
        t (int o float): Instante de tiempo. Debe ser mayor que 0.
        e (float, opcional): Término de error. Por defecto 0.

    Returns:
        float: Valor de la tendencia logarítmica en el tiempo t.
    """
    return a + b * math.log(t) + e

def tendencia_det(params,tipo,num_datos,coef_error=0):
    """
    Genera una serie de datos con tendencia determinista según el tipo especificado.

    Args:
        params (list): Parámetros del modelo de tendencia. Su contenido varía según el tipo:
            - Tipo 1 (lineal): [a, b]
            - Tipo 2 (polinómica): [a0, a1, a2, ..., an]
            - Tipo 3 (exponencial): [a, b]
            - Tipo 4 (logarítmica): [a, b]
        tipo (int): Tipo de tendencia determinista a aplicar:
            - 1: Lineal
            - 2: Polinómica
            - 3: Exponencial
            - 4: Logarítmica
        num_datos (int): Número de datos a generar.
        coef_error (float, opcional): Coeficiente que controla la magnitud del término de error aleatorio añadido a cada valor. Por defecto 0.

    Returns:
        np.ndarray: Array de tamaño `num_datos` con los valores generados según la tendencia seleccionada.
    """
    datos = np.zeros(num_datos)
    
    for t in range(1,num_datos+1):
        e = random()*coef_error
        
        if tipo==1:
            datos[t-1] = tendencia_determinista_lineal(params[0],params[1],t,e)
        elif tipo==2:
            datos[t-1] = tendencia_determinista_polinómica(params,t,e)
        elif tipo==3:
            datos[t-1] = tendencia_determinista_exponencial(params[0],params[1],t,e)
        elif tipo==4:
            datos[t-1] = tendencia_determinista_logaritmica(params[0],params[1],t,e)
            
    return datos   

# Series autorregresivas y de medias móviles

def modelo_AR(c,phi,num_datos,desv):
    """
    Genera una serie temporal utilizando un modelo autorregresivo (AR) de orden p.

    Args:
        c (float): Término constante del modelo AR.
        phi (list): Lista de coeficientes AR [phi₁, phi₂, ..., phiₚ], donde p es el orden del modelo.
        num_datos (int): Número total de datos a generar.
        desv (float): Desviación típica del ruido blanco añadido a la serie.

    Returns:
        np.ndarray: Serie temporal generada siguiendo el modelo AR.
    """
    orden = len(phi)
    
    a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    
    for k in range(0,orden):
        datos[k] = c + a[k]
            
    for i in range(orden,num_datos):
        datos [i]= c + a[i]
        for j in range (1,orden+1):
            datos[i] = datos[i] + phi[j-1]*datos[i-j]
    
    return datos

def modelo_MA(c,teta,num_datos,desv):
    """
    Genera una serie temporal utilizando un modelo de media móvil (MA) de orden q.

    Args:
        c (float): Término constante del modelo MA.
        teta (list): Lista de coeficientes MA [θ₁, θ₂, ..., θ_q], donde q es el orden del modelo.
        num_datos (int): Número total de datos a generar.
        desv (float): Desviación típica del ruido blanco.

    Returns:
        np.ndarray: Serie temporal generada siguiendo el modelo MA.
    """    
    orden = len(teta)
    
    a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    for i in range(0,orden):
        datos[i]= c + a[i]
            
    for i in range(orden,num_datos):
        datos[i] = c + a[i]
        for j in range (1,orden+1):
            datos[i]= datos[i] + teta[j-1]*a[i-j]
            
    return datos

def modelo_ARMA(c,phi,teta,num_datos,desv):
    """
    Genera una serie temporal utilizando un modelo ARMA(p, q), que combina
    componentes autorregresivas (AR) y de media móvil (MA).

    Args:
        c (float): Término constante del modelo ARMA.
        phi (list): Lista de coeficientes autorregresivos (AR) [φ₁, φ₂, ..., φ_p], donde p es el orden AR.
        teta (list): Lista de coeficientes de media móvil (MA) [θ₁, θ₂, ..., θ_q], donde q es el orden MA.
        num_datos (int): Número total de datos a generar.
        desv (float): Desviación típica del ruido blanco (asumido ~ N(0, desv)).

    Returns:
        np.ndarray: Serie temporal generada siguiendo el modelo ARMA(p, q).
    """
    a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    p = len(phi)
    q = len(teta)   
    maxi = max(p,q)
    
    for k in range(0,maxi):
        datos[k] = c + a[k]
            
    for i in range(maxi,num_datos):
        datos[i] = c + a[i]
            
        for j in range (1,p+1):
            datos[i]= datos[i] + phi[j-1]*datos[i-j]
                
        for k in range(1,q+1):
            datos[i] = datos[i] + teta[k-1]*a[i-k]   
                
    return datos
    
def modelo_AR_estacional(c,phi,s,num_datos,desv):
    """
    Genera una serie temporal utilizando un modelo autorregresivo (AR) estacional.

    En este modelo, los valores pasados que afectan al presente están separados por un
    periodo estacional `s`.

    Args:
        c (float): Término constante del modelo.
        phi (list): Lista de coeficientes autorregresivos para el componente estacional.
        s (int): Periodo estacional (número de observaciones que conforman un ciclo completo).
        num_datos (int): Número total de datos a generar.
        desv (float): Desviación típica del ruido blanco (asumido ~ N(0, desv)).

    Returns:
        np.ndarray: Serie temporal generada siguiendo el modelo AR estacional.
    """
    a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    orden = len(phi)
    
    for k in range(0,s*orden):
        datos[k] = a[k] + c
            
    for i in range(orden*s,num_datos):
        datos [i]= c + a[i]
        for j in range (1,orden+1):
            datos[i] = datos[i] + phi[j-1]*datos[i-j*s]
    
    return datos

def modelo_MA_estacional(c,teta,s,num_datos,desv):
    """
    Genera una serie temporal utilizando un modelo de medias móviles estacional (MA estacional).

    En este modelo, el valor actual depende del ruido blanco actual y de valores pasados 
    del ruido separados por una estacionalidad `s`.

    Args:
        c (float): Término constante del modelo.
        teta (list): Lista de coeficientes del componente de medias móviles estacional.
        s (int): Periodo estacional (número de observaciones entre términos relacionados).
        num_datos (int): Número total de datos a generar.
        desv (float): Desviación típica del ruido blanco (asumido ~ N(0, desv)).

    Returns:
        np.ndarray: Serie temporal generada siguiendo el modelo MA estacional.
    """
    a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    orden = len(teta)
    
    for i in range(0,s*orden):
        datos[i]= a[i] + c
            
    for i in range(s*orden,num_datos):
        datos[i] = c + a[i]
        for j in range (1,orden+1):
            datos[i]= datos[i] + teta[j-1]*a[i-j*s]
                
    return datos

def modelo_ARMA_estacional(c,phi,teta,s,num_datos,desv):
    """
    Genera una serie temporal utilizando un modelo ARMA estacional (SARMA).

    Este modelo combina componentes autorregresivos (AR) y de medias móviles (MA) 
    con estructura estacional. Los rezagos están determinados por el parámetro de estacionalidad `s`.

    Args:
        c (float): Término constante del modelo.
        phi (list): Coeficientes del componente autorregresivo estacional (AR).
        teta (list): Coeficientes del componente de medias móviles estacional (MA).
        s (int): Periodo estacional (número de observaciones entre términos relacionados).
        num_datos (int): Número total de datos a generar.
        desv (float): Desviación típica del ruido blanco (asumido ~ N(0, desv)).

    Returns:
        np.ndarray: Serie temporal generada siguiendo el modelo ARMA estacional.
    """
    a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    p = len(phi)
    q = len(teta)
    maxi = max(p,q)
    
    for k in range(0,s*maxi):
        datos[k] = a[k] + c
            
    for i in range(s*maxi,num_datos):
        datos[i] = c + a[i]
            
        for j in range (1,p+1):
            datos[i]= datos[i] + phi[j-1]*datos[i-j*s]
                
        for k in range(1,q+1):
            datos[i] = datos[i] + teta[k-1]*a[i-k*s]   
                
    return datos

def creacion_modelos_ARMA(c,num_datos,desv,s=0,phi=[],teta=[]):
    """
    Genera una serie temporal basada en un modelo AR, MA, ARMA o sus versiones estacionales
    (SAR, SMA, SARMA) dependiendo de los parámetros proporcionados.

    La elección del modelo se realiza automáticamente según:
        - Si s = 0: modelo no estacional (AR, MA o ARMA).
        - Si s > 0: modelo estacional (AR estacional, MA estacional o ARMA estacional).

    Args:
        c (float): Término constante del modelo.
        num_datos (int): Número total de datos a generar.
        desv (float): Desviación estándar del ruido blanco.
        s (int, opcional): Periodo de estacionalidad. Por defecto es 0 (no estacional).
        phi (list, opcional): Coeficientes del componente autorregresivo (AR). Por defecto, lista vacía.
        teta (list, opcional): Coeficientes del componente de medias móviles (MA). Por defecto, lista vacía.

    Returns:
        np.ndarray: Serie temporal generada siguiendo el modelo correspondiente.
    """
    p = len(phi)
    q = len(teta)
    
    if s == 0:
        
        if q == 0:
            datos = modelo_AR(c,phi,num_datos,desv)    
        elif p == 0: 
            datos = modelo_MA(c,teta,num_datos,desv)
        else:
            datos = modelo_ARMA(c,phi,teta,num_datos,desv)
            
    else :
        
        if q == 0:
            datos = modelo_AR_estacional(c,phi,s,num_datos,desv)   
        elif p == 0: 
            datos = modelo_MA_estacional(c,teta,s,num_datos,desv)
        else:
            datos = modelo_ARMA_estacional(c,phi,teta,s,num_datos,desv)
            
    return datos 

# Series que sufren drift 

def crear_drift(params1,params2,tipo,num_drift,num_datos):
    """
    Genera una secuencia de datos que presenta un drift en un punto dado.
    
    Args:
        params1(tuple): Parámetros necesarios para generar los datos antes del drift. Su estructura depende del tipo.
        params2(tuple): Parámetros necesarios para generar los datos después del drift. Su estructura depende del tipo.
        tipo (int): Tipo de drift a generar (valores entre 1 y 16).
        num_drift (int): Número de datos antes del drift.
        num_datos (int): Número total de datos (antes + después del drift).

    Tipos de drift:
        1. Distribución → Distribución
        2. Distribución → ARMA
        3. Distribución → Periódico
        4. Distribución → Tendencia determinista
        5. ARMA → ARMA
        6. ARMA → Distribución
        7. ARMA → Periódico
        8. ARMA → Tendencia determinista
        9. Periódico → Periódico
        10. Periódico → Distribución
        11. Periódico → ARMA
        12. Periódico → Tendencia determinista
        13. Tendencia determinista → Tendencia determinista
        14. Tendencia determinista → Distribución
        15. Tendencia determinista → ARMA
        16. Tendencia determinista → Periódico

    Returns:
        np.ndarray: Un array de `num_datos` datos concatenando los segmentos antes y después del drift.
    """

    if tipo==1:
        distr1=params1[0]
        parametros1=params1[1]
        datos1 = crear_datos(distr1,parametros1,num_drift)
        
        distr2=params2[0]
        parametros2=params2[1]
        datos2 = crear_datos(distr2,parametros2,num_datos-num_drift)
        
    elif tipo==2:
        distr1=params1[0]
        parametros1=params1[1]
        datos1 = crear_datos(distr1,parametros1,num_drift)
        
        c = params2[0]
        desv = params2[1]
        s = params2[2]
        phi = params2[3]
        teta = params2[4]
        datos2 = creacion_modelos_ARMA(c,num_datos-num_drift,desv,s,phi,teta)
    
    elif tipo==3:
        distr1=params1[0]
        parametros1=params1[1]
        datos1 = crear_datos(distr1,parametros1,num_drift)
        
        tipo2 = params2[0]
        distr,parametros2,p = params2[1],params2[2],params2[3]
        if tipo2==1:
            datos2=datos_periodicos_amplitud(distr,parametros2,num_datos-num_drift,p)
        elif tipo2==2:
            datos2=datos_periodicos_cantidad(distr,parametros2,num_datos-num_drift,p)
            
    elif tipo==4:
        distr1=params1[0]
        parametros1=params1[1]
        datos1 = crear_datos(distr1,parametros1,num_drift)
        
        parametros2,tipo2,coef_error = params2[0],params2[1],params2[2]
        datos2=tendencia_det(parametros2,tipo2,num_datos-num_drift,coef_error)
    
    elif tipo==5:
        c = params1[0]
        desv = params1[1]
        s = params1[2]
        phi = params1[3]
        teta = params1[4]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta)
        
        c2 = params2[0]
        desv2 = params2[1]
        s2 = params2[2]
        phi2 = params2[3]
        teta2 = params2[4]
        datos2 = creacion_modelos_ARMA(c2,num_datos-num_drift,desv2,s2,phi2,teta2)
        
    elif tipo==6: 
        c = params1[0]
        desv = params1[1]
        s = params1[2]
        phi = params1[3]
        teta = params1[4]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta)
        
        distr2=params2[0]
        parametros2=params2[1]
        datos2 = crear_datos(distr2,parametros2,num_datos - num_drift)
  
        
    elif tipo==7: 
        c = params1[0]
        desv = params1[1]
        s = params1[2]
        phi = params1[3]
        teta = params1[4]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta)

        tipo2 = params2[0]
        distr,param,p = params2[1],params2[2],params2[3]
        if tipo2==1:
            datos2=datos_periodicos_amplitud(distr,param,num_datos-num_drift,p)
        elif tipo2==2:
            datos2=datos_periodicos_cantidad(distr,param,num_datos-num_drift,p)
            
    elif tipo==8:
        
        c = params1[0]
        desv = params1[1]
        s = params1[2]
        phi = params1[3]
        teta = params1[4]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta)
    
        param,tipo2,coef_error = params2[0],params2[1],params2[2]
        datos2 = tendencia_det(param,tipo2,num_datos-num_drift,coef_error)  
         
    elif tipo==9: 
        tipo1 = params1[0]
        distr,params,p = params1[1],params1[2],params1[3]
        if tipo1==1:
            datos1=datos_periodicos_amplitud(distr,params,num_drift,p)
        elif tipo1==2:
            datos1=datos_periodicos_cantidad(distr,params,num_drift,p)
            
        tipo2 = params2[0]
        distr,param,p = params2[1],params2[2],params2[3]
        if tipo2==1:
            datos2=datos_periodicos_amplitud(distr,param,num_datos-num_drift,p)
        elif tipo2==2:
            datos2=datos_periodicos_cantidad(distr,param,num_datos-num_drift,p)
    
    elif tipo==10:
        tipo1 = params1[0]
        distr,params,p = params1[1],params1[2],params1[3]
        if tipo1==1:
            datos1=datos_periodicos_amplitud(distr,params,num_drift,p)
        elif tipo1==2:
            datos1=datos_periodicos_cantidad(distr,params,num_drift,p)
            
        distr2=params2[0]
        parametros2=params2[1]
        datos2 = crear_datos(distr2,parametros2,num_datos - num_drift)   
        
    elif tipo == 11:
        tipo1 = params1[0]
        distr,params,p = params1[1],params1[2],params1[3]
        if tipo1==1:
            datos1=datos_periodicos_amplitud(distr,params,num_drift,p)
        elif tipo1==2:
            datos1=datos_periodicos_cantidad(distr,params,num_drift,p)
            
        c2 = params2[0]
        desv2 = params2[1]
        s2 = params2[2]
        phi2 = params2[3]
        teta2 = params2[4]
        datos2 = creacion_modelos_ARMA(c2,num_datos-num_drift,desv2,s2,phi2,teta2)  
          
    elif tipo==12:
        tipo1 = params1[0]
        distr,params,p = params1[1],params1[2],params1[3]
        if tipo1==1:
            datos1=datos_periodicos_amplitud(distr,params,num_drift,p)
        elif tipo1==2:
            datos1=datos_periodicos_cantidad(distr,params,num_drift,p)
            
        param,tipo2,coef_error = params2[0],params2[1],params2[2]
        datos2=tendencia_det(param,tipo2,num_datos-num_drift,coef_error)  
           
    elif tipo==13:
        params,tipo1,coef_error = params1[0],params1[1],params1[2]
        datos1=tendencia_det(params,tipo1,num_drift,coef_error) 
        
        param,tipo2,coef_error2 = params2[0],params2[1],params2[2]
        datos2=tendencia_det(param,tipo2,num_datos-num_drift,coef_error2) 
         
    elif tipo==14:
        params,tipo1,coef_error = params1[0],params1[1],params1[2]
        datos1=tendencia_det(params,tipo1,num_drift,coef_error) 
        
        distr2=params2[0]
        parametros2=params2[1]
        datos2 = crear_datos(distr2,parametros2,num_datos - num_drift)
        
    elif tipo==15:
        params,tipo1,coef_error = params1[0],params1[1],params1[2]
        datos1=tendencia_det(params,tipo1,num_drift,coef_error) 
        
        c2 = params2[0]
        desv2 = params2[1]
        s2 = params2[2]
        phi2 = params2[3]
        teta2 = params2[4]
        datos2 = creacion_modelos_ARMA(c2,num_datos-num_drift,desv2,s2,phi2,teta2)
        
    elif tipo==16: 
        params,tipo1,coef_error = params1[0],params1[1],params1[2]
        datos1=tendencia_det(params,tipo1,num_drift,coef_error) 
        
        tipo2 = params2[0]
        distr,param,p = params2[1],params2[2],params2[3]
        if tipo2==1:
            datos2=datos_periodicos_amplitud(distr,param,num_datos-num_drift,p)
        elif tipo2==2:
            datos2=datos_periodicos_cantidad(distr,param,num_datos-num_drift,p)
             
    datos = np.concatenate((datos1,datos2))
    return datos 

# Creación de dataframes a partir de las observaciones

def crear_df_fin_tend_det(inicio,fin,freq,columna,params,tipo,coef_error=0):
    """
    Crea un DataFrame con una serie temporal generada a partir de una tendencia determinista,
    según los parámetros y tipo especificados, y la plotea.

    Args:
        inicio (str): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        fin (str): Fecha de fin de la serie temporal (formato 'YYYY-MM-DD').
        freq (str): Frecuencia de los datos. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundos,
            'us' - microsegundos,
            'ns' - nanosegundos.
        columna (list of str): Lista con el nombre de la columna del DataFrame.
        params (list or tuple): Parámetros que definen la forma de la tendencia.
        tipo (int): Tipo de tendencia determinista a generar. Valores posibles:
            1. Tendencia lineal
            2. Tendencia polinómica
            3. Tendencia exponencial
            4. Tendencia logarítmica
        coef_error (float, optional): Coeficiente del término de error aleatorio. Por defecto es 0 (sin ruido).

    Returns:
        pd.DataFrame: DataFrame con índice temporal y una columna con los datos generados. También muestra la gráfica.
    """
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos = tendencia_det(params,tipo,num_datos,coef_error)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df
 
def crear_df_periodos_tend_det(inicio,periodos,freq,columna,params,tipo,coef_error=0):
    """
    Crea un DataFrame con una serie temporal basada en una tendencia determinista,
    generando las fechas a partir de un número de periodos desde una fecha inicial.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal.
        periodos (int): Número de periodos (fechas) que tendrá la serie.
        freq (str): Frecuencia de la serie temporal. Puede ser:
                    'B' (días hábiles), 'D' (días calendario),
                    'W' (semanal), 'M' (mensual), 'Q' (trimestral),
                    'Y' (anual), 'h' (hora), 'min' (minuto), 's' (segundo),
                    'ms' (milisegundo), 'us' (microsegundo), 'ns' (nanosegundo).
        columna (list): Lista con el/los nombre(s) de la(s) columna(s) del DataFrame.
        params (tuple o list): Parámetros de la tendencia determinista.
        tipo (int): Tipo de tendencia determinista, posibles valores:
                    1. Lineal
                    2. Polinómica
                    3. Exponencial
                    4. Logarítmica
        coef_error (float, opcional): Coeficiente de error a añadir en la generación de la tendencia.
                                      Valor por defecto es 0.

    Returns:
        pd.DataFrame: DataFrame con la serie temporal generada, indexado por fechas calculadas
                      desde `inicio` y con una columna de valores según la tendencia especificada.
    """
    indice = series_periodos(inicio,periodos,freq)
    num_datos = indice.size
    datos = tendencia_det(params,tipo,num_datos,coef_error)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df

def crear_df_fin_datos(inicio,fin,freq,columna,distr,params):
    """
    Crea un DataFrame con una serie temporal cuyos valores siguen una distribución específica.
    Los datos se generan a partir de una fecha de inicio y una fecha de fin con una frecuencia dada.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        fin (str o datetime): Fecha de fin de la serie temporal (formato 'YYYY-MM-DD').
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list of str): Lista con el nombre de la(s) columna(s) del DataFrame.
        distr (int): Tipo de distribución a utilizar para generar los datos. Las opciones son:
            1.  Normal
            2.  Binomial
            3.  Poisson
            4.  Geométrica
            5.  Hipergeométrica
            6.  Constante
            7.  Uniforme
            8.  Lognormal
            9.  Exponencial
            10. Gamma
            11. Beta
            12. Chi-cuadrado
            13. T-Student
            14. Pareto
            15. Lineal descendente
            16. Lineal ascendente
            17. Aleatorio entre dos valores
        params (list or tuple): Lista de parámetros específicos para la distribución seleccionada.
            La cantidad y tipo de parámetros depende de la distribución.

    Returns:
        pd.DataFrame: DataFrame con índice temporal y una columna cuyos valores siguen la distribución indicada.
                      También genera y muestra la gráfica de la serie.
    """
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos = crear_datos(distr,params,num_datos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_datos(inicio,periodos,freq,columna,distr,params):
    """
    Crea un DataFrame con una serie temporal generada a partir de un número fijo de periodos.
    Los datos se generan siguiendo una distribución específica y se asignan a fechas generadas a partir de la frecuencia indicada.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        periodos (int): Número total de observaciones (periodos) que tendrá la serie.
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list of str): Lista con el nombre de la(s) columna(s) del DataFrame.
        distr (int): Tipo de distribución de los datos, según los mismos casos definidos para `crear_df_fin_datos`.
        params (list o tuple): Parámetros específicos para la distribución seleccionada.

    Returns:
        pd.DataFrame: DataFrame con índice temporal y una columna de datos generados.
                      También se muestra la gráfica de la serie generada.
    """
    indice = series_periodos(inicio,periodos,freq)
    datos = crear_datos(distr,params,periodos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_fin_periodicos(inicio,fin,freq,columna,distr,params,p,tipo):
    """
    Crea un DataFrame con una serie temporal cuyos datos presentan comportamiento periódico.
    Los datos se generan en función de un rango de fechas (inicio-fin), una distribución y un patrón periódico.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        fin (str o datetime): Fecha de fin de la serie temporal (formato 'YYYY-MM-DD').
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list of str): Lista con el nombre de la(s) columna(s) del DataFrame.
        distr (int): Tipo de distribución de los datos, según los mismos casos definidos para `crear_df_fin_datos`.
        params (list o tuple): Parámetros específicos para la distribución seleccionada.
        p (int): Periodo de la variación periódica. Es decir, cada cuántas observaciones se repite el patrón.
        tipo (int): Tipo de patrón periódico:
            1. Periodicidad basada en la variación de la amplitud.
            2. Periodicidad basada en la variación de la cantidad de elementos.

    Returns:
        pd.DataFrame: DataFrame con índice temporal y una columna de datos periódicos.
                      También se muestra la gráfica de la serie generada.
    """
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    if tipo==1:
        datos = datos_periodicos_amplitud(distr,params,num_datos,p)
    elif tipo ==2:
        datos=datos_periodicos_cantidad(distr,params,num_datos,p)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_periodicos(inicio,periodos,freq,columna,distr,params,p,tipo):
    """
    Crea un DataFrame con una serie temporal cuyos datos presentan comportamiento periódico.
    Los datos se generan en función del número de periodos, una distribución y el tipo de periodicidad.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        periodos (int): Número total de observaciones a generar.
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list of str): Lista con el nombre de la(s) columna(s) del DataFrame.
        distr (int): Tipo de distribución de los datos, según los mismos casos que en la función `crear_df_fin_datos`.
        params (list or tuple): Parámetros específicos para la distribución seleccionada.
        p (int): Periodo de la variación periódica. Es decir, cada cuántas observaciones se repite el patrón.
        tipo (int): Tipo de patrón periódico:
            1. Periodicidad basada en la variación de la amplitud.
            2. Periodicidad basada en la variación de la cantidad de elementos.

    Returns:
        pd.DataFrame: DataFrame con índice temporal y una columna de datos periódicos.
                      También se muestra la gráfica de la serie generada.
    """
    indice = series_periodos(inicio,periodos,freq)
    num_datos = indice.size
    if tipo==1:
        datos = datos_periodicos_amplitud(distr,params,num_datos,p)
    elif tipo==2:
        datos=datos_periodicos_cantidad(distr,params,num_datos,p)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_fin_ARMA(inicio,fin,freq,columna,c,desv,s=0,phi=[],teta=[]):
    """
    Genera un DataFrame con una serie temporal basada en un modelo ARMA (AutoRegresivo y de Medias Móviles),
    con posible componente estacional. El índice se genera a partir de fechas entre 'inicio' y 'fin'
    con la frecuencia especificada.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        fin (str o datetime): Fecha de fin de la serie temporal.
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list of str): Lista con el nombre de la(s) columna(s) del DataFrame.
        c (float): Término constante del modelo ARMA.
        desv (float): Desviación típica para la generación del ruido blanco (cuando 'a' está vacío).
        s (int, opcional): Tamaño del componente estacional (número de datos por estación). Por defecto 0 (sin estacionalidad).
        phi (list of float, opcional): Coeficientes del modelo autorregresivo (AR).
        teta (list of float, opcional): Coeficientes del modelo de medias móviles (MA).

    Returns:
        pd.DataFrame: DataFrame con índice temporal y una columna de datos generados según el modelo ARMA.
                      También se muestra una gráfica de la serie generada.
    """
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos=creacion_modelos_ARMA(c,num_datos,desv,s,phi,teta)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_ARMA(inicio,periodos,freq,columna,c,desv,s=0,phi=[],teta=[]):
    """
    Genera un DataFrame con una serie temporal basada en un modelo ARMA (AutoRegresivo y de Medias Móviles),
    con posible componente estacional. El índice se construye a partir de una fecha de inicio y un número
    de periodos, con la frecuencia especificada.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        periodos (int): Número total de observaciones a generar.
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list of str): Lista con el nombre de la(s) columna(s) del DataFrame.
        c (float): Término constante del modelo ARMA.
        desv (float): Desviación típica para la generación del ruido blanco (cuando 'a' está vacío).
        s (int, opcional): Tamaño del componente estacional (número de datos por estación). Por defecto 0 (sin estacionalidad).
        phi (list of float, opcional): Coeficientes del modelo autorregresivo (AR).
        teta (list of float, opcional): Coeficientes del modelo de medias móviles (MA).

    Returns:
        pd.DataFrame: DataFrame con índice temporal y una columna de datos generados según el modelo ARMA.
                      También se muestra una gráfica de la serie generada.
    """
    indice = series_periodos(inicio,periodos,freq)
    datos=creacion_modelos_ARMA(c,periodos,desv,s,phi,teta)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_fin_DRIFT(inicio,fin,freq,columna,params1,params2,tipo,num_drift):
    """
    Genera un DataFrame con una serie temporal que incluye un drift (cambio estructural) 
    en los datos, utilizando diferentes tipos de modelos para los datos previos y posteriores al drift.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        fin (str o datetime): Fecha de fin de la serie temporal (formato 'YYYY-MM-DD').
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list of str): Lista con el nombre de la(s) columna(s) del DataFrame.
        params1 (varios): Parámetros para crear los datos previos al drift. Pueden ser:
            1. Distribución y sus parámetros (distr, parámetros).
            2. Parámetros de un modelo ARMA (c, phi, teta, a, s, desv).
            3. Parámetros para datos periódicos (tipo, distr, params, p).
            4. Parámetros para tendencia determinista (tipo, parámetros, coeficiente de error).
        params2 (varios): Parámetros para crear los datos posteriores al drift (con las mismas opciones que params1).
        tipo (int): Tipo de drift.
        num_drift (int): Número de datos previos al drift (número de datos antes del cambio).

    Returns:
        pd.DataFrame: DataFrame con índice temporal y columna(s) con la serie temporal que contiene un drift.
                      También se muestra una gráfica de la serie generada.
    """
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos = crear_drift(params1,params2,tipo,num_drift,num_datos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_DRIFT(inicio,periodos,freq,columna,params1,params2,tipo,num_drift):
    """
    Genera un DataFrame con una serie temporal que incluye un drift (cambio estructural) 
    basado en un número de periodos y frecuencia, utilizando diferentes tipos de modelos 
    para los datos previos y posteriores al drift.

    Args:
        inicio (str o datetime): Fecha de inicio de la serie temporal (formato 'YYYY-MM-DD').
        periodos (int): Número de periodos que tendrá la serie temporal.
        freq (str): Frecuencia de la serie temporal. Valores posibles:
            'B'  - días hábiles (business days),
            'D'  - días calendario,
            'W'  - semanal,
            'M'  - mensual,
            'Q'  - trimestral,
            'Y'  - anual,
            'h'  - por hora,
            'min' - por minuto,
            's'  - por segundo,
            'ms' - milisegundo,
            'us' - microsegundo,
            'ns' - nanosegundo.
        columna (list de str): Lista con el nombre de la(s) columna(s) del DataFrame.
        params1 (varios): Parámetros para crear los datos previos al drift. Opciones:
            1. Distribución y parámetros (distr, parámetros).
            2. Parámetros de un modelo ARMA (c, phi, teta, a, s, desv).
            3. Parámetros para datos periódicos (tipo, distr, params, p).
            4. Parámetros para tendencia determinista (tipo, parámetros, coeficiente de error).
        params2 (varios): Parámetros para crear los datos posteriores al drift (con las mismas opciones que params1).
        tipo (int): Tipo de drift.
        num_drift (int): Número de datos previos al drift (número de datos antes del cambio).

    Returns:
        pd.DataFrame: DataFrame con índice temporal y columna(s) con la serie temporal que contiene un drift.
                      También muestra una gráfica con la serie generada.
    """
    indice = series_periodos(inicio,periodos,freq)
    datos = crear_drift(params1,params2,tipo,num_drift,periodos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_fin(inicio,fin,freq,columna,parametros,tipo):
    """
    Genera un DataFrame con una serie temporal desde una fecha de inicio hasta una fecha de fin, 
    utilizando diferentes métodos para crear las observaciones.

    Args:
        inicio (str o pd.Timestamp): Fecha de inicio de la serie temporal.
        fin (str o pd.Timestamp): Fecha de fin de la serie temporal.
        freq (str): Frecuencia temporal (e.g., 'B', 'D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'us', 'ns').
        columna (list o array-like): Lista con el nombre de la(s) columna(s) del DataFrame resultante.
        parametros (tuple): Parámetros específicos para el método de generación seleccionado. Según el `tipo`:
            1. Distribución: (distr, params) — tipo de distribución y parámetros asociados.
            2. ARMA: (c, desv, s, phi, teta, a) — constantes, desviación del ruido blanco, estacionalidad, parámetros AR y MA, ruido blanco.
            3. Modelos periódicos: (tipo_per, distr, params, p) — tipo de periodicidad, distribución y sus parámetros, número o amplitud de periodos.
            4. Tendencia determinista: (params, tipo_tend, coef_error) — parámetros del modelo de tendencia, tipo de tendencia y coeficiente de error.
            5. Drift: (params1, params2, tipo_drift, num_drift) — parámetros para datos antes y después del drift, tipo de drift y cantidad de datos previos al drift.
        tipo (int): Indica el método de generación de datos a usar (1 a 5).

    Returns:
        pd.DataFrame: DataFrame con la serie temporal generada, indexada por fechas según `inicio`, `fin` y `freq`, con las columnas especificadas en `columna`.
                      También muestra una gráfica con la serie generada.

    """
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    
    if tipo==1: # DISTRIBUCIÓN
        distr = parametros[0]
        params = parametros[1]
        datos = crear_datos(distr,params,num_datos)
        
    elif tipo ==2: # ARMA
        c = parametros[0]
        desv = parametros[1]
        s = parametros[2]
        phi = parametros[3]
        teta = parametros[4]
        a = parametros[5]
        datos = creacion_modelos_ARMA(c,num_datos,desv,s,phi,teta,a)
        
    elif tipo == 3: # MODELOS PERIÓDICOS
        tipo_per=parametros[0]
        distr = parametros[1]
        params = parametros[2]
        p = parametros[3]
        if tipo_per==1:
            datos = datos_periodicos_amplitud(distr,params,num_datos,p)
        elif tipo_per==2:
            datos = datos_periodicos_cantidad(distr,params,num_datos,p)
            
    elif tipo ==4: # TENDENCIA DETERMINISTA     
        params = parametros[0]
        tipo_tend = parametros[1]
        coef_error = parametros[2]
        datos = tendencia_det(params,tipo_tend,num_datos,coef_error)
    
    elif tipo == 5 : #DRIFT 
        params1 = parametros[0] 
        params2 = parametros[1]
        tipo_drift = parametros[2]
        num_drift = parametros[3]
        datos = crear_drift(params1,params2,tipo_drift,num_drift,num_datos)
        
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df

def crear_df_periodos(inicio,periodos,freq,columna,parametros,tipo):
    """
    Genera un DataFrame con una serie temporal desde una fecha de inicio y una cantidad de periodos, 
    utilizando diferentes métodos para crear las observaciones.

    Args:
        inicio (str o pd.Timestamp): Fecha de inicio de la serie temporal.
        periodos (int): Número de periodos que tendrá la serie temporal.
        freq (str): Frecuencia temporal (e.g., 'B', 'D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'us', 'ns').
        columna (list o array-like): Lista con el nombre de la(s) columna(s) del DataFrame resultante.
        parametros (tuple): Parámetros específicos para el método de generación seleccionado. Según el `tipo`:
            1. Distribución: (distr, params) — tipo de distribución y parámetros asociados.
            2. ARMA: (c, desv, s, phi, teta, a) — constantes, desviación del ruido blanco, estacionalidad, parámetros AR y MA, ruido blanco.
            3. Modelos periódicos: (tipo_per, distr, params, p) — tipo de periodicidad, distribución y sus parámetros, número o amplitud de periodos.
            4. Tendencia determinista: (params, tipo_tend, coef_error) — parámetros del modelo de tendencia, tipo de tendencia y coeficiente de error.
            5. Drift: (params1, params2, tipo_drift, num_drift) — parámetros para datos antes y después del drift, tipo de drift y cantidad de datos previos al drift.
        tipo (int): Indica el método de generación de datos a usar (1 a 5).

    Returns:
        pd.DataFrame: DataFrame con la serie temporal generada, indexada por fechas según `inicio`, `periodos` y `freq`, con las columnas especificadas en `columna`.
                      También muestra una gráfica con la serie generada.
    """
    indice = series_periodos(inicio,periodos,freq)
    num_datos = indice.size
    
    if tipo==1: # DISTRIBUCIÓN
        distr = parametros[0]
        params = parametros[1]
        datos = crear_datos(distr,params,num_datos)
        
    elif tipo ==2: # ARMA
        c = parametros[0]
        desv = parametros[1]
        s = parametros[2]
        phi = parametros[3]
        teta = parametros[4]
        a = parametros[5]
        datos = creacion_modelos_ARMA(c,num_datos,desv,s,phi,teta,a)
        
    elif tipo == 3: # MODELOS PERIÓDICOS
        tipo_per=parametros[0]
        distr = parametros[1]
        params = parametros[2]
        p = parametros[3]
        if tipo_per==1:
            datos = datos_periodicos_amplitud(distr,params,num_datos,p)
        elif tipo_per==2:
            datos = datos_periodicos_cantidad(distr,params,num_datos,p)
            
    elif tipo ==4: # TENDENCIA DETERMINISTA     
        params = parametros[0]
        tipo_tend = parametros[1]
        coef_error = parametros[2]
        datos = tendencia_det(params,tipo_tend,num_datos,coef_error)
    
    elif tipo == 5 : #DRIFT 
        params1 = parametros[0] 
        params2 = parametros[1]
        tipo_drift = parametros[2]
        num_drift = parametros[3]
        datos = crear_drift(params1,params2,tipo_drift,num_drift,num_datos)
        
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df

# Exportación de datos de un dataframe a csv y viceversa

def csv_df(df, folder, file):
    """
    Guarda un DataFrame en un archivo CSV dentro de una carpeta especificada.
    Crea la carpeta si no existe.

    Args:
        df (pd.DataFrame): DataFrame que se desea guardar.
        folder (str): Nombre o ruta de la carpeta donde se guardará el archivo.
        file (str): Nombre del archivo (sin extensión) donde se guardarán los datos.

    Returns:
        None
    """
    filepath = Path(folder + '/'+ file +'.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath,index_label='indice')

def df_csv(folder,file,indice):
    """
    Carga un archivo CSV en un DataFrame, usando una columna específica como índice.

    Args:
        folder (str): Nombre o ruta de la carpeta donde está guardado el archivo CSV.
        file (str): Nombre del archivo (sin extensión) que se quiere cargar.
        indice (str): Nombre de la columna que se usará como índice del DataFrame.

    Returns:
        pd.DataFrame: DataFrame cargado desde el archivo CSV con el índice especificado.
    """
    return pd.read_csv(folder+'/'+file+'.csv',index_col=indice)

# Multivariantes:

def crear_df_multi_fin(inicio,fin,freq,columnas,parameters,tipos):
    """
    Crea un DataFrame con múltiples columnas de series temporales para un rango temporal definido por fecha de inicio y fin.

    Args:
        inicio (str|datetime): Fecha de inicio de la serie temporal.
        fin (str|datetime): Fecha de fin de la serie temporal.
        freq (str): Frecuencia temporal (B, D, W, M, Q, Y, h, min, s, ms, us, ns).
        columnas (list[str]): Lista con nombres de las columnas.
        parameters (list[list]): Lista con los parámetros para cada columna según el tipo de modelo.
        tipos (list[int]): Lista con el tipo de modelo para cada columna. 
            Tipos posibles:
            1 - Distribución
            2 - ARMA
            3 - Modelos periódicos
            4 - Tendencia determinista
            5 - Drift

    Returns:
        pd.DataFrame: DataFrame con las series temporales generadas para cada columna indexado por fecha.
    """
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    
    for k in range(0, len(tipos)): 
        
        tipo = tipos[k]  
        parametros = parameters[k]
        columna = columnas[k]
        
        if tipo==1: # DISTRIBUCIÓN
            distr = parametros[0]
            params = parametros[1]
            datos_n = crear_datos(distr,params,num_datos)
            
        elif tipo ==2: # ARMA
            c = parametros[0]
            desv = parametros[1]
            s = parametros[2]
            phi = parametros[3]
            teta = parametros[4]
            a = parametros[5]
            datos_n = creacion_modelos_ARMA(c,num_datos,desv,s,phi,teta,a)
            
        elif tipo == 3: # MODELOS PERIÓDICOS
            tipo_per=parametros[0]
            distr = parametros[1]
            params = parametros[2]
            p = parametros[3]
            if tipo_per==1:
                datos_n = datos_periodicos_amplitud(distr,params,num_datos,p)
            elif tipo_per==2:
                datos_n = datos_periodicos_cantidad(distr,params,num_datos,p)
                
        elif tipo ==4: # TENDENCIA DETERMINISTA     
            params = parametros[0]
            tipo_tend = parametros[1]
            coef_error = parametros[2]
            datos_n = tendencia_det(params,tipo_tend,num_datos,coef_error)
        
        elif tipo == 5 : # DRIFT 
            params1 = parametros[0] 
            params2 = parametros[1]
            tipo_drift = parametros[2]
            num_drift = parametros[3]
            datos_n = crear_drift(params1,params2,tipo_drift,num_drift,num_datos)
            
        if k==0:
            df = pd.DataFrame(data=datos_n,index=indice,columns=[columna])
        else :     
            df_n = pd.DataFrame(data=datos_n,index=indice,columns=[columna])
            df= df.join(df_n, how="outer")
        
    
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df

def crear_df_multi_periodos(inicio,periodos,freq,columnas,parameters,tipos):
    """
    Crea un DataFrame con múltiples columnas de series temporales para un número fijo de periodos.

    Args:
        inicio (str|datetime): Fecha de inicio de la serie temporal.
        periodos (int): Número total de periodos para la serie temporal.
        freq (str): Frecuencia temporal (B, D, W, M, Q, Y, h, min, s, ms, us, ns).
        columnas (list[str]): Lista con nombres de las columnas.
        parameters (list[list]): Lista con los parámetros para cada columna según el tipo de modelo.
        tipos (list[int]): Lista con el tipo de modelo para cada columna. 
            Tipos posibles:
            1 - Distribución
            2 - ARMA
            3 - Modelos periódicos
            4 - Tendencia determinista
            5 - Drift

    Returns:
        pd.DataFrame: DataFrame con las series temporales generadas para cada columna indexado por periodo.
    """
    indice = series_periodos(inicio,periodos,freq)
    num_datos = indice.size
    
    for k in range(0, len(tipos)): 
        
        columna = columnas[k]
        tipo = tipos[k]  
        parametros = parameters[k]
        
        if tipo==1: # DISTRIBUCIÓN
            distr = parametros[0]
            params = parametros[1]
            datos_n = crear_datos(distr,params,num_datos)
            
        elif tipo ==2: # ARMA
            c = parametros[0]
            desv = parametros[1]
            s = parametros[2]
            phi = parametros[3]
            teta = parametros[4]
            a = parametros[5]
            datos_n = creacion_modelos_ARMA(c,num_datos,desv,s,phi,teta,a)
            
        elif tipo == 3: # MODELOS PERIÓDICOS
            tipo_per=parametros[0]
            distr = parametros[1]
            params = parametros[2]
            p = parametros[3]
            if tipo_per==1:
                datos_n = datos_periodicos_amplitud(distr,params,num_datos,p)
            elif tipo_per==2:
                datos_n = datos_periodicos_cantidad(distr,params,num_datos,p)
                
        elif tipo ==4: # TENDENCIA DETERMINISTA     
            params = parametros[0]
            tipo_tend = parametros[1]
            coef_error = parametros[2]
            datos_n = tendencia_det(params,tipo_tend,num_datos,coef_error)
        
        elif tipo == 5 : # DRIFT 
            params1 = parametros[0] 
            params2 = parametros[1]
            tipo_drift = parametros[2]
            num_drift = parametros[3]
            datos_n = crear_drift(params1,params2,tipo_drift,num_drift,num_datos)
        
        if k==0:
            df = pd.DataFrame(data=datos_n,index=indice,columns=[columna])
        else :     
            df_n = pd.DataFrame(data=datos_n,index=indice,columns=[columna])
            df= df.join(df_n, how="outer")
        
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df