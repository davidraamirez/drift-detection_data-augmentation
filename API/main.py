from functools import partial
from io import StringIO
import io
import json
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import StreamingResponse

# Imports
import pandas as pd
import numpy as np
from pathlib import Path  
from scipy.stats import binom,poisson,geom,hypergeom,uniform,expon, gamma, beta,chi2,t,pareto,lognorm
from random import randrange, random
import math
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

app = FastAPI()

@app.get("/")
def read_root():
    return {"Contenido": "Esto es una API para generar datos sintéticos a partir de ciertos parámetros, aumentación de datos y generación de variables objetivos."}

# Creación de índices
def series_fin(inicio, fin,freq):
    serie = pd.date_range(start=inicio, end=fin, freq=freq)
    return serie

def series_periodos(inicio, periodos, freq): 
    serie = pd.date_range(start=inicio, periods=periodos, freq=freq)
    return serie

# Creación csv 
def pasar_csv(df):
    return df.to_csv()
    
# Modelos de tendencia determinista
def tendencia_lineal (a,b,t):
    return a + b * t 

def tendencia_determinista_lineal (a,b,t,e=0):
    return tendencia_lineal(a,b,t) + e

def tendencia_determinista_polinómica(params,t,e=0):
    res = params[0]
    for k in range(1,len(params)):
        res = res + params[k] * t**k
    return res + e

def tendencia_determinista_exponencial(a,b,t,e=0):
    return math.exp(a+b*t+e)

def tendencia_determinista_logaritmica(a,b,t,e=0):
    return a + b * math.log(t) + e

def tendencia_det(params,tipo,num_datos,coef_error=0):
    
    datos = np.zeros(num_datos)
    
    for t in range(1,num_datos+1):

        e = random() * coef_error
        
        if tipo==1:
            datos[t-1] = tendencia_determinista_lineal(params[0],params[1],t,e)
        elif tipo==2:
            datos[t-1] = tendencia_determinista_polinómica(params,t,e)
        elif tipo==3:
            datos[t-1] = tendencia_determinista_exponencial(params[0],params[1],t,e)
        elif tipo==4:
            datos[t-1] = tendencia_determinista_logaritmica(params[0],params[1],t,e)
            
    return datos

# Creación de dataframes de modelos deterministas 
def crear_df_fin_tend_det(inicio,fin,freq,columna,params,tipo,coef_error=0):
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos = tendencia_det(params,tipo,num_datos,coef_error)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df

def crear_df_periodos_tend_det(inicio,periodos,freq,columna,params,tipo,coef_error=0):
    indice = series_periodos(inicio,periodos,freq)
    num_datos = indice.size
    datos = tendencia_det(params,tipo,num_datos,coef_error)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df

def plot_df(df):
    plt.figure()
    df.plot(title="Serie temporal",figsize=(13,5))
    plt.xlabel("Tiempo")  
    
@app.get("/Report/tendencia/fin")
def obtener_report(inicio: str, fin:str, freq:str, tipo:int , error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la tendencia")):
   if tipo == 1:
       subtipo = "lineal"
       tendencia= "La serie es de tipo y = a + t * b + e0 donde a = " + str(params[0]) + " b = " +str (params[1]) +" e0 = " + str(error)
   elif tipo ==2:
       subtipo ="polinómica de grado "+ str(len(params)-1)
       tendencia= "La serie es de tipo y = a + b[1] * t + e0"  
       for k in range (2,len(params)):
           tendencia = " + b ["+str(k)+"] *  t ** " + str(k)
       tendencia = tendencia + "donde a = " + str(params[0]) + " b[1] = " + str (params[1])
       for k in range (2,len(params)):
           tendencia = tendencia  + " b["+ str(k)+"] = " + str (params[k])
       tendencia = tendencia +" e0 = " + str(error)
   elif tipo == 3: 
       subtipo ="exponencial"
       tendencia = "La serie es de tipo y = e ** (a + b*t + e0) donde a = " + str(params[0]) + ", b = " + str(params[1]) + " y e0 = " + str(error)
   elif tipo == 4:
       subtipo = "logaritmica" 
       tendencia = "La serie es de tipo y = a + b * log(t) donde a = " + str(params[0]) + " b = " + str(params[1]) + " e0 = " + str(error)

   tipos = "Modelo de tendencia determinista con tendencia " + subtipo
   explicacion = "Inicio: fecha de inicio " + str(inicio)
   explicacion = explicacion +" . Fin: fecha de fin --> "+ str(fin)
   explicacion = explicacion + " . Freq: frequencia de la serie temporal --> " + str(freq)
   explicacion = explicacion + ". Tipo: lineal(1), polinómica(2), exponencial(3), logarítmica(4) --> " + str(tipo)
   explicacion = explicacion + " . Error: coeficiente de error (e0) --> " + str(error)
   explicacion = explicacion + " . Columna: nombre de la columnas --> " + columna[0]
   for k in range (1, len (columna)):
       explicacion = explicacion+", " + columna [k]
   explicacion = explicacion + " . Params: parámetros de la tendencia, a = params [0] y b[k] = params[k] --> "+str(params [0])
   for k in range (1, len (params)):
       explicacion = explicacion+", " + str(params [k])
   return {"Tipo": tipos, "Serie" : tendencia, "Parámetros" : explicacion }

@app.get("/Datos/tendencia/fin")
async def obtener_serie(inicio: str, fin:str, freq:str, tipo:int , error: Union[float, None] = None, columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la tendencia")):
    
    df = crear_df_fin_tend_det(inicio,fin,freq,columna,params,tipo,error)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)
    texto = "Probando"
    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-tendencia-fin.csv"
    response.headers["X-Text-Info"] = texto
    
    return response

@app.get("/Plot/tendencia/fin")
async def obtener_grafica(inicio: str, fin:str, freq:str, tipo:int , error: Union[float, None] = None, columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la tendencia")):
    
    df = crear_df_fin_tend_det(inicio,fin,freq,columna,params,tipo,error)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Report/tendencia/periodos")
def obtener_report(inicio: str, periodos: int, freq:str, tipo:int , error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la tendencia")):
    if tipo == 1:
        subtipo = "lineal"
        tendencia= "La serie es de tipo y = a + t * b + e0 donde a = " + str(params[0]) + " b = " +str (params[1]) +" e0 = " + str(error)
    elif tipo ==2:
        subtipo ="polinómica de grado "+ str(len(params)-1)
        tendencia= "La serie es de tipo y = a + b[1] * t"  
        for k in range (2,len(params)):
            tendencia = tendencia+ " + b ["+str(k)+"] *  t ** " + str(k)
        tendencia = " +e0"
        tendencia = tendencia + " donde a = " + str(params[0]) + ", b[1] = " + str (params[1])
        for k in range (2,len(params)):
            tendencia = tendencia  + ", b["+ str(k)+"] = " + str (params[k])
        tendencia = tendencia +" y e0 = " + str(error)
    elif tipo == 3: 
        subtipo ="exponencial"
        tendencia = "La serie es de tipo y = e ** (a + b*t + e0) donde a = " + str(params[0]) + ", b = " + str(params[1]) + " y e0 = " + str(error)
    elif tipo == 4:
        subtipo = "logaritmica" 
        tendencia = "La serie es de tipo y = a + b * log(t) donde a = " + str(params[0]) + " b = " + str(params[1]) + " e0 = " + str(error)

    tipos = "Modelo de tendencia determinista con tendencia " + subtipo
    explicacion = "Inicio: fecha de inicio " + str(inicio)
    explicacion = explicacion +". Periodos: número de de periodos a generar --> "+ str(periodos)
    explicacion = explicacion + ". Freq: frequencia de la serie temporal --> " + str(freq)
    explicacion = explicacion + ". Tipo: lineal(1), polinómica(2), exponencial(3), logarítmica(4) --> " + str(tipo)
    explicacion = explicacion + ". Error: coeficiente de error (e0) --> " + str(error)
    explicacion = explicacion + ". Columna: nombre de la columnas --> " + columna[0]
    for k in range (1, len (columna)):
        explicacion = explicacion+", " + columna [k]
    explicacion = explicacion + ". Params: parámetros de la tendencia, a = params [0] y b[k] = params[k] --> "+str(params [0])
    for k in range (1, len (params)):
        explicacion = explicacion+", " + str(params [k])
    return {"Tipo": tipos, "Serie" : tendencia, "Parámetros" : explicacion }


@app.get("/Datos/tendencia/periodos")
async def obtener_serie(inicio: str, periodos:int, freq:str, tipo:int , error: Union[float, None] = None,  columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float]= Query(...,description="Parametros de la tendencia")):
    
    df=crear_df_periodos_tend_det(inicio,periodos,freq,columna,params,tipo,error)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-tendencia-periodos.csv"
    
    return response

@app.get("/Plot/Tendencia/periodos")   
async def obtener_grafica(inicio: str, periodos:int, freq:str, tipo:int , error: Union[float, None] = None,  columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float]= Query(...,description="Parametros de la tendencia")):

    df = crear_df_periodos_tend_det(inicio,periodos,freq,columna,params,tipo,error)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse( buffer,media_type="image/png")

# Modelos con ciertas distribuciones:

def crear_datos(distr,params,num_datos):
    
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

def crear_df_fin_datos(inicio,fin,freq,columna,distr,params):
    
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos = crear_datos(distr,params,num_datos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_datos(inicio,periodos,freq,columna,distr,params):
    
    indice = series_periodos(inicio,periodos,freq)
    datos = crear_datos(distr,params,periodos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 


@app.get("/Report/distribuciones/fin")
def obtener_report(inicio: str, fin: str, freq:str, distr:int , columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):
    
    if distr == 1 :
        subtipo = "normal"
        parametros ="Modelo con media = params[0] y desviación típica = params[1]. La media es " + str(params[0])+ " y la desviación típica es " + str(params[1])
        mean = params[0]
        var = params[1] **2
    elif distr ==2 :
        subtipo = "binomial"
        parametros = "Modelo con n = params[0] y p = params[1] donde n = número de pruebas y p = probabilidad de éxito. El valor de n es " + str(params[0])+" y el valor de p es "+str(params[1])+ ". Adicionalmente, se puede añadir un desplazamiento params[2] : "
        if len(params)==2:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 3:
            parametros += "en este caso es desplazamiento es de " + str(params[2])
        mean, var = binom.stats(params[0], params[1], moments='mv')
        if len (params) == 3 :
           mean += params[2]
    elif distr== 3 :
        subtipo = "poisson"
        parametros = "Modelo con mu = params[0] donde mu = parámetro de poisson . El valor de mu es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var= poisson.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
           
    elif distr == 4 :
        subtipo = "geométrica"
        parametros = "Modelo con p = params[0] donde p = probabilidad de éxito. El valor de p es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var = geom.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
            
    elif distr == 5:
        subtipo = "hipergeométrica"
        parametros = "Modelo con M = params[0], n = params[1] y N = params[2], donde M = tamaño población, n = exitosos en la población y N = tamaño muesta. El valor de M es " + str(params[0])+", el valor de n es " + str(params[1])+" y el valor de N es " + str(params[2])+". Adicionalmente, se puede añadir un desplazamiento params[3] : "
        if len(params)==3:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 4:
            parametros += "en este caso es desplazamiento es de " + str(params[3])
        mean, var= hypergeom.stats(params[0], params[1],params[2], moments='mv')
        if len (params) == 4 :
           mean += params[3]
            
    elif distr == 6: 
        subtipo ="constante"
        parametros = "Modelo con constante = " + str(params[0])
        mean = params[0]
        var = 0
        
    elif distr == 7:
        subtipo = "uniforme"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución uniforme y obteniendo una distribucion uniforme [despl,despl+escala],"
        if len(params)==0:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += " en este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= uniform.stats( moments='mv')
        if len (params) == 1 :
           mean = mean + params[0]
        elif len (params) == 2:
            mean = mean* params[1]
            mean += params[0]
            var = params[1]**2/12
            
    elif distr == 8:
        subtipo = "lognormal"
        parametros = "Modelo con s = params[0] donde s es el parámetro de la distribución lognormal. El valor de s es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución lognormal y escala = escalado de la distribución "
        if len(params)==1:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[1])
        elif len(params) == 3:
            parametros += " en este caso el desplazamiento es de " + str(params[1]) +" y la escala de "+str(params[2])
        mean, var= lognorm.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = lognorm.mean(params[0], loc=params[1])
            var = lognorm.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = lognorm.mean(params[0], loc=params[1],scale=params[2])
            var = lognorm.var(params[0], loc=params[1], scale=params[2])
            
    elif distr == 9: 
        subtipo = "exponencial"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución exponencial y escala = escalado de la distribución. "
        if len(params)==0:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= expon.stats( moments='mv')
        if len (params) == 1 :
            mean = expon.mean(loc=params[0])
        elif len (params) == 2:
            mean = expon.mean(loc=params[0],scale=params[1])
            var = expon.var(scale=params[1])
            
    elif distr == 10: 
        subtipo = "gamma"
        parametros = "Modelo con a = params[0] donde a es el parámetro de la distribución gamma. El valor de a es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución gamma y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[1])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[1]) +" y la escala de "+str(params[2])
        mean, var= gamma.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = gamma.mean(params[0], loc=params[1])
            var = gamma.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = gamma.mean(params[0], loc=params[1],scale=params[2])
            var = gamma.var(params[0], scale=params[2])
            
    elif distr == 11: 
        subtipo = "beta"
        parametros = "Modelo con a = params[0] y b = params[1] donde a y b son los parámetros de la distribución beta. El valor de a es "+ str(params[0])+ "y el de b es "+ str(params[1])+ ". Además, posee los parametros opcionales: despl = params[] y escala = params[1], donde despl= desplazamiento de la distribución beta y escala = escalado de la distribución. "
        if len(params)==2:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[2])
        elif len(params) == 4:
            parametros += "En este caso el desplazamiento es de " + str(params[2]) +" y la escala de "+str(params[3])
        mean, var= beta.stats(params[0],params[1], moments='mv')
        if len (params) == 3:
            mean = beta.mean(params[0],params[1], loc=params[2])
            var = beta.var(params[0],params[1], loc = params[2])
        elif len (params) == 4:
            mean = beta.mean(params[0],params[1], loc=params[2],scale=params[3])
            var = beta.var(params[0],params[1], loc=params[2], scale=params[3])
            
    elif distr == 12: 
        subtipo = "chi cuadrado"
        parametros = "Modelo con df = params[0] donde df es el parámetro de la distribución chi cuadrado. El valor de df es "+ str(params[0]) +". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución chi2 y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[1])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[1]) +" y la escala de "+str(params[2])

        mean, var= chi2.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = chi2.mean(params[0], loc=params[1])
            var = chi2.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = chi2.mean(params[0], loc=params[1],scale=params[2])
            var = chi2.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 13: 
        subtipo = "t-student"
        parametros = "Modelo con v = params[0] donde v es el parámetro de la distribución t-student. El valor de t es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= t.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = t.mean(params[0], loc=params[1])
            var = t.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = t.mean(params[0], loc=params[1],scale=params[2])
            var = t.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 14: 
        subtipo = "pareto"
        parametros = "Modelo con b = params[0] donde b es el parámetro de la distribución pareto. El valor de b es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[1])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[1]) +" y la escala de "+str(params[2])
        mean, var= pareto.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = pareto.mean(params[0], loc=params[1])
            var = pareto.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = pareto.mean(params[0], loc=params[1],scale=params[2])
            var = pareto.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 15:
        subtipo = "linealmente decreciente"
        parametros = "Modelo de tipo: y_i = y_i-1 - b, y_0 = a donde a = params[0] = "+ str(params[0])+" y b = params[1] = "+ str(params[1])
        mean = "Información no relevante"
        var = "Información no relevante"
            
    elif distr == 16:
        subtipo = "linealmente creciente"
        parametros = "Modelo de tipo: y_i = y_i-1 + b, y_0 = a donde a = params[0] = "+ str(params[0])+" y b = params[1] = "+ str(params[1])
        mean = "Información no relevante"
        var = "Información no relevante"
    
    elif distr == 17:
        subtipo = "random"
        parametros = "Modelo con una distribución con valores aleatorios entre params[0] = " + str(params[0]) +" y params [1] =" + str(params[1])
        mean = "Información no relevante"
        var = "Información no relevante"

    tipos = "Modelo con una distribución " + subtipo
    explicacion = "Inicio: fecha de inicio --> " + str(inicio)
    explicacion = explicacion +". Fin: fecha de fin --> "+ str(fin)
    explicacion = explicacion + ". Freq: frequencia de la serie temporal --> " + str(freq)
    explicacion = explicacion + ". Distr: normal (1), binomial(2), poisson(3), geométrica(4), hipergeométrica(5), constante(6), uniforme(7), lognormal(8), exponencial(9), gamma(10), beta (11), chi cuadrado (12), t-student(13), pareto (14), linealmente decreciente(15), linealmente creciente (16) y random (17) --> " + str(distr)
    explicacion = explicacion + ". Columna: nombre de la columnas --> " + columna[0]
    for k in range (1, len (columna)):
        explicacion = explicacion+", " + columna [k]
    explicacion = explicacion + ". Params: parámetros de la distribución --> "+str(params [0])
    for k in range (1, len (params)):
        explicacion = explicacion+", " + str(params [k])
    return {"Tipo": tipos,"Parametros de la distribución": parametros, "Parámetros de la query" : explicacion, "Media" :mean, "Varianza" : var}

@app.get("/Datos/distribucion/fin")
async def obtener_serie(inicio: str, fin:str, freq:str, distr:int , columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):

    df = crear_df_fin_datos(inicio,fin,freq,columna,distr,params)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-distribucion-fin.csv"
    
    return response

@app.get("/Plot/distribuciones/fin")
async def obtener_grafica(inicio: str, fin:str, freq:str, distr:int , columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):
    df = crear_df_fin_datos(inicio,fin,freq,columna,distr,params)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Report/distribuciones/periodos")
def obtener_report(inicio: str, periodos: int, freq:str, distr:int , columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la distribución")):
    
    if distr == 1 :
        subtipo = "normal"
        parametros ="Modelo con media = params[0] y desviación típica = params[1]. La media es " + str(params[0])+ " y la desviación típica es " + str(params[1])
        mean = params[0]
        var = params[1] **2
    elif distr ==2 :
        subtipo = "binomial"
        parametros = "Modelo con n = params[0] y p = params[1] donde n = número de pruebas y p = probabilidad de éxito. El valor de n es " + str(params[0])+" y el valor de p es "+str(params[1])+ ". Adicionalmente, se puede añadir un desplazamiento params[2] : "
        if len(params)==2:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 3:
            parametros += "en este caso es desplazamiento es de " + str(params[2])
        mean, var = binom.stats(params[0], params[1], moments='mv')
        if len (params) == 3 :
           mean += params[2]
    elif distr== 3 :
        subtipo = "poisson"
        parametros = "Modelo con mu = params[0] donde mu = parámetro de poisson . El valor de mu es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var= poisson.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
           
    elif distr == 4 :
        subtipo = "geométrica"
        parametros = "Modelo con p = params[0] donde p = probabilidad de éxito. El valor de p es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var = geom.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
            
    elif distr == 5:
        subtipo = "hipergeométrica"
        parametros = "Modelo con M = params[0], n = params[1] y N = params[2], donde M = tamaño población, n = exitosos en la población y N = tamaño muesta. El valor de M es " + str(params[0])+", el valor de n es " + str(params[1])+" y el valor de N es " + str(params[2])+". Adicionalmente, se puede añadir un desplazamiento params[3] : "
        if len(params)==3:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 4:
            parametros += "en este caso es desplazamiento es de " + str(params[3])
        mean, var= hypergeom.stats(params[0], params[1],params[2], moments='mv')
        if len (params) == 4 :
           mean += params[3]
            
    elif distr == 6: 
        subtipo ="constante"
        parametros = "Modelo con constante = " + str(params[0])
        mean = params[0]
        var = 0
        
    elif distr == 7:
        subtipo = "uniforme"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución uniforme y obteniendo una distribucion uniforme [despl,despl+escala],"
        if len(params)==0:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += " en este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= uniform.stats( moments='mv')
        if len (params) == 1 :
           mean = mean + params[0]
        elif len (params) == 2:
            mean = mean* params[1]
            mean += params[0]
            var = params[1]**2/12
            
    elif distr == 8:
        subtipo = "lognormal"
        parametros = "Modelo con s = params[0] donde s es el parámetro de la distribución lognormal. El valor de s es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución lognormal y escala = escalado de la distribución "
        if len(params)==1:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[1])
        elif len(params) == 3:
            parametros += " en este caso el desplazamiento es de " + str(params[1]) +" y la escala de "+str(params[2])
        mean, var= lognorm.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = lognorm.mean(params[0], loc=params[1])
            var = lognorm.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = lognorm.mean(params[0], loc=params[1],scale=params[2])
            var = lognorm.var(params[0], loc=params[1], scale=params[2])
            
    elif distr == 9: 
        subtipo = "exponencial"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución exponencial y escala = escalado de la distribución. "
        if len(params)==0:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= expon.stats( moments='mv')
        if len (params) == 1 :
            mean = expon.mean(loc=params[0])
        elif len (params) == 2:
            mean = expon.mean(loc=params[0],scale=params[1])
            var = expon.var(scale=params[1])
            
    elif distr == 10: 
        subtipo = "gamma"
        parametros = "Modelo con a = params[0] donde a es el parámetro de la distribución gamma. El valor de a es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución gamma y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= gamma.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = gamma.mean(params[0], loc=params[1])
            var = gamma.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = gamma.mean(params[0], loc=params[1],scale=params[2])
            var = gamma.var(params[0], scale=params[2])
            
    elif distr == 11: 
        subtipo = "beta"
        parametros = "Modelo con a = params[0] y b = params[1] donde a y b son los parámetros de la distribución beta. El valor de a es "+ str(params[0])+ "y el de b es "+ str(params[1])+ ". Además, posee los parametros opcionales: despl = params[] y escala = params[1], donde despl= desplazamiento de la distribución beta y escala = escalado de la distribución. "
        if len(params)==2:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[2])
        elif len(params) == 4:
            parametros += "En este caso el desplazamiento es de " + str(params[2]) +" y la escala de "+str(params[3])
        mean, var= beta.stats(params[0],params[1], moments='mv')
        if len (params) == 3:
            mean = beta.mean(params[0],params[1], loc=params[2])
            var = beta.var(params[0],params[1], loc = params[2])
        elif len (params) == 4:
            mean = beta.mean(params[0],params[1], loc=params[2],scale=params[3])
            var = beta.var(params[0],params[1], loc=params[2], scale=params[3])
            
    elif distr == 12: 
        subtipo = "chi cuadrado"
        parametros = "Modelo con df = params[0] donde df es el parámetro de la distribución chi cuadrado. El valor de df es "+ str(params[0]) +". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución chi2 y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= chi2.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = chi2.mean(params[0], loc=params[1])
            var = chi2.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = chi2.mean(params[0], loc=params[1],scale=params[2])
            var = chi2.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 13: 
        subtipo = "t-student"
        parametros = "Modelo con v = params[0] donde v es el parámetro de la distribución t-student. El valor de t es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= t.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = t.mean(params[0], loc=params[1])
            var = t.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = t.mean(params[0], loc=params[1],scale=params[2])
            var = t.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 14: 
        subtipo = "pareto"
        parametros = "Modelo con b = params[0] donde b es el parámetro de la distribución pareto. El valor de b es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= pareto.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = pareto.mean(params[0], loc=params[1])
            var = pareto.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = pareto.mean(params[0], loc=params[1],scale=params[2])
            var = pareto.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 15:
        subtipo = "linealmente decreciente"
        parametros = "Modelo de tipo: y_i = y_i-1 - b, y_0 = a donde a = params[0] y b = params[1]"
        mean = "Información no relevante"
        var = "Información no relevante"
            
    elif distr == 16:
        subtipo = "linealmente creciente"
        parametros = "Modelo de tipo: y_i = y_i-1 + b, y_0 = a donde a = params[0] y b = params[1]"
        mean = "Información no relevante"
        var = "Información no relevante"
    
    elif distr == 17:
        subtipo = "random"
        parametros = "Modelo con una distribución con valores aleatorios entre params[0] = " + str(params[0]) +" y params [1] =" + str(params[1])
        mean = "Información no relevante"
        var = "Información no relevante"

    tipos = "Modelo con una distribución " + subtipo
    explicacion = "Inicio: fecha de inicio --> " + str(inicio)
    explicacion = explicacion +". Periodos: número de de periodos a generar --> "+ str(periodos)
    explicacion = explicacion + ". Freq: frequencia de la serie temporal --> " + str(freq)
    explicacion = explicacion + ". Distr: normal (1), binomial(2), poisson(3), geométrica(4), hipergeométrica(5), constante(6), uniforme(7), lognormal(8), exponencial(9), gamma(10), beta (11), chi cuadrado (12), t-student(13), pareto (14), linealmente decreciente(15), linealmente creciente (16) y random (17) --> " + str(distr)
    explicacion = explicacion + ". Columna: nombre de la columnas --> " + columna[0]
    for k in range (1, len (columna)):
        explicacion = explicacion+", " + columna [k]
    explicacion = explicacion + ". Params: parámetros de la distribución --> "+str(params [0])
    for k in range (1, len (params)):
        explicacion = explicacion+", " + str(params [k])
    return {"Tipo": tipos,"Parametros de la distribución": parametros, "Parámetros de la query" : explicacion, "Media" :mean, "Varianza" : var}


@app.get("/Datos/distribucion/periodos")
async def obtener_serie(inicio: str, periodos:int, freq:str, distr:int,  columna: List[str]= Query(...,description="Nombres de las columnas"),params: Optional[List[float]]= Query([],description="Parametros de la distribución")):
    
    df = crear_df_periodos_datos(inicio,periodos,freq,columna,distr,params)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-distribucion-periodos.csv"
    
    return response

@app.get("/Plot/distribucion/periodos")
async def obtener_grafica(inicio: str, periodos:int, freq:str, distr:int,  columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):
    df = crear_df_periodos_datos(inicio,periodos,freq,columna,distr,params)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

#MODELOS PERIÓDICOS

def datos_periodicos_amplitud(distr,params,num_datos,periodo):
    
    num_periodos = int(num_datos/periodo)
    res = num_datos % periodo
    datos_base0 = crear_datos(distr,params,periodo)
    datos_base = datos_base0
    for i in range(0,num_periodos-1):
        datos_base=np.concatenate((datos_base0,datos_base))
    if res>0:
        datos_base=np.concatenate((datos_base,datos_base0[:res]))
    return datos_base

def datos_periodicos_cantidad(distr,params,num_datos,num_periodos):
    periodo = int(num_datos/num_periodos)
    res = num_datos % num_periodos
    datos_base0 = crear_datos(distr,params,periodo)
    datos_base = datos_base0
    for i in range(0,num_periodos-1):
        datos_base=np.concatenate((datos_base0,datos_base))
    while res>0:
        datos_base=np.concatenate((datos_base,datos_base0[:res]))
        res = res - periodo
    return datos_base

def crear_df_fin_periodicos(inicio,fin,freq,columna,distr,params,p,tipo):
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
    indice = series_periodos(inicio,periodos,freq)
    num_datos = indice.size
    if tipo==1:
        datos = datos_periodicos_amplitud(distr,params,num_datos,p)
    elif tipo==2:
        datos=datos_periodicos_cantidad(distr,params,num_datos,p)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

@app.get("/Report/periodicos/fin")
async def obtener_serie(inicio: str, fin:str, freq:str, distr:int, p: int, tipo:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):

    if tipo==1:
        periodicidad = "periodos de amplitud " + str(p)
    elif tipo==2 :
        periodicidad = str(p)+ " periodos"
    if distr == 1 :
        subtipo = "normal"
        parametros ="Modelo con media = params[0] y desviación típica = params[1]. La media es " + str(params[0])+ " y la desviación típica es " + str(params[1])
        mean = params[0]
        var = params[1] **2
    elif distr ==2 :
        subtipo = "binomial"
        parametros = "Modelo con n = params[0] y p = params[1] donde n = número de pruebas y p = probabilidad de éxito. El valor de n es " + str(params[0])+" y el valor de p es "+str(params[1])+ ". Adicionalmente, se puede añadir un desplazamiento params[2] : "
        if len(params)==2:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 3:
            parametros += "en este caso es desplazamiento es de " + str(params[2])
        mean, var = binom.stats(params[0], params[1], moments='mv')
        if len (params) == 3 :
           mean += params[2]
    elif distr== 3 :
        subtipo = "poisson"
        parametros = "Modelo con mu = params[0] donde mu = parámetro de poisson . El valor de mu es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var= poisson.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
           
    elif distr == 4 :
        subtipo = "geométrica"
        parametros = "Modelo con p = params[0] donde p = probabilidad de éxito. El valor de p es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var = geom.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
            
    elif distr == 5:
        subtipo = "hipergeométrica"
        parametros = "Modelo con M = params[0], n = params[1] y N = params[2], donde M = tamaño población, n = exitosos en la población y N = tamaño muesta. El valor de M es " + str(params[0])+", el valor de n es " + str(params[1])+" y el valor de N es " + str(params[2])+". Adicionalmente, se puede añadir un desplazamiento params[3] : "
        if len(params)==3:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 4:
            parametros += "en este caso es desplazamiento es de " + str(params[3])
        mean, var= hypergeom.stats(params[0], params[1],params[2], moments='mv')
        if len (params) == 4 :
           mean += params[3]
            
    elif distr == 6: 
        subtipo ="constante"
        parametros = "Modelo con constante = " + str(params[0])
        mean = params[0]
        var = 0
        
    elif distr == 7:
        subtipo = "uniforme"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución uniforme y obteniendo una distribucion uniforme [despl,despl+escala],"
        if len(params)==0:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += " en este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= uniform.stats( moments='mv')
        if len (params) == 1 :
           mean = mean + params[0]
        elif len (params) == 2:
            mean = mean* params[1]
            mean += params[0]
            var = params[1]**2/12
            
    elif distr == 8:
        subtipo = "lognormal"
        parametros = "Modelo con s = params[0] donde s es el parámetro de la distribución lognormal. El valor de s es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución lognormal y escala = escalado de la distribución "
        if len(params)==1:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[1])
        elif len(params) == 3:
            parametros += " en este caso el desplazamiento es de " + str(params[1]) +" y la escala de "+str(params[2])
        mean, var= lognorm.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = lognorm.mean(params[0], loc=params[1])
            var = lognorm.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = lognorm.mean(params[0], loc=params[1],scale=params[2])
            var = lognorm.var(params[0], loc=params[1], scale=params[2])
            
    elif distr == 9: 
        subtipo = "exponencial"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución exponencial y escala = escalado de la distribución. "
        if len(params)==0:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= expon.stats( moments='mv')
        if len (params) == 1 :
            mean = expon.mean(loc=params[0])
        elif len (params) == 2:
            mean = expon.mean(loc=params[0],scale=params[1])
            var = expon.var(scale=params[1])
            
    elif distr == 10: 
        subtipo = "gamma"
        parametros = "Modelo con a = params[0] donde a es el parámetro de la distribución gamma. El valor de a es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución gamma y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= gamma.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = gamma.mean(params[0], loc=params[1])
            var = gamma.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = gamma.mean(params[0], loc=params[1],scale=params[2])
            var = gamma.var(params[0], scale=params[2])
            
    elif distr == 11: 
        subtipo = "beta"
        parametros = "Modelo con a = params[0] y b = params[1] donde a y b son los parámetros de la distribución beta. El valor de a es "+ str(params[0])+ "y el de b es "+ str(params[1])+ ". Además, posee los parametros opcionales: despl = params[] y escala = params[1], donde despl= desplazamiento de la distribución beta y escala = escalado de la distribución. "
        if len(params)==2:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[2])
        elif len(params) == 4:
            parametros += "En este caso el desplazamiento es de " + str(params[2]) +" y la escala de "+str(params[3])
        mean, var= beta.stats(params[0],params[1], moments='mv')
        if len (params) == 3:
            mean = beta.mean(params[0],params[1], loc=params[2])
            var = beta.var(params[0],params[1], loc = params[2])
        elif len (params) == 4:
            mean = beta.mean(params[0],params[1], loc=params[2],scale=params[3])
            var = beta.var(params[0],params[1], loc=params[2], scale=params[3])
            
    elif distr == 12: 
        subtipo = "chi cuadrado"
        parametros = "Modelo con df = params[0] donde df es el parámetro de la distribución chi cuadrado. El valor de df es "+ str(params[0]) +". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución chi2 y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= chi2.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = chi2.mean(params[0], loc=params[1])
            var = chi2.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = chi2.mean(params[0], loc=params[1],scale=params[2])
            var = chi2.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 13: 
        subtipo = "t-student"
        parametros = "Modelo con v = params[0] donde v es el parámetro de la distribución t-student. El valor de t es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= t.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = t.mean(params[0], loc=params[1])
            var = t.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = t.mean(params[0], loc=params[1],scale=params[2])
            var = t.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 14: 
        subtipo = "pareto"
        parametros = "Modelo con b = params[0] donde b es el parámetro de la distribución pareto. El valor de b es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= pareto.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = pareto.mean(params[0], loc=params[1])
            var = pareto.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = pareto.mean(params[0], loc=params[1],scale=params[2])
            var = pareto.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 15:
        subtipo = "linealmente decreciente"
        parametros = "Modelo de tipo: y_i = y_i-1 - b, y_0 = a donde a = params[0] y b = params[1]"
        mean = "Información no relevante"
        var = "Información no relevante"
            
    elif distr == 16:
        subtipo = "linealmente creciente"
        parametros = "Modelo de tipo: y_i = y_i-1 + b, y_0 = a donde a = params[0] y b = params[1]"
        mean = "Información no relevante"
        var = "Información no relevante"
    
    elif distr == 17:
        subtipo = "random"
        parametros = "Modelo con una distribución con valores aleatorios entre params[0] = " + str(params[0]) +" y params [1] =" + str(params[1])
        mean = "Información no relevante"
        var = "Información no relevante"

    tipos = "Modelo periodico siguiendo una distribución " + subtipo + " con " + periodicidad
    explicacion = "Inicio: fecha de inicio --> " + str(inicio)
    explicacion = explicacion +". Fin: fecha de fin --> "+ str(fin)
    explicacion = explicacion + ". Freq: frequencia de la serie temporal --> " + str(freq)
    explicacion = explicacion + ". Distr: normal (1), binomial(2), poisson(3), geométrica(4), hipergeométrica(5), constante(6), uniforme(7), lognormal(8), exponencial(9), gamma(10), beta (11), chi cuadrado (12), t-student(13), pareto (14), linealmente decreciente(15), linealmente creciente (16) y random (17) --> " + str(distr)
    explicacion += ". p: indica la amplitud del periodo (tipo 1) o la cantidad de periodos (tipo 2) --> " + str(p)
    explicacion += ". Tipo: por amplitud (1) / por cantidad (2) --> "+ str(tipo)
    explicacion = explicacion + ". Columna: nombre de la columnas --> " + columna[0]
    for k in range (1, len (columna)):
        explicacion = explicacion+", " + columna [k]
    explicacion = explicacion + ". Params: parámetros de la distribución --> "+str(params [0])
    for k in range (1, len (params)):
        explicacion = explicacion+", " + str(params[k])
    return {"Tipo": tipos,"Parametros de la distribución": parametros, "Parámetros de la query" : explicacion, "Media" :mean, "Varianza" : var}


@app.get("/Datos/periodicos/fin")
async def obtener_serie(inicio: str, fin:str, freq:str, distr:int, p: int, tipo:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):

    df =crear_df_fin_periodicos(inicio,fin,freq,columna,distr,params,p,tipo)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-periodicos-fin.csv"
    
    return response

@app.get("/Plot/periodicos/fin")
async def obtener_grafica(inicio: str, fin:str, freq:str, distr:int, p: int, tipo:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):
    df = crear_df_fin_periodicos(inicio,fin,freq,columna,distr,params, p,tipo)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Report/periodicos/periodos")
async def obtener_serie(inicio: str, periodos:int, freq:str, distr:int, p: int, tipo:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):

    if tipo==1:
        periodicidad = "periodos de amplitud " + str(p)
    elif tipo==2 :
        periodicidad = str(p)+ " periodos"
        
    if distr == 1 :
        subtipo = "normal"
        parametros ="Modelo con media = params[0] y desviación típica = params[1]. La media es " + str(params[0])+ " y la desviación típica es " + str(params[1])
        mean = params[0]
        var = params[1] **2
    elif distr ==2 :
        subtipo = "binomial"
        parametros = "Modelo con n = params[0] y p = params[1] donde n = número de pruebas y p = probabilidad de éxito. El valor de n es " + str(params[0])+" y el valor de p es "+str(params[1])+ ". Adicionalmente, se puede añadir un desplazamiento params[2] : "
        if len(params)==2:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 3:
            parametros += "en este caso es desplazamiento es de " + str(params[2])
        mean, var = binom.stats(params[0], params[1], moments='mv')
        if len (params) == 3 :
           mean += params[2]
    elif distr== 3 :
        subtipo = "poisson"
        parametros = "Modelo con mu = params[0] donde mu = parámetro de poisson . El valor de mu es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var= poisson.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
           
    elif distr == 4 :
        subtipo = "geométrica"
        parametros = "Modelo con p = params[0] donde p = probabilidad de éxito. El valor de p es " + str(params[0])+". Adicionalmente, se puede añadir un desplazamiento params[1] : "
        if len(params)==1:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 2:
            parametros += "en este caso es desplazamiento es de " + str(params[1])
        mean, var = geom.stats(params[0], moments='mv')
        if len (params) == 2 :
           mean += params[1]
            
    elif distr == 5:
        subtipo = "hipergeométrica"
        parametros = "Modelo con M = params[0], n = params[1] y N = params[2], donde M = tamaño población, n = exitosos en la población y N = tamaño muesta. El valor de M es " + str(params[0])+", el valor de n es " + str(params[1])+" y el valor de N es " + str(params[2])+". Adicionalmente, se puede añadir un desplazamiento params[3] : "
        if len(params)==3:
            parametros += "en este caso no hay desplazamiento"
        elif len(params) == 4:
            parametros += "en este caso es desplazamiento es de " + str(params[3])
        mean, var= hypergeom.stats(params[0], params[1],params[2], moments='mv')
        if len (params) == 4 :
           mean += params[3]
            
    elif distr == 6: 
        subtipo ="constante"
        parametros = "Modelo con constante = " + str(params[0])
        mean = params[0]
        var = 0
        
    elif distr == 7:
        subtipo = "uniforme"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución uniforme y obteniendo una distribucion uniforme [despl,despl+escala],"
        if len(params)==0:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += " en este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= uniform.stats( moments='mv')
        if len (params) == 1 :
           mean = mean + params[0]
        elif len (params) == 2:
            mean = mean* params[1]
            mean += params[0]
            var = params[1]**2/12
            
    elif distr == 8:
        subtipo = "lognormal"
        parametros = "Modelo con s = params[0] donde s es el parámetro de la distribución lognormal. El valor de s es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución lognormal y escala = escalado de la distribución "
        if len(params)==1:
            parametros += " en este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += " en este caso el desplazamiento es de " + str(params[1])
        elif len(params) == 3:
            parametros += " en este caso el desplazamiento es de " + str(params[1]) +" y la escala de "+str(params[2])
        mean, var= lognorm.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = lognorm.mean(params[0], loc=params[1])
            var = lognorm.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = lognorm.mean(params[0], loc=params[1],scale=params[2])
            var = lognorm.var(params[0], loc=params[1], scale=params[2])
            
    elif distr == 9: 
        subtipo = "exponencial"
        parametros = "Modelo con parametros opcionales: despl = params[0] y escala = params[1], donde despl= desplazamiento de la distribución exponencial y escala = escalado de la distribución. "
        if len(params)==0:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 1:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= expon.stats( moments='mv')
        if len (params) == 1 :
            mean = expon.mean(loc=params[0])
        elif len (params) == 2:
            mean = expon.mean(loc=params[0],scale=params[1])
            var = expon.var(scale=params[1])
            
    elif distr == 10: 
        subtipo = "gamma"
        parametros = "Modelo con a = params[0] donde a es el parámetro de la distribución gamma. El valor de a es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución gamma y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= gamma.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = gamma.mean(params[0], loc=params[1])
            var = gamma.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = gamma.mean(params[0], loc=params[1],scale=params[2])
            var = gamma.var(params[0], scale=params[2])
            
    elif distr == 11: 
        subtipo = "beta"
        parametros = "Modelo con a = params[0] y b = params[1] donde a y b son los parámetros de la distribución beta. El valor de a es "+ str(params[0])+ "y el de b es "+ str(params[1])+ ". Además, posee los parametros opcionales: despl = params[] y escala = params[1], donde despl= desplazamiento de la distribución beta y escala = escalado de la distribución. "
        if len(params)==2:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[2])
        elif len(params) == 4:
            parametros += "En este caso el desplazamiento es de " + str(params[2]) +" y la escala de "+str(params[3])
        mean, var= beta.stats(params[0],params[1], moments='mv')
        if len (params) == 3:
            mean = beta.mean(params[0],params[1], loc=params[2])
            var = beta.var(params[0],params[1], loc = params[2])
        elif len (params) == 4:
            mean = beta.mean(params[0],params[1], loc=params[2],scale=params[3])
            var = beta.var(params[0],params[1], loc=params[2], scale=params[3])
            
    elif distr == 12: 
        subtipo = "chi cuadrado"
        parametros = "Modelo con df = params[0] donde df es el parámetro de la distribución chi cuadrado. El valor de df es "+ str(params[0]) +". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución chi2 y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= chi2.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = chi2.mean(params[0], loc=params[1])
            var = chi2.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = chi2.mean(params[0], loc=params[1],scale=params[2])
            var = chi2.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 13: 
        subtipo = "t-student"
        parametros = "Modelo con v = params[0] donde v es el parámetro de la distribución t-student. El valor de t es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= t.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = t.mean(params[0], loc=params[1])
            var = t.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = t.mean(params[0], loc=params[1],scale=params[2])
            var = t.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 14: 
        subtipo = "pareto"
        parametros = "Modelo con b = params[0] donde b es el parámetro de la distribución pareto. El valor de b es "+ str(params[0])+ ". Además, posee los parametros opcionales: despl = params[1] y escala = params[2], donde despl= desplazamiento de la distribución t-student y escala = escalado de la distribución. "
        if len(params)==1:
            parametros += "En este caso no hay desplazamiento ni escala "
        elif len(params) == 2:
            parametros += "En este caso el desplazamiento es de " + str(params[0])
        elif len(params) == 3:
            parametros += "En este caso el desplazamiento es de " + str(params[0]) +" y la escala de "+str(params[1])
        mean, var= pareto.stats(params[0], moments='mv')
        if len (params) == 2:
            mean = pareto.mean(params[0], loc=params[1])
            var = pareto.var(params[0], loc = params[1])
        elif len (params) == 3:
            mean = pareto.mean(params[0], loc=params[1],scale=params[2])
            var = pareto.var(params[0], loc=params[1],scale=params[2])
            
    elif distr == 15:
        subtipo = "linealmente decreciente"
        parametros = "Modelo de tipo: y_i = y_i-1 - b, y_0 = a donde a = params[0] y b = params[1]"
        mean = "Información no relevante"
        var = "Información no relevante"
            
    elif distr == 16:
        subtipo = "linealmente creciente"
        parametros = "Modelo de tipo: y_i = y_i-1 + b, y_0 = a donde a = params[0] y b = params[1]"
        mean = "Información no relevante"
        var = "Información no relevante"
    
    elif distr == 17:
        subtipo = "random"
        parametros = "Modelo con una distribución con valores aleatorios entre params[0] = " + str(params[0]) +" y params [1] =" + str(params[1])
        mean = "Información no relevante"
        var = "Información no relevante"

    tipos = "Modelo periodico siguiendo una distribución " + subtipo + " con " + periodicidad
    explicacion = "Inicio: fecha de inicio --> " + str(inicio)
    explicacion = explicacion +". Periodos: número de datos a generar --> "+ str(periodos)
    explicacion = explicacion + ". Freq: frequencia de la serie temporal --> " + str(freq)
    explicacion = explicacion + ". Distr: normal (1), binomial(2), poisson(3), geométrica(4), hipergeométrica(5), constante(6), uniforme(7), lognormal(8), exponencial(9), gamma(10), beta (11), chi cuadrado (12), t-student(13), pareto (14), linealmente decreciente(15), linealmente creciente (16) y random (17) --> " + str(distr)
    explicacion += ". p: indica la amplitud del periodo (tipo 1) o la cantidad de periodos (tipo 2) --> " + str(p)
    explicacion += ". Tipo: por amplitud (1) / por cantidad (2) --> "+ str(tipo)
    explicacion = explicacion + ". Columna: nombre de la columnas --> " + columna[0]
    for k in range (1, len (columna)):
        explicacion = explicacion+", " + columna [k]
    explicacion = explicacion + ". Params: parámetros de la distribución --> "+str(params [0])
    for k in range (1, len (params)):
        explicacion = explicacion+", " + str(params[k])
    return {"Tipo": tipos,"Parametros de la distribución": parametros, "Parámetros de la query" : explicacion, "Media" :mean, "Varianza" : var}

@app.get("/Datos/periodicos/periodos")
async def obtener_serie(inicio: str, periodos:int, freq:str, distr:int, p:int, tipo:int,  columna: List[str]= Query(...,description="Nombres de las columnas"), params: Optional[List[float]]= Query([],description="Parametros de la distribución")):

    df= crear_df_periodos_periodicos(inicio,periodos,freq,columna,distr,params,p,tipo)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-periodicos-periodos.csv"
    
    return response
 
@app.get("/Plot/periodicos/periodos")
async def obtener_grafica(inicio: str, periodos:int, freq:str, distr:int, p:int, tipo:int,  columna: List[str]= Query(...,description="Nombres de las columnas"),params: Optional[List[float]]= Query([],description="Parametros de la distribución")):
    df = crear_df_periodos_periodicos(inicio,periodos,freq,columna,distr,params,p,tipo)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

# Modelos ARMA

def modelo_AR(c,phi,num_datos,desv,a=[]):
    
    orden = len(phi)
    
    if len(a)==0: 
        a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    
    for k in range(0,orden):
        datos[k] = c + a[k]
            
    for i in range(orden,num_datos):
        datos [i]= c + a[i]
        for j in range (1,orden+1):
            datos[i] = datos[i] + phi[j-1]*datos[i-j]
    
    return datos

def modelo_MA(c,teta,num_datos,desv,a=[]):
    
    orden = len(teta)
    
    if len(a)==0:  
        a = crear_datos(1,[0,desv],num_datos)
    
    datos = np.zeros(num_datos)
    for i in range(0,orden):
        datos[i]= c + a[i]
            
    for i in range(orden,num_datos):
        datos[i] = c + a[i]
        for j in range (1,orden+1):
            datos[i]= datos[i] + teta[j-1]*a[i-j]
            
    return datos

def modelo_ARMA(c,phi,teta,num_datos,desv,a=[]):
    
    if len(a)==0:  
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
    

def modelo_AR_estacional(c,phi,s,num_datos,desv,a=[]):
    
    if len(a)==0: 
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

def modelo_MA_estacional(c,teta,s,num_datos,desv,a=[]):
    
    if len(a)==0: 
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

def modelo_ARMA_estacional(c,phi,teta,s,num_datos,desv,a=[]):
    
    if len(a)==0:  
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

def creacion_modelos_ARMA(c,num_datos,desv,s=0,phi=[],teta=[],a=[]):
    
    p = len(phi)
    q = len(teta)
    
    if s == 0:
        
        if q == 0:
            datos = modelo_AR(c,phi,num_datos,desv,a)    
        elif p == 0: 
            datos = modelo_MA(c,teta,num_datos,desv,a)
        else:
            datos = modelo_ARMA(c,phi,teta,num_datos,desv,a)
            
    else :
        
        if q == 0:
            datos = modelo_AR_estacional(c,phi,s,num_datos,desv,a)   
        elif p == 0: 
            datos = modelo_MA_estacional(c,teta,s,num_datos,desv,a)
        else:
            datos = modelo_ARMA_estacional(c,phi,teta,s,num_datos,desv,a)
            
    return datos 

def crear_df_fin_ARMA(inicio,fin,freq,columna,c,desv,s=0,phi=[],teta=[],a=[]):
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos=creacion_modelos_ARMA(c,num_datos,desv,s,phi,teta,a)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_ARMA(inicio,periodos,freq,columna,c,desv,s=0,phi=[],teta=[],a=[]):
    indice = series_periodos(inicio,periodos,freq)
    datos=creacion_modelos_ARMA(c,periodos,desv,s,phi,teta,a)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 


@app.get("/Report/ARMA/fin")
async def obtener_report(inicio: str, fin:str, freq:str,c:float, desv:float, s : Union[int, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
    
    if phi == []:
        subtipo = "de medias móviles"
        parametros= "La serie sigue una distribución de medias móviles con constante c = "+ str(c)+" y con valores de teta: teta_0 " + str(teta[0])
        for k in range (1,len(teta)):
           parametros = parametros  + " teta_"+ str(k)+" = " + str (teta[k])
           
    elif teta ==[]:
        subtipo = "autorregresivo"
        parametros= "La serie sigue una distribución autorregresiva con constante c = "+ str(c)+" y con valores de phi: phi_0 " + str(phi[0])
        for k in range (1,len(phi)):
           parametros = parametros  + " phi_"+ str(k)+" = " + str (phi[k])
    else: 
        subtipo ="autorregresivo y de medias móviles"
        parametros = "La serie sigue una distribución autorregresiva y de medias móviles con constante c = "+ str(c)+" y con valores de phi: phi_0 " + str(phi[0])
        for k in range (1,len(phi)):
           parametros = parametros  + " phi_"+ str(k)+" = " + str (phi[k])
        parametros = parametros + " y con valores de teta: teta_0 " + str(teta[0])
        for k in range (1,len(teta)):
           parametros = parametros  + " teta_"+ str(k)+" = " + str (teta[k])
    if s != 0:
        subtipo += " estacional con amplitud de la estación: " + str(s)
    tipos = "Modelo " + subtipo
    explicacion = "Inicio: fecha de inicio " + str(inicio)
    explicacion = explicacion +". Fin: fecha de fin --> "+ str(fin)
    explicacion = explicacion + ". Freq: frequencia de la serie temporal --> " + str(freq)
    explicacion = explicacion + ". c: constante del modelo --> " + str(c)
    explicacion = explicacion + ". s: amplitud de la estación --> " + str(s)
    explicacion = explicacion + ". Desv: desviación típica del ruido blaco --> " + str(desv)
    explicacion = explicacion + ". Columna: nombre de la columnas --> " + columna[0]
    for k in range (1, len (columna)):
        explicacion = explicacion+", " + columna [k]
    explicacion = explicacion + ". Params: parámetros del modelo, phi y teta --> "
    if phi != []:
        explicacion+= "phi : " + str(phi[0])
        for k in range (1, len (phi)):
            explicacion = explicacion+", " + str(phi[k])
    if teta!=[]:
        explicacion+= "teta : " + str(teta[0])
        for k in range (1, len (teta)):
            explicacion = explicacion+", " + str(teta[k])
    return {"Tipo": tipos, "Parámetro del modelo" : parametros, "Parámetros de la query" : explicacion }


@app.get("/Datos/ARMA/fin")
async def obtener_serie(inicio: str, fin:str, freq:str,c:float, desv:float, s : Union[int, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
      
    df = crear_df_fin_ARMA(inicio,fin,freq,columna,c,desv,s,phi,teta)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-ARMA-fin.csv"
    
    return response

@app.get("/Plot/ARMA/fin")
async def obtener_grafica(inicio: str, fin:str, freq:str,c:float, desv:float, s : Union[int, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
    df = crear_df_fin_ARMA(inicio,fin,freq,columna,c,desv,s,phi,teta)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Report/ARMA/periodos")
async def obtener_report(inicio: str, periodos:int, freq:str,c:float, desv:float, s : Union[int, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
    
    if phi == []:
        subtipo = "de medias móviles"
        parametros= "La serie sigue una distribución de medias móviles con constante c "+ str(c)+" y con valores de teta: teta_0 " + str(teta[0])
        for k in range (1,len(teta)):
           parametros = parametros  + " teta_"+ str(k)+" = " + str (teta[k])
           
    elif teta ==[]:
        subtipo = "autorregresivo"
        parametros= "La serie sigue una distribución autorregresiva con constante c "+ str(c)+" valores de phi: phi_0 " + str(phi[0])
        for k in range (1,len(phi)):
           parametros = parametros  + " phi_"+ str(k)+" = " + str (phi[k])
    else: 
        subtipo ="autorregresivo y de medias móviles"
        parametros = "La serie sigue una distribución autorregresiva y de medias móviles con constante c "+ str(c)+" y con valores de phi: phi_0 " + str(phi[0])
        for k in range (1,len(phi)):
           parametros = parametros  + " phi_"+ str(k)+" = " + str (phi[k])
        parametros = parametros + " y con valores de teta: teta_0 " + str(teta[0])
        for k in range (1,len(teta)):
           parametros = parametros  + " teta_"+ str(k)+" = " + str (teta[k])
    if s != 0:
        subtipo += " estacional con amplitud de la estación: " + str(s)
    tipos = "Modelo " + subtipo
    explicacion = "Inicio: fecha de inicio " + str(inicio)
    explicacion = explicacion +". Periodos: número de datos a generar  --> "+ str(periodos)
    explicacion = explicacion + ". Freq: frequencia de la serie temporal --> " + str(freq)
    explicacion = explicacion + ". c: constante del modelo --> " + str(c)
    explicacion = explicacion + ". s: amplitud de la estación --> " + str(s)
    explicacion = explicacion + ". Desv: desviación típica del ruido blaco --> " + str(desv)
    explicacion = explicacion + ". Columna: nombre de la columnas --> " + columna[0]
    for k in range (1, len (columna)):
        explicacion = explicacion+", " + columna [k]
    explicacion = explicacion + ". Params: parámetros del modelo, phi y teta --> "
    if phi != []:
        explicacion+= "phi : " + str(phi[0])
        for k in range (1, len (phi)):
            explicacion = explicacion+", " + str(phi[k])
    if teta!=[]:
        explicacion+= "teta : " + str(teta[0])
        for k in range (1, len (teta)):
            explicacion = explicacion+", " + str(teta[k])
    return {"Tipo": tipos, "Parámetro del modelo" : parametros, "Parámetros de la query" : explicacion }

@app.get("/Datos/ARMA/periodos")
async def obtener_serie(inicio: str, periodos:int, freq:str,c:float, desv:float, s : Union[int, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
    df = crear_df_periodos_ARMA(inicio,periodos,freq,columna,c,desv,s,phi,teta)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-ARMA-periodos.csv"
    
    return response
 
@app.get("/Plot/ARMA/periodos") 
async def obtener_grafica(inicio: str, periodos:int, freq:str,c:float, desv:float, s : Union[int, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):

    df = crear_df_periodos_ARMA(inicio,periodos,freq,columna,c,desv,s,phi,teta)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

def crear_drift(params1,params2,tipo,num_drift,num_datos):
    
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
        a = params2[5]
        datos2 = creacion_modelos_ARMA(c,num_datos-num_drift,desv,s,phi,teta,a)
    
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
        a = params1[5]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta,a)
        
        c2 = params2[0]
        desv2 = params2[1]
        s2 = params2[2]
        phi2 = params2[3]
        teta2 = params2[4]
        a2 = params2[5]
        datos2 = creacion_modelos_ARMA(c2,num_datos-num_drift,desv2,s2,phi2,teta2,a2)
        
    elif tipo==6: 
        c = params1[0]
        desv = params1[1]
        s = params1[2]
        phi = params1[3]
        teta = params1[4]
        a = params1[5]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta,a)
        
        distr2=params2[0]
        parametros2=params2[1]
        datos2 = crear_datos(distr2,parametros2,num_datos - num_drift)
  
        
    elif tipo==7: 
        c = params1[0]
        desv = params1[1]
        s = params1[2]
        phi = params1[3]
        teta = params1[4]
        a = params1[5]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta,a)

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
        a = params1[5]
        datos1 = creacion_modelos_ARMA(c,num_drift,desv,s,phi,teta,a)
    
        param,tipo2,coef_error = params2[0],params2[1],params2[2]
        datos2 = tendencia_det(param,tipo2,num_datos-num_drift,coef_error)  
         
    elif tipo==9: 
        tipo1 = params1[0]
        distr,params,p = params1[1],params1[2],params1[3]
        if tipo1==1:
            datos1=datos_periodicos_amplitud(distr,params,num_drift,p)
        elif tipo1==2:
            datos1=datos_periodicos_cantidad(distr,params,num_drift,p)
        print(datos1.shape)

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
        a2 = params2[5]
        datos2 = creacion_modelos_ARMA(c2,num_datos-num_drift,desv2,s2,phi2,teta2,a2)  
          
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
        a2 = params2[5]
        datos2 = creacion_modelos_ARMA(c2,num_datos-num_drift,desv2,s2,phi2,teta2,a2)
        
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

def crear_df_fin_DRIFT(inicio,fin,freq,columna,params1,params2,tipo,num_drift):
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos = crear_drift(params1,params2,tipo,num_drift,num_datos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_DRIFT(inicio,periodos,freq,columna,params1,params2,tipo,num_drift):
    indice = series_periodos(inicio,periodos,freq)
    datos = crear_drift(params1,params2,tipo,num_drift,periodos)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df  

@app.get("/Datos/drift/fin/dist-dist")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, dist2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[dist2,params2],1,num_drift)
     # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-dist-fin.csv"
    
    return response

@app.get("/Plot/drift/fin/dist-dist")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, dist2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[dist2,params2],1,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/periodos/dist-dist")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int,dist2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[dist2,params2],1,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-dist-periodos.csv"
    
    return response

@app.get("/Plot/drift/periodos/dist-dist")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int,dist2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna, [dist1,params1],[dist2,params2],1,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/fin/dist-ARMA")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, c:float, desv:float, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[c,desv,s,phi,teta,[]],2,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-ARMA-fin.csv"
    
    return response

@app.get("/Plot/drift/fin/dist-ARMA")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, c:float, desv:float, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[c,desv,s,phi,teta,[]],2,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/dist-ARMA")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, c:float ,desv:float ,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[c,desv,s,phi,teta,[]],2,num_drift)

    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-ARMA-periodos.csv"
    
    return response  

@app.get("/Plot/drift/periodos/dist-ARMA")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, c:float ,desv:float ,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[c,desv,s,phi,teta,[]],2,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/fin/dist-periodico")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, tipo2:int, dist2:int, p2:int,columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[tipo2,dist2,params2,p2],3,num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-periodico-fin.csv"
    
    return response  
    

@app.get("/Plot/drift/fin/dist-periodico")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, tipo2:int, dist2:int, p2:int,columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna, [dist1,params1],[tipo2,dist2,params2,p2],3,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/periodos/dist-periodico")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, tipo2:int, dist2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[tipo2,dist2,params2,p2],3,num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-periodico-periodos.csv"
    
    return response  

@app.get("/Plot/drift/periodos/dist-periodico")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, tipo2:int, dist2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[tipo2,dist2,params2,p2],3,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

    
@app.get("/Datos/drift/fin/dist-tendencia")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, tipo2:int, coef_error: Union[int,None] = 0 ,columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la tendencia")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[params2,tipo2,coef_error],4,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-tendencia-fin.csv"
    
    return response  

@app.get("/Plot/drift/fin/dist-tendencia")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, tipo2:int, coef_error: Union[int,None] = 0 ,columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la tendencia")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[params2,tipo2,coef_error],4,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/dist-tendencia")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, tipo2:int, coef_error : Union[int,None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la tendencia")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[params2,tipo2,coef_error],4,num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-dist-tendencia-periodos.csv"
    
    return response 

@app.get("/Plot/drift/periodos/dist-tendencia")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, tipo2:int, coef_error : Union[int,None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la tendencia")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[params2,tipo2,coef_error],4,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

    
@app.get("/Datos/drift/fin/ARMA-ARMA")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int ,c1:float , desv1:float, c2:float, desv2:float, s1: Union[int,None] = 0, s2: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi1: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta1:Optional[List[float]]= Query([],description="Parámetros medias móviles"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c1,desv1,s1,phi1,teta1,[]],[c2,desv2,s2,phi2,teta2,[]],5,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-ARMA-fin.csv"
    
    return response 


@app.get("/Plot/drift/fin/ARMA-ARMA")
async def obtener_gráfica(inicio: str, fin:str, freq:str, num_drift:int ,c1:float , desv1:float, c2:float, desv2:float, s1: Union[int,None] = 0, s2: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi1: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta1:Optional[List[float]]= Query([],description="Parámetros medias móviles"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c1,desv1,s1,phi1,teta1,[]],[c2,desv2,s2,phi2,teta2,[]],5,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/periodos/ARMA-ARMA")
def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int ,c1:float , desv1:float, c2:float , desv2:float, s1: Union[int,None] = 0, s2: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi1: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta1:Optional[List[float]]= Query([],description="Parámetros medias móviles"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c1,desv1,s1,phi1,teta1,[]],[c2,desv2,s2,phi2,teta2,[]],5,num_drift)
    
     # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-ARMA-periodos.csv"
    
    return response 

@app.get("/Plot/drift/periodos/ARMA-ARMA")
async def obtener_gráfica(inicio: str, periodos:int, freq:str, num_drift:int ,c1:float , desv1:float, c2:float , desv2:float, s1: Union[int,None] = 0, s2: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi1: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta1:Optional[List[float]]= Query([],description="Parámetros medias móviles"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c1,desv1,s1,phi1,teta1,[]],[c2,desv2,s2,phi2,teta2,[]],5,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/fin/ARMA-dist")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int ,c:float , desv:float, dist2:int,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[dist2,params2],6,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-dist-fin.csv"
    
    return response 

@app.get("/Plot/drift/fin/ARMA-dist")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int ,c:float , desv:float, dist2:int,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[dist2,params2],6,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/periodos/ARMA-dist")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int ,c:float, desv:float, dist2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[dist2,params2],6,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-dist-periodos.csv"
    
    return response 

@app.get("/Plot/drift/periodos/ARMA-dist")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int ,c:float, desv:float, dist2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[dist2,params2],6,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/fin/ARMA-periodicos")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, c:float, desv:float, tipo2:int, distr2:int, p2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[tipo2,distr2,params2,p2],7,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-periodicos-fin.csv"
    
    return response 

@app.get("/Plot/drift/fin/ARMA-periodicos")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, c:float, desv:float, tipo2:int, distr2:int, p2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[tipo2,distr2,params2,p2],7,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/ARMA-periodicos")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int ,c:float , desv:float, tipo2:int, distr2:int, p2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[tipo2,distr2,params2,p2],7,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-periodicos-periodos.csv"
    
    return response 

@app.get("/Plot/drift/periodos/ARMA-periodicos")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int ,c:float , desv:float, tipo2:int, distr2:int, p2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[tipo2,distr2,params2,p2],7,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/fin/ARMA-tendencia")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int ,c:float , desv:float, tipo2:int,coef_error: Union[float, None] = 0,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la tendencia determinista")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[params2,tipo2,coef_error],8,num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-tendencia-fin.csv"
    
    return response 

@app.get("/Plot/drift/fin/ARMA-tendencia")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int ,c:float , desv:float, tipo2:int,coef_error: Union[float, None] = 0,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la tendencia determinista")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[params2,tipo2,coef_error],8,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/ARMA-tendencia")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int ,c:float , desv:float, tipo2:int,coef_error: Union[float, None] = 0, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la tendencia determinista")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[params2,tipo2,coef_error],8,num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-ARMA-tendencia-periodos.csv"
    
    return response 

@app.get("/Plot/drift/periodos/ARMA-tendencia")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int ,c:float , desv:float, tipo2:int,coef_error: Union[float, None] = 0, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la tendencia determinista")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[params2,tipo2,coef_error],8,num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/fin/periodico-periodico")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int, distr2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[tipo2,distr2,params2,p2], 9, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-periodico-fin.csv"
    
    return response 

@app.get("/Plot/drift/fin/periodico-periodico")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int, distr2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[tipo2,distr2,params2,p2], 9, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/periodico-periodico")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int, distr2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[tipo2,distr2,params2,p2], 9, num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-periodico-periodos.csv"
    
    return response 

    
@app.get("/Plot/drift/periodos/periodico-periodico")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int, distr2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[tipo2,distr2,params2,p2], 9, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/fin/periodico-distr")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, distr2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[distr2,params2], 10, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-dist-fin.csv"
    
    return response 



@app.get("/Plot/drift/fin/periodico-distr")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, distr2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[distr2,params2], 10, num_drift)

    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/periodico-distr")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, distr2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[distr2,params2], 10, num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-dist-periodos.csv"
    
    return response 
    
@app.get("/Plot/drift/periodos/periodico-distr")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, distr2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[distr2,params2], 10, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/fin/periodico-ARMA")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, c2:float, desv2:float, s2 : Union[None,int] = 0,  columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"),  phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[c2,desv2,s2,phi2,teta2,[]], 11, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-ARMA-fin.csv"
    
    return response

@app.get("/Plot/drift/fin/periodico-ARMA")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, c2:float, desv2:float, s2 : Union[None,int] = 0,  columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"),  phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[c2,desv2,s2,phi2,teta2,[]], 11, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/periodico-ARMA")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, c2:float, desv2:float, s2 : Union[None,int] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"),  phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[c2,desv2,s2,phi2,teta2,[]], 11, num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-ARMA-periodos.csv"
    
    return response

@app.get("/Plot/drift/periodos/periodico-ARMA")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, c2:float, desv2:float, s2 : Union[None,int] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"),  phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[c2,desv2,s2,phi2,teta2,[]], 11, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")
    
@app.get("/Datos/drift/fin/periodico-tendencia")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int,coef_error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[params2,tipo2,coef_error], 12, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-tendencia-fin.csv"
    
    return response 

@app.get("/Plot/drift/fin/periodico-tendencia")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int,coef_error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):
    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[params2,tipo2,coef_error], 12, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/periodico-tendencia")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int,coef_error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[params2,tipo2,coef_error], 12, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-periodico-tendencia-periodos.csv"
    
    return response

@app.get("/Plot/drift/periodos/periodico-tendencia")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int,coef_error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[params2,tipo2,coef_error], 12, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

    
@app.get("/Datos/drift/fin/tendencia-tendencia")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,tipo2:int, coef_error1: Union[float, None] = 0,coef_error2: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[params2,tipo2,coef_error2], 13, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-tendencia-fin.csv"
    
    return response

@app.get("/Plot/drift/fin/tendencia-tendencia")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,tipo2:int, coef_error1: Union[float, None] = 0,coef_error2: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[params2,tipo2,coef_error2], 13, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/tendencia-tendencia")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, tipo2:int, coef_error1: Union[float, None] = 0, coef_error2: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[params2,tipo2,coef_error2], 13, num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-tendencia-periodos.csv"
    
    return response

@app.get("/Plot/drift/periodos/tendencia-tendencia")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, tipo2:int, coef_error1: Union[float, None] = 0, coef_error2: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):
    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[params2,tipo2,coef_error2], 13, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/fin/tendencia-distr")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,distr2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[distr2,params2], 14, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-dist-fin.csv"
    
    return response

@app.get("/Plot/drift/fin/tendencia-distr")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,distr2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[distr2,params2], 14, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/periodos/tendencia-distr")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[distr2,params2], 14, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-dist-periodos.csv"
    return response 

@app.get("/Plot/drift/periodos/tendencia-distr")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):


    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[distr2,params2], 14, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")


@app.get("/Datos/drift/fin/tendencia-ARMA")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,c2:float,desv2:float,s2: Union[int,None] = 0, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[c2,desv2,s2,phi2,teta2,[]], 15, num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-ARMA-fin.csv"
    return response 

@app.get("/Plot/drift/fin/tendencia-ARMA")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,c2:float,desv2:float,s2: Union[int,None] = 0, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[c2,desv2,s2,phi2,teta2,[]], 15, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/tendencia-ARMA")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, c2:float,desv2:float,s2: Union[int,None] = 0, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"),phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df =crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[c2,desv2,s2,phi2,teta2,[]], 15, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-ARMA-periodos.csv"
    return response 

@app.get("/Plot/drift/periodos/tendencia-ARMA")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, c2:float,desv2:float,s2: Union[int,None] = 0, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"),phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[c2,desv2,s2,phi2,teta2,[]], 15, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")
   
@app.get("/Datos/drift/fin/tendencia-periodicos")
async def obtener_serie(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, tipo2:int, distr2:int, p2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[tipo2,distr2,params2,p2], 16, num_drift)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-periodicos-fin.csv"
    return response 

@app.get("/Plot/drift/fin/tendencia-periodicos")
async def obtener_grafica(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, tipo2:int, distr2:int, p2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[tipo2,distr2,params2,p2], 16, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

@app.get("/Datos/drift/periodos/tendencia-periodicos")
async def obtener_serie(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int,tipo2:int, distr2:int,p2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[tipo2,distr2,params2,p2], 16, num_drift)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=datos-drift-tendencia-periodicos-periodos.csv"
    return response 


@app.get("/Plot/drift/periodos/tendencia-periodicos")
async def obtener_grafica(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int,tipo2:int, distr2:int,p2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    df = crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[tipo2,distr2,params2,p2], 16, num_drift)
    plot_df(df)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()

    return StreamingResponse(buffer,media_type="image/png")

# Crea una columna Target: y = a + b * x
def objetivo_lineal(df_caract,a,b,columna):
    df = df_caract.copy()
    df[columna] = a + b * df_caract[df_caract.columns[0]]
    return df

 
# Generación de nuevas variables a partir de las anterionres  
@app.post("/Variables/Lineal")
async def modify_item(a : float, b: float, indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_lineal(df,a,b, columna)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-lineal.csv"
    return response 
    
@app.post("/Plot/Variables/Lineal")
async def modify_item(a : float, b: float, indice:str, columna:str, file: UploadFile = File(...)) :

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_lineal(df,a,b, columna)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Crea una columna Target: y = sumatorio a[i] * x ^ i

def objetivo_polinomico(df_caract,a, columna):
    df = df_caract.copy()
    df[columna] = np.zeros(df.shape[0])
    for i in range(0,len(a)):
        df[columna] = df[columna] + a[i]*df_caract[df_caract.columns[0]]**i
    return df

@app.post("/Variables/Polinomico")
async def modify_item( indice:str, columna:str, a: List[float]= Query(...,description="Coeficientes"), file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = objetivo_polinomico(df,a, columna)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-polinomico.csv"
    return response 

@app.post("/Plot/Variables/Polinomico")
async def modify_item( indice:str, columna:str, a: List[float]= Query(...,description="Coeficientes"), file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_polinomico(df,a, columna)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Crea una columna Target: y = a * e ^ (b * x)

def objetivo_exp(df_caract,a,b,columna):
    df = df_caract.copy()
    df[columna] = a * np.exp(b* df_caract[df_caract.columns[0]])
    return df

@app.post("/Variables/Exponencial")
async def modify_item( a:float,b:float,indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_exp(df,a,b, columna)    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-exponencial.csv"
    return response 

@app.post("/Plot/Variables/Exponencial")
async def modify_item( a:float,b:float,indice:str, columna:str, file: UploadFile = File(...)) :

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_exp(df,a,b, columna)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Crea una columna Target: y = a + b * log (x)

def objetivo_log(df_caract,a,b,columna):
    df = df_caract.copy()
    df[columna] = a + b * np.log(df_caract[df_caract.columns[0]])
    return df

@app.post("/Variables/Logaritmica")
async def modify_item( a:float,b:float,indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_log(df,a,b, columna)    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-logaritmico.csv"
    return response 

@app.post("/Plot/Variables/Logaritmica")
async def modify_item( a:float,b:float,indice:str, columna:str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_log(df,a,b, columna)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Crea una columna Target: y = a + sumatorio b[i] * x [i]
def multivariante(df_caract,a,b,columna):
    df = df_caract.copy()
    df[columna] = a
    for k in range(0,len(b)):
        df[columna] = df[columna] + b[k] * df_caract[df.columns[k]]
    return df

@app.post("/Variables/Multivariante")
async def modify_item(a:float, indice:str, columna:str, b :List[float]= Query(...,description="Coeficientes"), file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    
    df1 = multivariante(df,a,b, columna)
    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-multivariante.csv"
    return response 

@app.post("/Plot/Variables/Multivariante")
async def modify_item(a:float, indice:str, columna:str, b :List[float]= Query(...,description="Coeficientes"), file: UploadFile = File(...)) :

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = multivariante(df,a,b, columna)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Crea una columna Target: y = a + sumatorio b[i][i] * x [i] + sumatorio b[j][k]x[j]x[k]

def interaccion(df_caract,a,b,columna):
    df = df_caract.copy()
    df[columna] = a
    for k in range(0,b.shape[0]):
        df[columna] = df[columna] + b[k][k] * df_caract[df.columns[k]]
        for i in range(k+1,b.shape[1]):
            df[columna] = df[columna] + b[k][i] * df_caract[df.columns[k]] * df_caract[df.columns[i]]
    return df

class MatrixBody(BaseModel):
    matrix: List[List[float]]
    
#Mirar matriz
@app.post("/Variables/Interaccion")
async def modify_item(a:float, indice:str, columna:str, b: str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    b1 = json.loads(b)
    m = np.zeros((len(b1),len(b1)))
    for i in range(0,m.shape[0]):
        for j in range(0,m.shape[1]):
            m[i][j] = b1[i][j]
    df1 = interaccion(df,a,m, columna)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-interaccion.csv"
    return response 

@app.post("/Plot/Variables/Interaccion")
async def modify_item(a:float, indice:str, columna:str, b: str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    b1 = json.loads(b)
    m = np.zeros((len(b1),len(b1)))
    for i in range(0,m.shape[0]):
        for j in range(0,m.shape[1]):
            m[i][j] = b1[i][j]

    df1 = interaccion(df,a,m, columna)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Crea una columna Target: y = a / x ^ n

def objetivo_prop_inversa(df_caract,a,n,columna):
    df = df_caract.copy()
    df[columna] = a / (df_caract[df_caract.columns[0]] ** n)
    return df

@app.post("/Variables/Inversa")
async def modify_item(a:float, n:int , indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_prop_inversa(df,a,n, columna)    
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-inversa.csv"
    return response 

@app.post("/Plot/Variables/Inversa")
async def modify_item(a:float, n:int , indice:str, columna:str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = objetivo_prop_inversa(df,a, n, columna)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Crea una columna Target: Si x < umbral, y = f(x). En otro caso, y = g(x)

def objetivo_escalonada(df_caract,f,g,umbral,columna):
    df = df_caract.copy()
    df[columna] = np.zeros(df_caract.shape[0])
    for k in df.index:
        x = df_caract.loc[k,df.columns[0]]
        if x < umbral:
            df.loc[k,columna] = f(x)
        else :
            df.loc[k,columna] = g(x)     
    return df

def lineal(x):
    return x + 5

def polinomica2(x):
    return 5 * x **2 + 3 * x - 10

def polinomica3(x):
    return 2 * x ** 3 - 7 * x ** 2 + 2 * x - 50

def polinomica4(x):
    return 3 * x **4 - 5 * x ** 3 + 2 * x ** 2 -10 * x + 3

def expon1(x):
    return math.exp(x-2) + 7

def expon2(x):
    return math.pow(2,x-4) - 19

def logaritmo(x):
    return math.log(x+8) -3

def raiz(x):
    return math.sqrt(x+3) - 24

def elegir_funcion(funcion):
    
    if funcion=='Lineal':
        return lineal
    elif funcion == 'Polinomica2':
        return polinomica2
    elif funcion == 'Polinomica3':
        return polinomica3
    elif funcion == 'Polinomica4':
        return polinomica4
    elif funcion == 'Exponencial':
        return expon1
    elif funcion == 'Exponencial2':
        return expon2
    elif funcion == 'Log':
        return logaritmo
    elif funcion == 'Raiz':
        return raiz
    elif funcion == 'Seno':
        return math.sin
    elif funcion == 'Coseno':
        return math.cos
    elif funcion == 'Tangente':
        return math.tan
    elif funcion == 'Absoluto':
        return math.fabs    
    elif funcion == 'Truncar':
        return math.trunc
    elif funcion == 'Log10':
        return math.log10
    elif funcion == 'Log1p':
        return math.log1p
    elif funcion == 'Log2p':
        return math.log2p
    elif funcion == 'Exp1':
        return math.expm1
    elif funcion == 'Ceil':
        return math.ceil
    
@app.post("/Variables/Escalonada")
async def modify_item(umbral:float, f: str,g:str , indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    f1 = elegir_funcion(f)
    g1 = elegir_funcion(g)
    df1 = objetivo_escalonada(df,f1,g1, umbral,columna)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-escalonada.csv"
    return response 

@app.post("/Plot/Variables/Escalonada")
async def modify_item(umbral:float, f: str,g:str , indice:str, columna:str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    f1 = elegir_funcion(f)
    g1 = elegir_funcion(g)
    df1 = objetivo_escalonada(df,f1,g1,umbral, columna)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")



# Crea una columna Target: Si x cumple condM --> y = fM(x1,...,xN)

def objetivo_condicional(df_caract,columna,cond,func):
    df = df_caract.copy()
    df[columna] = np.zeros(df_caract.shape[0])
    
    for k in df_caract.index:
        a = list()
        for j in df_caract.columns:
            a.append(df_caract.loc[k,j])
        find = False  
        i = 0
        while find == False and i < len(cond):
            if cond[i](a):
               df.loc[k,columna] = func[i](a)
               find = True
            i +=1
            
    return df

def igual(x,valor,indice):
    return x[indice]==valor

def menor(x,valor,indice):
    return x[indice]<valor

def mayor(x,valor,indice):
    return x[indice]>valor

def menorI(x,valor,indice):
    return x[indice] <= valor

def mayorI(x,valor,indice):
    return x[indice] >= valor

def iguales(x,valor1,valor2):
    return x[valor1] == x[valor2]

def mayores(x,valor1,valor2):
    return x[valor1] > x[valor2]

def menores(x,valor1,valor2):
    return x[valor1] < x[valor2]

def mayoresI(x,valor1,valor2):
    return x[valor1] >= x[valor2]

def menoresI(x,valor1,valor2):
    return x[valor1] <= x[valor2]

def verdad(x):
    return True

def elegir_condicion(cond):
    
    if cond[1:7]=='MenorI':
        ind = cond[0]
        indice = int(ind)
        valor = cond[7:]
        num = float(valor)
        return partial(menorI,valor=num,indice=indice)
    elif cond[1:7]=='MayorI':
        ind = cond[0]
        indice = int(ind)
        valor = cond[7:]
        num = float(valor)
        return partial(mayorI,valor=num,indice=indice)

    
    elif cond[1:9]=='MenoresI':
        valor1 = cond[0]
        valor2 = cond[9]
        num1 = int(valor1)
        num2 = int(valor2)
        return partial(menoresI,valor1=num1,valor2=num2)
    elif cond[1:9]=='MayoresI':
        valor1 = cond[0]
        valor2 = cond[9]
        num1 = int(valor1)
        num2 = int(valor2)
        return partial(mayoresI,valor1=num1,valor2=num2)
    
    elif cond[1:8]=='Iguales':
        valor1 = cond[0]
        valor2 = cond[8]
        num1 = int(valor1)
        num2 = int(valor2)
        return partial(iguales,valor1=num1,valor2=num2)
    elif cond[1:8]=='Menores':
        valor1 = cond[0]
        valor2 = cond[8]
        num1 = int(valor1)
        num2 = int(valor2)
        return partial(menores,valor1=num1,valor2=num2)
    elif cond[1:8]=='Mayores':
        valor1 = cond[0]
        valor2 = cond[8]
        num1 = int(valor1)
        num2 = int(valor2)
        return partial(mayores,valor1=num1,valor2=num2)
    
    if cond[1:6] == 'Igual':
        ind = cond[0]
        indice = int(ind)
        valor = cond[6:]
        num = float(valor)
        return partial(igual,valor=num,indice=indice)
    elif cond[1:6]=='Menor':
        ind = cond[0]
        indice = int(ind)
        valor = cond[6:]
        num = float(valor)
        return partial(menor,valor=num,indice=indice)
    elif cond[1:6]=='Mayor':
        ind = cond[0]
        indice = int(ind)
        valor = cond[6:]
        num = float(valor)
        return partial(mayor,valor=num,indice=indice)
    
    elif cond == 'default' :
        return verdad
   
def linealM(x):
    result = (random()-0.5) * 20
    for k in range(0,len(x)):
        n = (random()-0.5) * 20
        result += n * x[k]
    return result

def polinomica2M(x):
    result = 0
    for k in range(0,len(x)):
        n = (random()-0.5) * 20
        for j in range (1,3):
            result += n * x[k] ** j
    return result

def polinomica3M(x):
    result = 0
    for k in range(0,len(x)):
        n = (random()-0.5) * 20
        for j in range (1,4):
            result += n * x[k] ** j
    return result

def polinomica4M(x):
    result = 0
    for k in range(0,len(x)):
        n = (random()-0.5) * 20
        for j in range (1,5):
            result += n * x[k] ** j
    return result

def exponM(x):
    a = (random()-0.5) * 20
    b = (random()-0.5) * 20
    result = linealM(x)
    return math.exp(result-a) + b

def expon2M(x):
    a = (random()) * 10
    b = (random()-0.5) * 20
    result = linealM(x)/100
    return math.pow(2,result-a) - b

def logaritmoM(x):
    a = (random()) * 10
    b = (random()-0.5) * 20
    result = math.fabs(linealM(x))
    return math.log(result+a) - b

def raizM(x):
    a = (random()) * 10
    b = (random()-0.5) * 20
    result = math.fabs(linealM(x))
    return math.sqrt(result+a) - b

def sinM(x):
    result = linealM(x)
    return math.sin(result)

def cosM(x):
    result = linealM(x)
    return math.cos(result)

def tanM(x):
    result = linealM(x)
    return math.tan(result)

def absM(x):
    result = linealM(x)
    return math.fabs(result)

def truncM(x):
    result = linealM(x)
    return math.trunc(result)

def log10M(x):
    result = absM(x)
    return math.log10(result)

def log1pM(x):
    result = absM(x)
    return math.log1p(result)

def log2pM(x):
    result = absM(x)
    return math.log2(result)

def exp1M(x):
    result = linealM(x)
    return math.expm1(result)

def ceilM (x):
    result = linealM(x)
    return math.ceil(result)


def elegir_funcion_multi(funcion):
    
    if funcion=='Lineal':
        return linealM
    elif funcion == 'Polinomica2':
        return polinomica2M
    elif funcion == 'Polinomica3':
        return polinomica3M
    elif funcion == 'Polinomica4':
        return polinomica4M
    elif funcion == 'Exponencial':
        return exponM
    elif funcion == 'Exponencial2':
        return expon2M
    elif funcion == 'Log':
        return logaritmoM
    elif funcion == 'Raiz':
        return raizM
    elif funcion == 'Seno':
        return sinM
    elif funcion == 'Coseno':
        return cosM
    elif funcion == 'Tangente':
        return tanM
    elif funcion == 'Absoluto':
        return absM
    elif funcion == 'Truncar':
        return truncM
    elif funcion == 'Log10':
        return log10M
    elif funcion == 'Log1p':
        return log1pM
    elif funcion == 'Log2':
        return log2pM
    elif funcion == 'Exp1':
        return exp1M
    elif funcion == 'Ceil':
        return ceilM     


@app.post("/Variables/Condicional")
async def modify_item( funciones: List[str],condiciones: List[str] , indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    func = funciones[0].split(",")
    cond = condiciones[0].split(",")
    f = list()
    c = list ()
    for k in range(0,len(func)):
        f.append(elegir_funcion_multi(func[k]))
        
    for k in range(0, len(cond)):
        c.append(elegir_condicion(cond[k]))
        
    df1 = objetivo_condicional(df,columna,c,f)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-condicional.csv"
    return response 

@app.post("/Plot/Variables/Condicional")
async def modify_item( funciones: List[str],condiciones: List[str] , indice:str, columna:str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    func = funciones[0].split(",")
    cond = condiciones[0].split(",")
    f=list()
    c=list()

    for k in range(0,len(func)):
        f.append(elegir_funcion_multi(func[k]))

    for k in range(0,len(f)):
        c.append(elegir_condicion(cond[k]))

    df1 = objetivo_condicional(df,columna,c,f)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Crea una columna Target: y = f(x1,...,xN)

def objetivo_funcional(df_caract,columna,f):
    df = df_caract.copy()       
    df[columna] = np.zeros(df_caract.shape[0])
    for k in df_caract.index:
        a = list()
        for j in df_caract.columns:
            a.append(df_caract.loc[k,j])
        df.loc[k,columna]=f(a)
    return df



@app.post("/Variables/Funcional")
async def modify_item( funciones: str , indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    f = elegir_funcion_multi(funciones)
    df1 = objetivo_funcional(df,columna,f)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=objetivo-funcional.csv"
    return response 

@app.post("/Plot/Variables/Funcional")
async def modify_item( funciones: str , indice:str, columna:str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    f = elegir_funcion_multi(funciones)
    df1 = objetivo_funcional(df,columna,f)    
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Técnicas de aumentación de datos: 

# Definimos nuevos datos indicando el número de datos a generar, la frequencia y el tipo de interpolación (lineal/cubico).
def interpolacion_max(df,kind,num,freq):
    df=df.reset_index()
    indices=df.index.values
    indice=series_periodos(df[df.columns[0]][0],num,freq)
    x =indices 
    for i in range(1,len(df.columns)):
        y = df[df.columns[i]]
        f = interp1d(x, y, kind=kind) # kind ='linear' / 'cubic'
        x_new = np.linspace(0,df[df.columns[i]].argmax(), num=num)  # New x values
        y_new = f(x_new)  # Interpolated y values
        if i==1:
            df_int = pd.DataFrame(data=y_new,index=indice,columns=[df.columns[i]])
        else :     
            df_n = pd.DataFrame(data=y_new,index=indice,columns=[df.columns[i]])
            df_int= df_int.join(df_n, how="outer")
    return df_int

# Definimos nuevos datos indicando el número de datos a generar, la frequencia y el tipo de interpolación (lineal/cubico).
def interpolacion_min(df,kind,num,freq):
    df=df.reset_index()
    indices=df.index.values
    indice=series_periodos(df[df.columns[0]][0],num,freq)
    x =indices 
    for i in range(1,len(df.columns)):
        y = df[df.columns[i]]
        f = interp1d(x, y, kind=kind) # kind ='linear' / 'cubic'
        x_new = np.linspace(0,df[df.columns[i]].argmin(), num=num)  # New x values
        y_new = f(x_new)  # Interpolated y values
        if i==1:
            df_int = pd.DataFrame(data=y_new,index=indice,columns=[df.columns[i]])
        else :     
            df_n = pd.DataFrame(data=y_new,index=indice,columns=[df.columns[i]])
            df_int= df_int.join(df_n, how="outer")
            
    return df_int

def interpolate(data):
    interpolated_data = []
    for i in range(len(data) - 1):
        interpolated_data.append(data.iloc[i])
        interpolated_data.append((data.iloc[i] + data.iloc[i + 1]) / 2)  # Punto intermedio
    interpolated_data.append(data.iloc[-1])
    return np.array(interpolated_data)

# Añadimos datos que sean el punto de medio entre dos datos consecutivos
def punto_medio(df,freq):
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

def spline_interpolation(data, s=1):
    x = np.arange(len(data))
    spline = UnivariateSpline(x, data, s=s)
    return spline(x)

# Realizamos la interpolación spline 
def interpolacion_spline(df,s):
    for x in df.columns:
        df[x]=spline_interpolation(df[x],s)
    return df

# Interpolación
@app.post("/Aumentar/Interpolacion/Min")
async def modify_item(tipo : str, num: int, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = interpolacion_min(df,tipo,num,freq)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-interpolacion-min.csv"
    return response 

@app.post("/Plot/Aumentar/Interpolacion/Min")
async def modify_item(tipo : str, num: int, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    
    df1 = interpolacion_min(df,tipo,num,freq)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


@app.post("/Aumentar/Interpolacion/Max")
async def modify_item(tipo : str, num: int, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = interpolacion_max(df,tipo,num,freq)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-interpolacion-max.csv"
    return response 


@app.post("/Plot/Aumentar/Interpolacion/Max")
async def modify_item(tipo : str, num: int, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = interpolacion_max(df,tipo,num,freq)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

@app.post("/Aumentar/Interpolacion/Medio")
async def modify_item(freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = punto_medio(df,freq)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-interpolacion-medio.csv"
    return response 


@app.post("/Plot/Aumentar/Interpolacion/Medio")
async def modify_item(freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = punto_medio(df,freq)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


@app.post("/Aumentar/Interpolacion/Spline")
async def modify_item(s:int, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = interpolacion_spline(df,s)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-interpolacion-spline.csv"
    return response 

@app.post("/Plot/Aumentar/Interpolacion/Spline")
async def modify_item(s:int, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = interpolacion_spline(df,s)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Random Sampling

def sampling(df,size,freq):
    indice = series_periodos(df.index[0],size+df.shape[0],freq)
    for x in df.columns:
        data = df[x]
        sampled_data = np.random.choice(data, size=size, replace=True)
        if x == df.columns[0]:
            df_sampling=pd.DataFrame(data=np.concatenate((data,sampled_data)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((data,sampled_data)),index=indice,columns=[x])
            df_sampling= df_sampling.join(df_new, how="outer")
    return df_sampling


@app.post("/Aumentar/Sampling")
async def modify_item(size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = sampling(df,size,freq)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-sampling.csv"
    return response 


@app.post("/Plot/Aumentar/Sampling")
async def modify_item(size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = sampling(df,size,freq)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Técnicas estadísticas

# Devuelve df con datos añadidos calculados a partir de una distribución normal con la media y desviación de los datos pasados 

def normal(df,freq,size):
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

# Devuelve df con datos añadidos calculados a partir de una distribución lognormal cuya media es el logaritmo de la media de los datos pasados 
def log_normal(df,freq,sigma,size):
    indice=series_periodos(df.index[0],size+df.shape[0],freq)
    for x in df.columns:
        data = df[x].values
        data_augmented = np.random.lognormal(mean=np.log(data.mean()),sigma=sigma,size=size)
        if x == df.columns[0]:
            df_lognormal=pd.DataFrame(data=np.concatenate((df[x].values,data_augmented)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((data,data_augmented)),index=indice,columns=[x])
            df_lognormal= df_lognormal.join(df_new, how="outer")
    return df_lognormal

# Calcula nuevos datos usando: media + z * desv donde la media y las desv son las de los datos pasados y z = raiz (-2 * log u1) cos(2 pi u2) tal que u1,u2 son dos randoms entre 0 e 1

def box_muller_transform(mean, std_dev, size=100):
    u1, u2 = np.random.rand(size), np.random.rand(size)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + z1 * std_dev

def box_muller(df,freq,size):
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

@app.post("/Aumentar/Normal")
async def modify_item(size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = normal(df,freq,size)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-normal.csv"
    return response 

@app.post("/Plot/Aumentar/Normal")
async def modify_item(size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = normal(df,freq,size)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

@app.post("/Aumentar/Lognormal")
async def modify_item(sigma:float, size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = log_normal(df,freq,sigma,size)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-lognormal.csv"
    return response 

@app.post("/Plot/Aumentar/Lognormal")
async def modify_item(sigma:float, size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = log_normal(df,freq,sigma,size)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


@app.post("/Aumentar/Muller")
async def modify_item(size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = box_muller(df,freq,size)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-muller.csv"
    return response 


@app.post("/Plot/Aumentar/Muller")
async def modify_item(size:int,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = box_muller(df,freq,size)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Bootstrapping 
# Obtenemos nuevos datos barajando los originales + introduciendo ruido

def agregar_bootstrapping(df,freq):
    
    for x in df.columns:
        synthetic_data = df.sample(frac=1, replace=True).reset_index(drop=True)
        synthetic_data[x] += np.random.normal(0, 0.1, len(synthetic_data))  # Añadir ruido
        indice=series_periodos(df.index[0],len(df)+len(synthetic_data),freq)
        a=pd.concat([df[x],synthetic_data[x]])
        a.index=indice
        if x == df.columns[0]:
            df_bootstrap=pd.DataFrame(data=a)
        else:
            df_bootstrap= df_bootstrap.join(a, how="outer")
    return df_bootstrap

@app.post("/Aumentar/Bootstrap")
async def modify_item(freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_bootstrapping(df,freq)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-bootstrapping.csv"
    return response 

@app.post("/Plot/Aumentar/Bootstrap")
async def modify_item(freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_bootstrapping(df,freq)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Duplicar

def duplicate_and_perturb(data, duplication_factor=0.3, perturbation_std=0.05):
    duplicated_data = []
    np.random.seed(8)
    for point in data:
        duplicated_data.append(point)
        if np.random.rand() < duplication_factor:
            duplicated_data.append(point + np.random.normal(0, perturbation_std))
    return np.array(duplicated_data)

# Duplicamos algunos datos añadiendole cierto ruido.

def duplicados(df,freq,duplication_factor=0.3,perturbation_std=0.05):
    
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

@app.post("/Aumentar/Duplicado")
async def modify_item(freq:str,duplication_factor:float, perturbation_std: float, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = duplicados(df,freq,duplication_factor,perturbation_std)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-duplicado.csv"
    return response 



@app.post("/Plot/Aumentar/Duplicado")
async def modify_item(freq:str,duplication_factor:float, perturbation_std: float, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = duplicados(df,freq,duplication_factor,perturbation_std)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Comb lineal

# Calculamos nuevos datos como combinación lineal de los otros 

def linear_combinations(data, n_combinations):
    combinations = []
    for _ in range(n_combinations):
        weights = np.random.rand(data.shape[0])
        weights /= np.sum(weights)  # Normalizar pesos
        combination = np.dot(weights, data)
        combinations.append(combination)
    return np.array(combinations)

def agregar_comb(df,freq,size):
    for x in df.columns:
        data = df[x]
        data_augmented = linear_combinations(data,size)
        datos=np.concatenate((data.values,data_augmented))
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_dl = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_dl = df_dl.join(df_new, how="outer")
    return df_dl

@app.post("/Aumentar/Comb_lineal")
async def modify_item(freq:str,size:int, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_comb(df,freq,size)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-comb-lineal.csv"
    return response 

@app.post("/Plot/Aumentar/Comb_lineal")
async def modify_item(freq:str,size:int, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_comb(df,freq,size)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Técnicas que realizan modificaciones en los datos:

# Traslacion

# Desplazamiento espacial de la serie
def traslacion(df,shift,freq):
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

@app.post("/Aumentar/Traslacion")
async def modify_item(shift:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = traslacion(df,shift,freq)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-traslacion.csv"
    return response 

@app.post("/Plot/Aumentar/Traslacion")
async def modify_item(shift:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = traslacion(df,shift,freq)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")
    
# Agregación de ruido harmónico

# Añadimos ruido harmonico a la muestra con cierta amplitud y frequencia

def add_harmonic_noise(df,freq, amplitude=0.1, frequency=0.5):
    df_harm = df.copy()
    for x in df_harm.columns:
        data = df[x]
        time = np.arange(len(data))
        harmonic_noise = amplitude * np.sin(2 * np.pi * frequency * time)
        data_augmented = data + harmonic_noise
        datos = np.concatenate((data.values,data_augmented))
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_harm = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_harm = df_harm.join(df_new, how="outer")
    
    return df_harm


@app.post("/Aumentar/Harmonico")
async def modify_item(amplitude:float, frequency:float,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = add_harmonic_noise(df,freq, amplitude, frequency)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-ruido-harmonico.csv"
    return response 

@app.post("/Plot/Aumentar/Harmonico")
async def modify_item(amplitude:float, frequency:float,freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = add_harmonic_noise(df,freq, amplitude, frequency)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Escalado

# Desplazamiento espacial de la serie
def escalado(df,freq,factor):
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

@app.post("/Aumentar/Escalado")
async def modify_item(factor:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = escalado(df,freq,factor)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-escalado.csv"
    return response 

@app.post("/Plot/Aumentar/Escalado")
async def modify_item(factor:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = escalado(df,freq,factor)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Saltos
def pulse_noise(data, num_pulses=5, amplitude=1):
    pulse_indices = np.random.choice(len(data), num_pulses, replace=False)
    pulse_data = list()
    for i in pulse_indices:
        pulse_data.append(data[i]+ np.random.uniform(-amplitude, amplitude))
    return pulse_data

# Calculamos nuevos datos como combinación lineal de los otros 

def agregar_saltos(df,freq,num_saltos,amplitud):
    for x in df.columns:
        data = df[x]
        data_augmented = pulse_noise(data,num_saltos,amplitud)
        datos=np.concatenate((data.values,data_augmented))
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_saltos = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_saltos = df_saltos.join(df_new, how="outer")
    return df_saltos


@app.post("/Aumentar/Saltos")
async def modify_item(num_saltos:int, amplitud:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    df1 = agregar_saltos(df,freq,num_saltos,amplitud)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-saltos.csv"
    return response 

@app.post("/Plot/Aumentar/Saltos")
async def modify_item(num_saltos:int, amplitud:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    df1 = agregar_saltos(df,freq,num_saltos,amplitud)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Mixup
def mixup(data, alpha=0.2):
    lambda_ = np.random.beta(alpha, alpha)
    indices = np.random.permutation(len(data))
    data_mixup = lambda_ * data + (1 - lambda_) * data[indices]
    return data_mixup

# Agregar combinación del data junto a otro dato aleatorio
def agregar_mixup(df,freq,alpha=0.2):
    df_mix =df.copy()
    for x in df_mix.columns:
        data = df[x]
        data_augmented = mixup(data,alpha)
        datos = np.concatenate((data.values,data_augmented))
        if x == df.columns[0]:
            indice = series_periodos(df.index[0],len(datos),freq)
            df_mix = pd.DataFrame(data=datos,index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data = datos,index=indice,columns=[x])
            df_mix= df_mix.join(df_new, how="outer")
    return df_mix

@app.post("/Aumentar/Mixup")
async def modify_item(alpha:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_mixup(df,freq,alpha)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-mixup.csv"
    return response 

@app.post("/Plot/Aumentar/Mixup")
async def modify_item(alpha:float, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_mixup(df,freq,alpha)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

def random_mix(data, n_samples=100):
    mixed_data = []
    for _ in range(n_samples):
        i, j = np.random.choice(len(data), 2, replace=False)
        mixed_data.append((data[i] + data[j]) / 2)
    return np.array(mixed_data)

# Los valores se calculan tomando dos valores al azar y haciendo la media

def agregar_random_mix(df,freq,n_samples):
    indice=series_periodos(df.index[0],n_samples+df.shape[0],freq)
    for x in df.columns:
        sampled_data = random_mix(df[x],n_samples)
        if x == df.columns[0]:
            df_sampling=pd.DataFrame(data=np.concatenate((df[x],sampled_data)),index=indice,columns=[x])
        else:
            df_new = pd.DataFrame(data=np.concatenate((df[x],sampled_data)),index=indice,columns=[x])
            df_sampling= df_sampling.join(df_new, how="outer")
    return df_sampling

@app.post("/Aumentar/Random_mix")
async def modify_item(n_samples:int, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_random_mix(df,freq,n_samples)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-random-mix.csv"
    return response 

@app.post("/Plot/Aumentar/Random_mix")
async def modify_item(n_samples:int, freq:str, indice:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_random_mix(df,freq,n_samples)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")

# Transformaciones matemáticas

# Aplicamos log
def agregar_log(df):
    df_o = df.copy()
    for x in df_o.columns:
        df_o[x] = np.log1p(df[x])
    return df_o


# Aplicamos la raíz cuadrada 
def agregar_matematica(df,freq,funcion,factor=1):
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


@app.post("/Aumentar/Matematica")
async def modify_item(funcion:str, freq:str, indice:str,factor : Union[float,None] = 1, file: UploadFile = File(...)) :

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_matematica(df,freq,funcion,factor)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-matematica.csv"
    return response 

@app.post("/Plot/Aumentar/Matematica")
async def modify_item(funcion:str, freq:str, indice:str,factor : Union[float,None] = 1, file: UploadFile = File(...)) :

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")
    
    df1 = agregar_matematica(df,freq,funcion,factor)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


# Técnicas de reducción 

# Ventana deslizante

# Calculo con ventanas deslizantes del tamaño pasado como parámetro

def ventanas(df,ventana):
    df_o = df.copy()
    for x in df.columns:
        df_o[x] = df_o[x].rolling(window=ventana).mean()
    df_o = df_o.dropna()
    return df_o

@app.post("/Aumentar/Ventana")
async def modify_item(ventana:int, indice:str, file: UploadFile = File(...)) :

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = ventanas(df,ventana)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-ventana.csv"
    return response 

@app.post("/Plot/Aumentar/Ventana")
async def modify_item(ventana:int, indice:str, file: UploadFile = File(...)) :

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = ventanas(df,ventana)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")
     
# Recorte

def crop(series, start, end):
    return series[start:end]

# Recortamos la serie quedándonos solo con una parte desde la posición de inicio al fin

def recorte(df_i,start,end):
    df_o = df_i.copy()
    for x in df_o.columns:
        df_o[x] = crop(df_o[x],start,end)
    df_o=df_o.dropna()
    return df_o

@app.post("/Aumentar/Recorte")
async def modify_item(start:int,end:int, indice:str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = recorte(df,start,end)
    # Convertir el DataFrame a un buffer de CSV
    stream = io.StringIO()
    df1.to_csv(stream,index_label="Indice")
    stream.seek(0)

    # Devolver el archivo CSV como respuesta
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=aumentar-recorte.csv"
    return response 

@app.post("/Plot/Aumentar/Recorte")
async def modify_item(start:int,end:int, indice:str, file: UploadFile = File(...)) :
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    df1 = recorte(df,start,end)
    plot_df(df1)
    buffer = io.BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer,media_type="image/png")


