from io import StringIO
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# Imports
import pandas as pd
import numpy as np
from pathlib import Path  
from scipy.stats import binom,poisson,geom,hypergeom,uniform,expon, gamma, beta,chi2,t,pareto,lognorm
from random import randrange, random
import math
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
def read_root():
    return {"Contenido": "Esto es una API para generar datos sintéticos a partir de ciertos parámetros"}

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
    return a + b * math.log(t)

def tendencia_det(params,tipo,num_datos,coef_error=0):
    
    datos = np.zeros(num_datos)
    
    for t in range(1,num_datos+1):
        if coef_error is None:
            coef_error=0
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

@app.get("/Datos/tendencia/fin")
def read_item(inicio: str, fin:str, freq:str, tipo:int , error: Union[float, None] = None, columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la tendencia")):

    return {
        pasar_csv(crear_df_fin_tend_det(inicio,fin,freq,columna,params,tipo,error))

    }

@app.get("/Datos/tendencia/periodos")
def read_item(inicio: str, periodos:int, freq:str, tipo:int , error: Union[float, None] = None,  columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float]= Query(...,description="Parametros de la tendencia")):
    return {
        pasar_csv(crear_df_periodos_tend_det(inicio,periodos,freq,columna,params,tipo,error))
    }
    
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

@app.get("/Datos/distribucion/fin")
def read_item(inicio: str, fin:str, freq:str, distr:int , columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la distribución")):

    return {
        pasar_csv(crear_df_fin_datos(inicio,fin,freq,columna,distr,params))

    }

@app.get("/Datos/distribucion/periodos")
def read_item(inicio: str, periodos:int, freq:str, distr:int,  columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float]= Query(...,description="Parametros de la distribución")):
    return {
        pasar_csv(crear_df_periodos_datos(inicio,periodos,freq,columna,distr,params))
    }
    

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

@app.get("/Datos/periodicos/fin")
def read_item(inicio: str, fin:str, freq:str, distr:int, p: int, tipo:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float] = Query(...,description="Parametros de la distribución")):

    return {
        pasar_csv(crear_df_fin_periodicos(inicio,fin,freq,columna,distr,params,p,tipo))

    }

@app.get("/Datos/periodicos/periodos")
def read_item(inicio: str, periodos:int, freq:str, distr:int, p:int, tipo:int,  columna: List[str]= Query(...,description="Nombres de las columnas"), params: List[float]= Query(...,description="Parametros de la distribución")):
    return {
        pasar_csv(crear_df_periodos_periodicos(inicio,periodos,freq,columna,distr,params,p,tipo))
    }
 
 
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
    if s is None :
        s = 0
    indice = series_fin(inicio,fin,freq)
    num_datos = indice.size
    datos=creacion_modelos_ARMA(c,num_datos,desv,s,phi,teta,a)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

def crear_df_periodos_ARMA(inicio,periodos,freq,columna,c,desv,s=0,phi=[],teta=[],a=[]):
    if s is None :
        s = 0
    indice = series_periodos(inicio,periodos,freq)
    datos=creacion_modelos_ARMA(c,periodos,desv,s,phi,teta,a)
    df = pd.DataFrame(data=datos,index=indice,columns=columna)
    df.plot(title='Serie Temporal',figsize=(13,5))
    return df 

@app.get("/Datos/ARMA/fin")
def read_item(inicio: str, fin:str, freq:str,c:float, desv:float, s : Union[int, None] = None, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
    return {
        pasar_csv(crear_df_fin_ARMA(inicio,fin,freq,columna,c,desv,s,phi,teta))

    }
    
@app.get("/Datos/ARMA/periodos")
def read_item(inicio: str, periodos:int, freq:str,c:float, desv:float, s : Union[int, None] = None, columna: List[str]= Query(...,description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):
    return {
        pasar_csv(crear_df_periodos_ARMA(inicio,periodos,freq,columna,c,desv,s,phi,teta))

    }
    
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
def read_item(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, dist2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[dist2,params2],1,num_drift))

    }

@app.get("/Datos/drift/periodos/dist-dist")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int,dist2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[dist2,params2],1,num_drift))
    }
 
@app.get("/Datos/drift/fin/dist-ARMA")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, c:float, desv:float, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[c,desv,s,phi,teta,[]],2,num_drift))

    }

@app.get("/Datos/drift/periodos/dist-ARMA")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, c:float ,desv:float ,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[c,desv,s,phi,teta,[]],2,num_drift))
    }

 
@app.get("/Datos/drift/fin/dist-periodico")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, tipo2:int, dist2:int, p2:int,columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[tipo2,dist2,params2,p2],3,num_drift))

    }

@app.get("/Datos/drift/periodos/dist-periodico")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, tipo2:int, dist2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[tipo2,dist2,params2,p2],3,num_drift))
    }
    
@app.get("/Datos/drift/fin/dist-tendencia")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, dist1:int, tipo2:int, coef_error: Union[int,None] = 0 ,columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la tendencia")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[dist1,params1],[params2,tipo2,coef_error],4,num_drift))

    }

@app.get("/Datos/drift/periodos/dist-tendencia")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, dist1:int, tipo2:int, coef_error : Union[int,None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la tendencia")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[dist1,params1],[params2,tipo2,coef_error],4,num_drift))
    }
    
@app.get("/Datos/drift/fin/ARMA-ARMA")
def read_item(inicio: str, fin:str, freq:str, num_drift:int ,c1:float , desv1:float, c2:float, desv2:float, s1: Union[int,None] = 0, s2: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi1: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta1:Optional[List[float]]= Query([],description="Parámetros medias móviles"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[c1,desv1,s1,phi1,teta1,[]],[c2,desv2,s2,phi2,teta2,[]],5,num_drift))

    }

@app.get("/Datos/drift/periodos/ARMA-ARMA")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int ,c1:float , desv1:float, c2:float , desv2:float, s1: Union[int,None] = 0, s2: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi1: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta1:Optional[List[float]]= Query([],description="Parámetros medias móviles"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c1,desv1,s1,phi1,teta1,[]],[c2,desv2,s2,phi2,teta2,[]],5,num_drift))
    }



@app.get("/Datos/drift/fin/ARMA-dist")
def read_item(inicio: str, fin:str, freq:str, num_drift:int ,c:float , desv:float, dist2:int,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[dist2,params2],6,num_drift))

    }

@app.get("/Datos/drift/periodos/ARMA-dist")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int ,c:float, desv:float, dist2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[dist2,params2],6,num_drift))
    }

@app.get("/Datos/drift/fin/ARMA-periodicos")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, c:float, desv:float, tipo2:int, distr2:int, p2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[tipo2,distr2,params2,p2],7,num_drift))

    }

@app.get("/Datos/drift/periodos/ARMA-periodicos")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int ,c:float , desv:float, tipo2:int, distr2:int, p2:int, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[tipo2,distr2,params2,p2],7,num_drift))
    }
    
@app.get("/Datos/drift/fin/ARMA-tendencia")
def read_item(inicio: str, fin:str, freq:str, num_drift:int ,c:float , desv:float, tipo2:int,coef_error: Union[float, None] = 0,s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la tendencia determinista")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[c,desv,s,phi,teta,[]],[params2,tipo2,coef_error],8,num_drift))

    }

@app.get("/Datos/drift/periodos/ARMA-tendencia")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int ,c:float , desv:float, tipo2:int,coef_error: Union[float, None] = 0, s: Union[int,None] = 0, columna: List[str]= Query(description="Nombres de las columnas"), phi: Optional[List[float]]= Query([],description="Parámetros autorregresivos"), teta:Optional[List[float]]= Query([],description="Parámetros medias móviles"), params2: List[float]= Query(...,description="Parametros de la tendencia determinista")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[c,desv,s,phi,teta,[]],[params2,tipo2,coef_error],8,num_drift))
    }

@app.get("/Datos/drift/fin/periodico-periodico")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int, distr2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[tipo2,distr2,params2,p2], 9, num_drift))

    }

@app.get("/Datos/drift/periodos/periodico-periodico")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int, distr2:int, p2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[tipo2,distr2,params2,p2], 9, num_drift))
    }
    
@app.get("/Datos/drift/fin/periodico-distr")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, distr2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[distr2,params2], 10, num_drift))

    }

@app.get("/Datos/drift/periodos/periodico-distr")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, distr2:int, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[distr2,params2], 10, num_drift))
    }
    

@app.get("/Datos/drift/fin/periodico-ARMA")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, c2:float, desv2:float, s2 : Union[None,int] = 0,  columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"),  phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[c2,desv2,s2,phi2,teta2,[]], 11, num_drift))

    }

@app.get("/Datos/drift/periodos/periodico-ARMA")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, c2:float, desv2:float, s2 : Union[None,int] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"),  phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[c2,desv2,s2,phi2,teta2,[]], 11, num_drift))
    }
    
@app.get("/Datos/drift/fin/periodico-tendencia")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int,coef_error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[tipo1,distr1,params1,p1],[params2,tipo2,coef_error], 12, num_drift))

    }

@app.get("/Datos/drift/periodos/periodico-tendencia")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr1:int, p1:int, tipo2:int,coef_error: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera distribución"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[tipo1,distr1,params1,p1],[params2,tipo2,coef_error], 12, num_drift))
    }
    
@app.get("/Datos/drift/fin/tendencia-tendencia")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,tipo2:int, coef_error1: Union[float, None] = 0,coef_error2: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[params2,tipo2,coef_error2], 13, num_drift))

    }

@app.get("/Datos/drift/periodos/tendencia-tendencia")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, tipo2:int, coef_error1: Union[float, None] = 0, coef_error2: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda tendencia")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[params2,tipo2,coef_error2], 13, num_drift))
    }
 
@app.get("/Datos/drift/fin/tendencia-distr")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,distr2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[distr2,params2], 14, num_drift))

    }

@app.get("/Datos/drift/periodos/tendencia-distr")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, distr2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[distr2,params2], 14, num_drift))
    }
   

@app.get("/Datos/drift/fin/tendencia-ARMA")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int,c2:float,desv2:float,s2: Union[int,None] = 0, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[c2,desv2,s2,phi2,teta2,[]], 15, num_drift))

    }

@app.get("/Datos/drift/periodos/tendencia-ARMA")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int, c2:float,desv2:float,s2: Union[int,None] = 0, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"),phi2: Optional[List[float]]= Query([],description="Parámetros autorregresivos 2"), teta2:Optional[List[float]]= Query([],description="Parámetros medias móviles 2")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[c2,desv2,s2,phi2,teta2,[]], 15, num_drift))
    }
   
@app.get("/Datos/drift/fin/tendencia-periodicos")
def read_item(inicio: str, fin:str, freq:str, num_drift:int, tipo1:int, tipo2:int, distr2:int, p2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float] = Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_fin_DRIFT(inicio,fin,freq,columna,[params1,tipo1,coef_error1],[tipo2,distr2,params2,p2], 16, num_drift))

    }

@app.get("/Datos/drift/periodos/tendencia-periodicos")
def read_item(inicio: str, periodos:int, freq:str, num_drift:int, tipo1:int,tipo2:int, distr2:int,p2:int, coef_error1: Union[float, None] = 0, columna: List[str]= Query(...,description="Nombres de las columnas"), params1: List[float]= Query(...,description="Parametros de la primera tendencia"), params2: List[float]= Query(...,description="Parametros de la segunda distribución")):

    return {
        pasar_csv(crear_df_periodos_DRIFT(inicio,periodos,freq,columna,[params1,tipo1,coef_error1],[tipo2,distr2,params2,p2], 16, num_drift))
    }
    

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
    
    result = pasar_csv(df1)

    return {result}
    

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
    
    result = pasar_csv(df1)

    return {result}

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
    
    result = pasar_csv(df1)

    return {result}

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
    
    result = pasar_csv(df1)

    return {result}

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
    
    result = pasar_csv(df1)

    return {result}

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
async def modify_item(a:float, indice:str, columna:str, b: MatrixBody, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    
    df1 = interaccion(df,a,b, columna)
    
    result = pasar_csv(df1)

    return {result}

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
    
    result = pasar_csv(df1)

    return {result}

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

@app.post("/Variables/Escalonada")
async def modify_item(umbral:float, n:int, f:float,g:float , indice:str, columna:str, file: UploadFile = File(...)) :
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Leer el archivo CSV en un DataFrame de pandas
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data,index_col=indice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {e}")

    
    df1 = objetivo_escalonada(df,f,g, umbral,columna)
    
    result = pasar_csv(df1)

    return {result}