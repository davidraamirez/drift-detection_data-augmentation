import pandas as pd
import numpy as np
import math
from functools import partial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Relación lineal

def objetivo_lineal(df_caract: pd.DataFrame, a: float, b: float, columna: str) -> pd.DataFrame:
    """
    Crea una variable exógena con una relación lineal de la forma y = a + b * x.

    Args:
        df_caract (pd.DataFrame): DataFrame que contiene una columna con la variable x.
        a (float): Término independiente de la función lineal.
        b (float): Coeficiente que multiplica a la variable x.
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores de y.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y.
    """
    df = df_caract.copy()
    df[columna] = a + b * df_caract[df_caract.columns[0]]
    return df

# Relación Polinómica

def objetivo_polinomico(df_caract,a,columna):
    """
    Genera una variable exógena basada en un polinomio de la forma y = ∑ a[i] * x^i.

    Args:
        df_caract (pd.DataFrame): DataFrame que contiene una columna con la variable x.
        a (list[float]): Lista de coeficientes del polinomio, donde a[i] es el coeficiente del término de grado i.
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores de y.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y calculada polinómicamente.
    """
    df = df_caract.copy()
    df[columna] = np.zeros(df.shape[0])
    for i in range(0,len(a)):
        df[columna] = df[columna] + a[i]*df_caract[df_caract.columns[0]]**i
    return df

# Relación Exponencial

def objetivo_exp(df_caract,a,b,columna):
    """
    Genera una variable exógena con una relación exponencial de la forma y = a + e^(b * x).

    Args:
        df_caract (pd.DataFrame): DataFrame que contiene una columna con la variable x.
        a (float): Término constante que se suma al resultado exponencial.
        b (float): Exponente que multiplica a la variable x dentro de la función exponencial.
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores de y.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y calculada exponencialmente.
    """
    df = df_caract.copy()
    df[columna] = a + np.exp(b* df_caract[df_caract.columns[0]])
    return df

# Relación Logarítmica

def objetivo_log(df_caract,a,b,columna):
    """
    Genera una variable exógena basada en una relación logarítmica: y = a + b * log(x).

    Args:
        df_caract (pd.DataFrame): DataFrame que contiene una columna con la variable x.
        a (float): Término constante que se suma al resultado logarítmico.
        b (float): Coeficiente que multiplica al logaritmo de la variable x.
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores de y.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y calculada logarítmicamente.
    """
    df = df_caract.copy()
    df[columna] = a + b * np.log(df_caract[df_caract.columns[0]])
    return df

# Relación multivariante lineal

def multivariante(df_caract,a,b,columna):
    """
    Genera una variable exógena a partir de una combinación lineal multivariante: y = a + ∑ b[i] * x[i].

    Args:
        df_caract (pd.DataFrame): DataFrame con múltiples columnas que representan las variables independientes x[i].
        a (float): Término constante que se suma en la combinación lineal.
        b (list[float]): Lista de coeficientes correspondientes a cada variable independiente.
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores de y.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y generada mediante combinación lineal multivariante.
    """
    df = df_caract.copy()
    df[columna] = a
    for k in range(0,len(b)):
        df[columna] = df[columna] + b[k] * df_caract[df.columns[k]]
    return df

# Relación de interacción

def interaccion(df_caract,a,b,columna):
    """
    Genera una variable exógena considerando términos lineales y de interacción entre variables:  
    y = a + ∑ b[i][i] * x[i] + ∑ b[j][k] * sqrt(x[j] * x[k]) para j ≠ k.

    Args:
        df_caract (pd.DataFrame): DataFrame con múltiples columnas que representan variables independientes x[i].
        a (float): Término constante que se suma a la combinación.
        b (np.ndarray): Matriz cuadrada de coeficientes, donde b[i][i] corresponde a términos lineales y b[j][k] a coeficientes de interacción.
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores de y.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y calculada con términos de interacción.
    """
    df = df_caract.copy()
    df[columna] = a
    for k in range(0,b.shape[0]):
        df[columna] = df[columna] + b[k][k] * df_caract[df.columns[k]]
        for i in range(k+1,b.shape[1]):
            df[columna] = df[columna] + b[k][i] * np.sqrt(df_caract[df.columns[k]] * df_caract[df.columns[i]])
    return df

# Relación inversa

def objetivo_prop_inversa(df_caract,a,n,columna):
    """
    Genera una variable exógena con una relación de proporcionalidad inversa: y = a / x^n.

    Args:
        df_caract (pd.DataFrame): DataFrame que contiene una columna con la variable x.
        a (float): Constante numerador.
        n (float): Exponente aplicado a la variable x en el denominador.
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores de y.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y calculada como una función de proporcionalidad inversa.
    """
    df = df_caract.copy()
    df[columna] = a / (df_caract[df_caract.columns[0]] ** n)
    return df

# Relación escalonada

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
    """
    Devuelve una función matemática en función de una cadena identificadora.

    Args:
        funcion (str): Nombre de la función (e.g., 'Lineal', 'Log', 'Polinomica2', etc.).

    Returns:
        function: Función correspondiente, que toma un float y devuelve un float.
    """
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
    elif funcion == 'Log2':
        return math.log2
    elif funcion == 'Exp1':
        return math.expm1
    elif funcion == 'Ceil':
        return math.ceil

def objetivo_escalonada(df_caract,f,g,umbral,columna):
    """
    Genera una variable exógena aplicando una función por tramos:  
    Si x < umbral → y = f(x), si x ≥ umbral → y = g(x).

    Args:
        df_caract (pd.DataFrame): DataFrame con una columna con la variable x.
        f (function): Función aplicada a los valores de x menores que el umbral.
        g (function): Función aplicada a los valores de x mayores o iguales al umbral.
        umbral (float): Valor umbral que define el cambio entre f(x) y g(x).
        columna (str): Nombre de la nueva columna que se añadirá al DataFrame con los valores generados.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable y, definida por tramos.
    """
    df = df_caract.copy()
    df[columna] = np.zeros(df_caract.shape[0])
    for k in df.index:
        x = df_caract.loc[k,df.columns[0]]
        if x < umbral:
            df.loc[k,columna] = f(x)
        else :
            df.loc[k,columna] = g(x)     
    return df

# Relación funcional

def linealM(x):
    np.random.seed(1)
    result = (np.random.rand()-0.5) * 5
    for k in range(0,len(x)):
        n = (np.random.rand()-0.5) * 5
        result += n * x[k]
    return result

def polinomica2M(x):
    np.random.seed(1)
    result = 0
    for k in range(0,len(x)):
        n = (np.random.rand()-0.5) * 2
        for j in range (1,3):
            result += n * x[k] ** j
    return result

def polinomica3M(x):
    np.random.seed(1)
    result = 0
    for k in range(0,len(x)):
        n = (np.random.rand()-0.5) 
        for j in range (1,4):
            result += n * x[k] ** j
    return result

def polinomica4M(x):
    np.random.seed(1)
    result = 0
    for k in range(0,len(x)):
        n = (np.random.rand()-0.5)
        for j in range (1,5):
            result += n * x[k] ** j
    return result

def exponM(x):
    np.random.seed(1)
    a = (np.random.rand()-0.5) * 5
    b = (np.random.rand()-0.5) * 5
    result = linealM(x)
    return math.exp(result-a) + b

def expon2M(x):
    np.random.seed(1) 
    a = (np.random.rand()) * 5
    b = (np.random.rand()-0.5) * 5
    result = linealM(x)/100
    return math.pow(2,result-a) - b

def logaritmoM(x):
    np.random.seed(1)
    a = (np.random.rand()) * 10
    b = (np.random.rand()-0.5) * 20
    result = math.fabs(linealM(x))
    return math.log(result+a) - b

def raizM(x):
    np.random.seed(1)
    a = (np.random.rand()) * 10
    b = (np.random.rand()-0.5) * 20
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

def ceilM (x):
    result = linealM(x)
    return math.ceil(result)

def elegir_funcion_multi(funcion):
    """
    Devuelve una función multivariable con comportamiento matemático predefinido.

    Args:
        funcion (str): Nombre de la función deseada (e.g., 'Lineal', 'Polinomica2', 'Exponencial', etc.).

    Returns:
        function: Función que toma una lista de floats (valores de entrada) y devuelve un float (resultado).
    """
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
    elif funcion == 'Ceil':
        return ceilM  
    
def objetivo_funcional(df_caract,columna,f):
    """
    Genera una variable exógena aplicando una función multivariante arbitraria a cada fila del DataFrame.

    Args:
        df_caract (pd.DataFrame): DataFrame con múltiples columnas, cada una representando una variable independiente.
        columna (str): Nombre de la nueva columna a añadir al DataFrame con los valores generados.
        f (function): Función que toma como entrada una lista con los valores de todas las columnas de una fila.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna que contiene la variable generada.
    """
    df = df_caract.copy()       
    df[columna] = np.zeros(df_caract.shape[0])
    for k in df_caract.index:
        a = list()
        for j in df_caract.columns:
            a.append(df_caract.loc[k,j])
        df.loc[k,columna]=f(a)
    return df

# Relación condicional

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

# Condiciones predefinidas
def elegir_condicion(cond):
    """
    Convierte una cadena descriptiva en una función condicional sobre una fila de valores.

    Args:
        cond (str): Condición en formato string.

    Returns:
        Callable: Función booleana que evalúa una fila (lista de valores).
    """
    if cond[1:7]=='Menor=':
        ind = cond[0]
        indice = int(ind)
        valor = cond[7:]
        num = float(valor)
        return partial(menorI,valor=num,indice=indice)
    
    elif cond[1:7]=='Mayor=':
        ind = cond[0]
        indice = int(ind)
        valor = cond[7:]
        num = float(valor)
        return partial(mayorI,valor=num,indice=indice)
    
    elif cond[1:9]=='Menores=':
        valor1 = cond[0]
        valor2 = cond[9]
        num1 = int(valor1)
        num2 = int(valor2)
        return partial(menoresI,valor1=num1,valor2=num2)
    
    elif cond[1:9]=='Mayores=':
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
    
def objetivo_condicional(df_caract,columna,cond,func):
    """
    Genera una variable exógena aplicando condiciones personalizadas sobre las variables de entrada.

    Args:
        df_caract (pd.DataFrame): DataFrame con variables independientes.
        columna (str): Nombre de la nueva columna a generar.
        cond (list): Lista de funciones booleanas. Cada una evalúa si se debe aplicar una transformación.
        func (list): Lista de funciones de transformación asociadas a cada condición. Deben tener la misma longitud que `cond`.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna con los valores generados.
    """
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

# Análisis de Componentes Principales
def objetivo_PCA(df,columna):
    """
    Aplica Análisis de Componentes Principales (PCA) para generar una variable exógena.

    Args:
        df (pd.DataFrame): DataFrame con las variables originales (puede ser multivariado).
        columna (str): Nombre de la nueva columna donde se almacenará el primer componente principal.

    Returns:
        pd.DataFrame: Copia del DataFrame original con una nueva columna basada en el primer componente principal.
    """
    # Estandarización
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # Paso 2: Aplicar PCA
    pca = PCA(n_components=1)  # Queremos sólo el primer componente principal
    df[columna] = pca.fit_transform(df_scaled)
    return df
    
# Matriz de correlación
def objetivo_correlacion(df,columna):
    """
    Genera una variable exógena como combinación lineal e interactiva de las columnas originales, 
    utilizando como pesos la matriz de correlaciones entre las variables.

    Args:
        df (pd.DataFrame): DataFrame de características multivariadas.
        columna (str): Nombre de la nueva columna generada.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna construida a partir de las correlaciones entre las variables.
    """
    return interaccion(df,0,df.corr().values,columna)

# Matriz de covarianza
def objetivo_covarianza(df,columna):
    """
    Genera una variable exógena utilizando la matriz de covarianza del DataFrame, 
    aplicando un modelo de interacción cuadrática.

    Args:
        df (pd.DataFrame): DataFrame de características multivariadas.
        columna (str): Nombre de la nueva columna que será añadida.

    Returns:
        pd.DataFrame: DataFrame con la nueva variable exógena construida a partir de la matriz de covarianza.
    """
    return interaccion(df,0,df.cov().values,columna)

