from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from detecta import detect_cusum
import copy
import warnings
from collections import defaultdict
import inspect
import re
from abc import ABCMeta, abstractmethod
from mitten import mcusum, hotelling_t2, pc_mewma,interpret_multivariate_signal,apply_mewma

# KOLMOGOROV-SMIRNOV TEST
def detect_dataset_drift_ks(df1,df2,threshold):
    """
    Detecta si existe drift (cambio en la distribución) entre dos conjuntos de datos
    utilizando la prueba estadística de Kolmogorov-Smirnov para cada columna numérica.

    Args:
        df1 (pd.DataFrame): Primer conjunto de datos (referencia).
        df2 (pd.DataFrame): Segundo conjunto de datos (nuevo o de producción).
        threshold (float): Umbral de significancia para la prueba KS. Un valor típico es 0.05.

    Returns:
        tuple:
            - deteccion (str): 'Detectado' si se detecta drift en alguna columna, 'No detectado' si no.
            - reporte (dict): Diccionario con las columnas como claves, y como valores otro diccionario con:
                - 'p_valor': valor p de la prueba KS para esa columna.
                - 'drift': booleano indicando si se ha detectado drift (True si p < threshold).
    """
  
    deteccion = 'No detectado'
    reporte={}
  
    for x in df1.columns:
        
        d1 = df1[x]
        d2 = df2[x]
        ks = ks_2samp(d1,d2)

        if ks.pvalue>=threshold:
            drift=False
        else:
            deteccion= 'Detectado'
            drift=True

        reporte.update({x:{
        "p_valor":float(ks.pvalue),
        "drift":drift}}) 
        
    return (deteccion,reporte)

# DIVERGENCIA DE JENSEN-SHANNON

def preparacion(df1, df2, col_name, num_bins=10):
    """
    Divide los datos continuos de una columna en intervalos (bins) comunes para dos dataframes
    y calcula la distribución relativa de frecuencias en esos intervalos.

    Args:
        df1 (pd.DataFrame): Primer conjunto de datos (referencia).
        df2 (pd.DataFrame): Segundo conjunto de datos (nuevo o de producción).
        col_name (str): Nombre de la columna a discretizar.
        num_bins (int, opcional): Número de intervalos a crear. Por defecto es 10.

    Returns:
        tuple:
            - df1_bins (pd.Series): Distribución relativa de frecuencias por bin en df1.
            - df2_bins (pd.Series): Distribución relativa de frecuencias por bin en df2.
    """

    # Determinar los límites mínimos y máximos conjuntos
    maxi = max(df1[col_name].max(), df2[col_name].max())
    mini = min(df1[col_name].min(), df2[col_name].min())

    # Crear los intervalos (bins)
    bins = np.linspace(mini, maxi, num_bins + 1)

    # Aplicar los bins a ambos dataframes
    df1_copy = df1.copy()
    df1_copy['bin'] = pd.cut(df1[col_name], bins=bins, include_lowest=True)

    df2_copy = df2.copy()
    df2_copy['bin'] = pd.cut(df2[col_name], bins=bins, include_lowest=True)

    # Calcular distribución relativa por bin
    df1_bins = df1_copy.groupby('bin')[col_name].count() / len(df1)
    df2_bins = df2_copy.groupby('bin')[col_name].count() / len(df2)

    return df1_bins, df2_bins

def jensenshannon(p, q, base=None, axis=0, keepdims=False):
    """
    Calcula la divergencia de Jensen-Shannon entre dos distribuciones de probabilidad.

    Args:
        p (array-like): Primera distribución de probabilidad.
        q (array-like): Segunda distribución de probabilidad.
        base (float, opcional): Base del logaritmo para normalizar el resultado.
        axis (int, opcional): Eje a lo largo del cual se realiza la operación.
        keepdims (bool, opcional): Si mantener las dimensiones originales.

    Returns:
        float: Divergencia de Jensen-Shannon entre p y q (valor entre 0 y 1).
    """
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum
    if base is not None:
        js /= np.log(base)
    return js / 2.0

def detect_dataset_drift_js(df1, df2, threshold, num_bins=10):
    """
    Detecta si existe drift entre dos conjuntos de datos utilizando la divergencia de Jensen-Shannon
    sobre distribuciones discretizadas de cada columna.

    Args:
        df1 (pd.DataFrame): Primer conjunto de datos (referencia).
        df2 (pd.DataFrame): Segundo conjunto de datos (nuevo o de producción).
        threshold (float): Umbral de divergencia a partir del cual se considera que hay drift.
        num_bins (int, opcional): Número de bins para discretización de las columnas continuas. Por defecto es 10.

    Returns:
        tuple:
            - deteccion (str): 'Detectado' si se detecta drift en alguna columna, 'No detectado' si no.
            - report (dict): Diccionario con claves como nombres de columnas, y valores con:
                - 'Jensen-Shannon': valor de la divergencia JS.
                - 'drift': booleano indicando si se ha detectado drift (True si JS >= threshold).
    """

    deteccion = 'No detectado'
    report = {}

    for column in df1.columns:
        # Obtener las distribuciones discretizadas
        d1, d2 = preparacion(df1, df2, column, num_bins)

        # Calcular la divergencia de Jensen-Shannon
        js = jensenshannon(d1, d2)

        if js < threshold:
            drift = False
        else:
            deteccion = 'Detectado'
            drift = True

        report.update({
            column: {
                "Jensen-Shannon": float(js),
                "drift": drift
            }
        })

    return (deteccion, report)

# POPULATION STABILITY INDEX

def population_stability_index(dev_data, val_data,col_name, num_bins=10):
    """
        Calcula el Population Stability Index (PSI) para una variable continua.

        El PSI mide el cambio en la distribución de una variable entre dos muestras (por ejemplo, datos de entrenamiento y de validación).
        Una mayor divergencia indica mayor probabilidad de que haya *dataset drift*.

        Args:
            dev_data (pd.DataFrame): Conjunto de datos de desarrollo (por ejemplo, entrenamiento).
            val_data (pd.DataFrame): Conjunto de datos de validación o producción.
            col_name (str): Nombre de la columna sobre la que se calculará el PSI.
            num_bins (int, opcional): Número de intervalos (bins) en los que dividir los datos. Por defecto es 10.

        Returns:
            float: Valor del PSI calculado. Valores típicos:
                - < 0.1: sin drift significativo
                - 0.1 - 0.25: posible drift
                - > 0.25: drift severo
    """
    maxi = max(dev_data[col_name].max(),val_data[col_name].max())
    mini = min(dev_data[col_name].min(),val_data[col_name].min())
   
    bins = np.linspace(mini, maxi, num_bins + 1)
    
    dev_data_copy = dev_data.copy()
    dev_data_copy['bin'] = pd.cut(dev_data[col_name], bins=bins, include_lowest=True)
    
    val_data_copy=val_data.copy()
    val_data_copy['bin'] = pd.cut(val_data[col_name], bins=bins, include_lowest=True)

    dev_group = dev_data_copy.groupby('bin')[col_name].count().reset_index(name='dev_count')
    val_group = val_data_copy.groupby('bin')[col_name].count().reset_index(name='val_count')

    merged_counts = dev_group.merge(val_group, on='bin', how='left')
    
    small_constant = 1e-10
    merged_counts['dev_pct'] = (merged_counts['dev_count'] / len(dev_data)) + small_constant
    merged_counts['val_pct'] = (merged_counts['val_count'] / len(val_data)) + small_constant
    merged_counts['psi'] = (merged_counts['val_pct'] - merged_counts['dev_pct']) * np.log(merged_counts['val_pct'] / merged_counts['dev_pct'])

    return merged_counts['psi'].sum()

def detect_dataset_drift_psi(df1,df2,threshold,num_bins=10):
    """
    Detecta si existe *drift* entre dos conjuntos de datos usando el índice PSI por columna.

    Args:
        df1 (pd.DataFrame): Primer conjunto de datos (referencia).
        df2 (pd.DataFrame): Segundo conjunto de datos (actual).
        threshold (float): Umbral a partir del cual se considera que existe *drift*. Umbral típico 0,25.
        num_bins (int, opcional): Número de bins usados para el cálculo del PSI. Por defecto 10.

    Returns:
        tuple:
            - deteccion (str): 'Detectado' si se detecta *drift* en alguna columna, 'No detectado' en caso contrario.
            - report (dict): Diccionario con nombres de columnas como claves y valores que contienen:
                - "PSI": valor del índice PSI calculado.
                - "drift": booleano indicando si se detecta *drift* (True si PSI ≥ threshold).
    """
    deteccion = 'No detectado'
    report = {}
    for column in df1.columns:
            
        psiC = population_stability_index(df1,df2,column,num_bins)

        if threshold>psiC:
            drift=False
        else:
            deteccion = 'Detectado'
            drift=True

        report.update({column:{
        "PSI":float(psiC),
        "drift":drift}}) 
        
    return (deteccion,report)

def sub_psi(e_perc, a_perc):
    """
    Calcula la contribución al PSI entre una fracción esperada y una observada, manejando ceros.

    Args:
        e_perc (float): Porcentaje esperado.
        a_perc (float): Porcentaje actual.

    Returns:
        float: Contribución al PSI de ese intervalo.
    """ 
    if a_perc == 0:
         a_perc = 0.0001
    if e_perc == 0:
         e_perc = 0.0001

    value = (e_perc - a_perc) * np.log(e_perc / a_perc)
    
    return(value)

def psi_quantiles(expected_array, actual_array, buckets):
    """
    Calcula el PSI entre dos distribuciones utilizando cuantiles como umbrales.

    Args:
        expected_array (np.ndarray): Array con los valores esperados (referencia).
        actual_array (np.ndarray): Array con los valores actuales (observados).
        buckets (int): Número de cuantiles para dividir la distribución.

    Returns:
        float: Valor del PSI entre ambas distribuciones.
    """
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
    
    psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))

    return psi_value

def calculate_psi_quantiles(expected, actual, buckets=10):
    """
    Calcula el PSI para cada variable numérica de dos DataFrames usando cuantiles.

    Args:
        expected (pd.DataFrame): DataFrame de referencia (por ejemplo, entrenamiento).
        actual (pd.DataFrame): DataFrame actual (por ejemplo, validación o producción).
        buckets (int, opcional): Número de cuantiles. Por defecto 10.

    Returns:
        np.ndarray: Array con los valores de PSI por columna.
    """
    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1])
        
    for i in range (0 , len(psi_values) ):
        if len(psi_values) == 1:
            psi_values = psi_quantiles(expected, actual, buckets)
        else:
            psi_values[i] = psi_quantiles(expected.iloc[:,i].values, actual.iloc[:,i].values, buckets)
    
    return psi_values

def detect_dataset_drift_psi_quantiles(df1,df2,threshold,num_quantiles=10):
    """
    Detecta *drift* en variables continuas mediante PSI usando cuantiles como método de discretización.

    Args:
        df1 (pd.DataFrame): Conjunto de datos de referencia.
        df2 (pd.DataFrame): Conjunto de datos actual.
        threshold (float): Umbral a partir del cual se considera que existe *drift*.
        num_quantiles (int, opcional): Número de cuantiles para dividir la distribución. Por defecto 10.

    Returns:
        tuple:
            - deteccion (str): 'Detectado' si se detecta drift en alguna columna, 'No detectado' si no.
            - report (dict): Diccionario con el nombre de cada columna y su respectivo resultado:
                - "PSI": valor calculado del índice de estabilidad poblacional.
                - "drift": booleano que indica si se ha detectado drift en la columna (True si PSI ≥ threshold).
    """
    deteccion = 'No detectado'
    report={}
    psi_values = np.empty(df1.shape[1])
    psi_values = calculate_psi_quantiles(df1,df2,num_quantiles)
    i=0
    for column in df1.columns:
        
        if len(df1.columns)>1:
            psiC = psi_values[i]
        else:
            psiC = psi_values
        if threshold>psiC:
            drift=False
        else:
            deteccion = 'Detectado'
            drift=True
        report.update({column:{
        "PSI":float(psiC),
        "drift":drift}})         
        i=i+1

    return (deteccion,report) 

# CUSUM 

def detect_dataset_drift_cusum(df,threshold,drift=0.02):
    """
    Detecta cambios abruptos (drift) en variables univariadas usando el algoritmo CUSUM (Cumulative Sum).

    Esta función recorre todas las columnas del DataFrame y aplica el algoritmo CUSUM para detectar desviaciones significativas
    en la secuencia temporal o en la distribución de cada variable. Se utiliza principalmente para detectar *concept drift* o 
    *dataset drift* en datos secuenciales o flujos de datos.

    Args:
        df (pd.DataFrame): Conjunto de datos a analizar. Cada columna representa una variable.
        threshold (float): Umbral de amplitud que determina cuándo se considera que ha habido un cambio.
        drift (float, opcional): Término de deriva que evita detectar cambios en ausencia de cambio real (por defecto 0.02).

    Returns:
        tuple:
            - deteccion (str): 'Detectado' si se ha detectado drift en alguna columna, 'No detectado' en caso contrario.
            - report (dict): Diccionario con nombres de columnas como claves y valores que contienen:
                - "drift" (bool): True si se ha detectado drift en la columna, False si no.
    """
    deteccion = 'No detectado'
    report = {}
    i = 0
    for column in df.columns:
        
        _, tai, _, _ = detect_cusum(df.iloc[:,i].values, threshold, drift, True, True)
        if len(tai)==0:
            drift=False
        else:
            deteccion='Detectado'
            drift=True

        report.update({column:{
            "drift":drift}}) 
        i=i+1

    return (deteccion,report)

# Drift Detection Method

# Código obtenido de https://github.com/online-ml/river/tree/main/river
__version__ = '0.5.3'

_DEFAULT_TAGS = {
    'non_deterministic': False,
    'requires_positive_data': False,
    'X_types': ['2darray'],
    'poor_score': False,
    'no_validation': False,
    'multioutput': False,
    "allow_nan": False,
    'stateless': False,
    'multilabel': False,
    '_skip_test': False,
    'multioutput_only': False}


def clone(estimator, safe=True):
    """Constructs a new estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.

    Parameters
    ----------
    estimator : estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned

    safe : boolean, optional
        If safe is false, clone will fall back to a deep copy on objects
        that are not estimators.

    Notes
    -----
    Taken from sklearn for compatibility.
    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params') or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a valid estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'either does not set or modifies parameter %s' %
                               (estimator, name))
    return new_object

def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print

    offset : int
        The offset in characters to add at the begin of each line.

    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr

    Notes
    -----
    Taken from sklearn for compatibility.
    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(line.rstrip(' ') for line in lines.split('\n'))
    return lines


def _update_if_consistent(dict1, dict2):
    common_keys = set(dict1.keys()).intersection(dict2.keys())
    for key in common_keys:
        if dict1[key] != dict2[key]:
            raise TypeError("Inconsistent values for tag {}: {} != {}".format(
                key, dict1[key], dict2[key]
            ))
    dict1.update(dict2)
    return dict1

class BaseEstimator:
    """Base Estimator class for compatibility with scikit-learn.

    Notes
    -----
    * All estimators should specify all the parameters that can be set
      at the class level in their ``__init__`` as explicit keyword
      arguments (no ``*args`` or ``**kwargs``).
    * Taken from sklearn for compatibility.
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-multiflow estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self, N_CHAR_MAX=700):
        # N_CHAR_MAX is the (approximate) maximum number of non-blank
        # characters to render. We pass it as an optional parameter to ease
        # the tests.

        from ..utils._pprint import _EstimatorPrettyPrinter

        N_MAX_ELEMENTS_TO_SHOW = 30  # number of elements to show in sequences

        # use ellipsis for sequences with a lot of elements
        pp = _EstimatorPrettyPrinter(
            compact=True, indent=1, indent_at_name=True,
            n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW)

        repr_ = pp.pformat(self)

        # Use bruteforce ellipsis when there are a lot of non-blank characters
        n_nonblank = len(''.join(repr_.split()))
        if n_nonblank > N_CHAR_MAX:
            lim = N_CHAR_MAX // 2  # apprx number of chars to keep on both ends
            regex = r'^(\s*\S){%d}' % lim
            # The regex '^(\s*\S){%d}' % n
            # matches from the start of the string until the nth non-blank
            # character:
            # - ^ matches the start of string
            # - (pattern){n} matches n repetitions of pattern
            # - \s*\S matches a non-blank char following zero or more blanks
            left_lim = re.match(regex, repr_).end()
            right_lim = re.match(regex, repr_[::-1]).end()

            if '\n' in repr_[left_lim:-right_lim]:
                # The left side and right side aren't on the same line.
                # To avoid weird cuts, e.g.:
                # categoric...ore',
                # we need to start the right side with an appropriate newline
                # character so that it renders properly as:
                # categoric...
                # handle_unknown='ignore',
                # so we add [^\n]*\n which matches until the next \n
                regex += r'[^\n]*\n'
                right_lim = re.match(regex, repr_[::-1]).end()

            ellipsis = '...'
            if left_lim + len(ellipsis) < len(repr_) - right_lim:
                # Only add ellipsis if it results in a shorter repr
                repr_ = repr_[:left_lim] + '...' + repr_[-right_lim:]

        return repr_

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith('skmultiflow.'):
            return dict(state.items(), _skmultiflow_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith('skmultiflow.'):
            pickle_version = state.pop("_skmultiflow_version", "pre-0.18")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

    def _get_tags(self):
        collected_tags = {}
        for base_class in inspect.getmro(self.__class__):
            if (hasattr(base_class, '_more_tags') and base_class != self.__class__):
                more_tags = base_class._more_tags(self)
                collected_tags = _update_if_consistent(collected_tags,
                                                       more_tags)
        if hasattr(self, '_more_tags'):
            more_tags = self._more_tags()
            collected_tags = _update_if_consistent(collected_tags, more_tags)
        tags = _DEFAULT_TAGS.copy()
        tags.update(collected_tags)
        return tags

class BaseSKMObject(BaseEstimator):
    """Base class for most objects in scikit-multiflow

        Notes
        -----
        This class provides additional functionality not available in the base estimator
        from scikit-learn
    """
    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
        self

        """
        # non-optimized default implementation; override if a better
        # method is possible for a given object
        command = ''.join([line.strip() for line in self.__repr__().split()])
        command = command.replace(str(self.__class__.__name__), 'self.__init__')
        exec(command)

    def get_info(self):
        """ Collects and returns the information about the configuration of the estimator

        Returns
        -------
        string
            Configuration of the estimator.
        """
        return self.__repr__()

class ClassifierMixin(metaclass=ABCMeta):
    """Mixin class for all classifiers in scikit-multiflow."""
    _estimator_type = "classifier"

    def fit(self, X, y, classes=None, sample_weight=None):
        """ Fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Contains all possible/known class labels. Usage varies depending
            on the learning method.

        sample_weight: numpy.ndarray, optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
        self

        """
        # non-optimized default implementation; override if a better
        # method is possible for a given classifier
        self.partial_fit(X, y, classes=classes, sample_weight=sample_weight)

        return self

    @abstractmethod
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. Usage varies depending
            on the learning method.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
        self

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the class labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated
        with the X entry of the same index. And where the list in index [i] contains
        len(self.target_values) elements, each of which represents the probability that
        the i-th sample of X belongs to a certain class-label.

        """
        raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

class RegressorMixin(metaclass=ABCMeta):
    """Mixin class for all regression estimators in scikit-multiflow."""
    _estimator_type = "regressor"

    def fit(self, X, y, sample_weight=None):
        """ Fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the target values of all samples in X.

        sample_weight: numpy.ndarray, optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies
            depending on the learning method.

        Returns
        -------
        self

        """
        # non-optimized default implementation; override if a better
        # method is possible for a given regressor
        self.partial_fit(X, y, sample_weight=sample_weight)

        return self

    @abstractmethod
    def partial_fit(self, X, y, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the target values of all samples in X.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies
            depending on the learning method.

        Returns
        -------
        self

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """ Predict target values for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the target values for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        """ Estimates the probability for probabilistic/bayesian regressors

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the probabilities for.

        Returns
        -------
        numpy.ndarray

        """
        raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.

        Notes
        -----
        The R2 score used when calling ``score`` on a regressor will use
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with `metrics.r2_score`. This will influence the ``score`` method of
        all the multioutput regressors (except for
        `multioutput.MultiOutputRegressor`). To specify the default value
        manually and avoid the warning, please either call `metrics.r2_score`
        directly or make a custom scorer with `metrics.make_scorer` (the
        built-in scorer ``'r2'`` uses ``multioutput='uniform_average'``).
        """

        from sklearn.metrics import r2_score
        from sklearn.metrics.regression import _check_reg_targets
        y_pred = self.predict(X)
        # XXX: Remove the check in 0.23
        y_type, _, _, _ = _check_reg_targets(y, y_pred, None)
        if y_type == 'continuous-multioutput':
            warnings.warn("The default value of multioutput (not exposed in "
                          "score method) will change from 'variance_weighted' "
                          "to 'uniform_average' in 0.23 to keep consistent "
                          "with 'metrics.r2_score'. To specify the default "
                          "value manually and avoid the warning, please "
                          "either call 'metrics.r2_score' directly or make a "
                          "custom scorer with 'metrics.make_scorer' (the "
                          "built-in scorer 'r2' uses "
                          "multioutput='uniform_average').", FutureWarning)
        return r2_score(y, y_pred, sample_weight=sample_weight,
                        multioutput='variance_weighted')

class MetaEstimatorMixin(object):
    """Mixin class for all meta estimators in scikit-multiflow."""
    _required_parameters = ["estimator"]

class MultiOutputMixin(object):
    """Mixin to mark estimators that support multioutput."""
    def _more_tags(self):
        return {'multioutput': True}

def is_classifier(estimator):
    """Returns True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"

def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "regressor"

class BaseDriftDetector(BaseSKMObject, metaclass=ABCMeta):
    """ Abstract Drift Detector
    
    Any drift detector class should follow this minimum structure in 
    order to allow interchangeability between all change detection 
    methods.
    
    Raises
    ------
    NotImplementedError. All child classes should implement the
    get_info function.
    
    """

    estimator_type = "drift_detector"

    def __init__(self):
        super().__init__()
        self.in_concept_change = None
        self.in_warning_zone = None
        self.estimation = None
        self.delay = None

    def reset(self):
        """ reset
        
        Resets the change detector parameters.
         
        """
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0
        self.delay = 0.0

    def detected_change(self):
        """ detected_change
        
        This function returns whether concept drift was detected or not.
        
        Returns
        -------
        bool
            Whether concept drift was detected or not.
        
        """
        return self.in_concept_change

    def detected_warning_zone(self):
        """ detected_warning_zone

        If the change detector supports the warning zone, this function will return 
        whether it's inside the warning zone or not.

        Returns
        -------
        bool
            Whether the change detector is in the warning zone or not.

        """
        return self.in_warning_zone

    def get_length_estimation(self):
        """ get_length_estimation
        
        Returns the length estimation.
        
        Returns
        -------
        int
            The length estimation
        
        """
        return self.estimation

    @abstractmethod
    def add_element(self, input_value):
        """ add_element
        
        Adds the relevant data from a sample into the change detector.
        
        Parameters
        ----------
        input_value: Not defined
            Whatever input value the change detector takes.
        
        Returns
        -------
        BaseDriftDetector
            self, optional
        
        """
        raise NotImplementedError

class DDM(BaseDriftDetector):
    """ Drift Detection Method.
    
    Parameters
    ----------
    min_num_instances: int (default=30)
        The minimum required number of analyzed samples so change can be 
        detected. This is used to avoid false detections during the early 
        moments of the detector, when the weight of one sample is important.

    warning_level: float (default=2.0)
        Warning Level

    out_control_level: float (default=3.0)
        Out-control Level

    Notes
    -----
    DDM (Drift Detection Method) [1]_ is a concept change detection method
    based on the PAC learning model premise, that the learner's error rate
    will decrease as the number of analysed samples increase, as long as the
    data distribution is stationary.

    If the algorithm detects an increase in the error rate, that surpasses
    a calculated threshold, either change is detected or the algorithm will
    warn the user that change may occur in the near future, which is called
    the warning zone.

    The detection threshold is calculated in function of two statistics,
    obtained when `(pi + si)` is minimum:

    * :math:`p_{min}`: The minimum recorded error rate.
    * `s_{min}`: The minimum recorded standard deviation.

    At instant :math:`i`, the detection algorithm uses:

    * :math:`p_i`: The error rate at instant i.
    * :math:`s_i`: The standard deviation at instant i.

    The conditions for entering the warning zone and detecting change are
    as follows:

    * if :math:`p_i + s_i \geq p_{min} + 2 * s_{min}` -> Warning zone
    * if :math:`p_i + s_i \geq p_{min} + 3 * s_{min}` -> Change detected



    """

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0):
        super().__init__()
        self.sample_count = None
        self.miss_prob = None
        self.miss_std = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")

    def add_element(self, prediction):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).
        
        Notes
        -----
        After calling this method, to verify if change was detected or if  
        the learner is in the warning zone, one should call the super method 
        detected_change, which returns True if concept drift was detected and
        False otherwise.
        
        """
        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
            self.in_concept_change = True

        elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False

def detect_dataset_drift_ddm(model,x,y,min_inst=100,warning=2,threshold=3):
    """
    Detecta *concept drift* en una secuencia de predicciones utilizando el algoritmo DDM (Drift Detection Method).

    Este método evalúa el rendimiento de un modelo de clasificación sobre una secuencia de instancias (por ejemplo, en un flujo de datos)
    y determina si hay un deterioro estadísticamente significativo en la tasa de error, lo que indicaría la presencia de *drift* en los datos.

    Args:
        model (sklearn-like estimator): Modelo previamente entrenado que implementa el método `.predict()`.
        x (np.ndarray o pd.DataFrame): Conjunto de características de entrada.
        y (np.ndarray o pd.Series): Etiquetas verdaderas correspondientes a las instancias de `x`.
        min_inst (int, opcional): Número mínimo de instancias necesarias antes de comenzar a evaluar el drift (por defecto 100).
        warning (float, opcional): Número de desviaciones estándar que indica la entrada en zona de advertencia (por defecto 2).
        threshold (float, opcional): Número de desviaciones estándar que indica que se ha detectado un cambio (por defecto 3).

    Returns:
        bool:
            - `True` si se ha detectado drift en algún momento de la secuencia.
            - `False` si no se ha detectado ningún cambio significativo.
    """
    drift = False
    ddm = DDM (min_inst,warning,threshold)
    y_pred=model.predict(x)
    
    for i in range(y.shape[0]):
        
        y_true = y[i]
        
        if y_pred[i]==y_true:
            ddm.add_element(0)
        else:
            ddm.add_element(1)
        
        if ddm.detected_warning_zone():
            print('Warning zone has been detected in data of index: ' + str(i))
            
        if ddm.detected_change():
            print('Change has been detected in data of index: ' + str(i))
            drift= True 
            
    return drift 

# Page-Hinkley 

# Código obtenido de: https://github.com/online-ml/river/tree/main/river/drift

class PageHinkley(BaseDriftDetector):
    """ Page-Hinkley method for concept drift detection.

    Notes
    -----
    This change detection method works by computing the observed 
    values and their mean up to the current moment. Page-Hinkley
    won't output warning zone warnings, only change detections. 
    The method works by means of the Page-Hinkley test [1]_. In general
    lines it will detect a concept drift if the observed mean at 
    some instant is greater then a threshold value lambda.

    References
    ----------
    .. [1] E. S. Page. 1954. Continuous Inspection Schemes.
       Biometrika 41, 1/2 (1954), 100–115.
    
    Parameters
    ----------
    min_instances: int (default=30)
        The minimum number of instances before detecting change.
    delta: float (default=0.005)
        The delta factor for the Page Hinkley test.
    threshold: int (default=50)
        The change detection threshold (lambda).
    alpha: float (default=1 - 0.0001)
        The forgetting factor, used to weight the observed value 
        and the mean.
    
    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection import PageHinkley
    >>> ph = PageHinkley()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 2000
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>> # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
    >>> for i in range(2000):
    ...     ph.add_element(data_stream[i])
    ...     if ph.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    
    """
    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def add_element(self, x):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        x: numeric value
            The observed value, from which we want to detect the
            concept change.
        
        Notes
        -----
        After calling this method, to verify if change was detected, one 
        should call the super method detected_change, which returns True 
        if concept drift was detected and False otherwise.
        
        """
        if self.in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (x - self.x_mean) / float(self.sample_count)
        self.sum = max(0., self.alpha * self.sum + (x - self.x_mean - self.delta))

        self.sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        if self.sample_count < self.min_instances:
            return None

        if self.sum > self.threshold:
            self.in_concept_change = True

def detect_dataset_drift_ph(df,min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
    """
    Detecta *concept drift* en un conjunto de datos univariado o multivariado utilizando el método Page-Hinkley.

    Este algoritmo detecta cambios significativos en la media de una secuencia de datos. Es útil para entornos de flujo
    de datos donde es necesario monitorizar variables y detectar desviaciones del comportamiento esperado.

    Args:
        df (pd.DataFrame): Conjunto de datos con una o más columnas, donde se evaluará la presencia de drift en cada variable.
        min_instances (int, opcional): Número mínimo de observaciones requeridas antes de comenzar a evaluar cambios (por defecto 30).
        delta (float, opcional): Tolerancia al cambio. Es la magnitud mínima de desviación considerada significativa (por defecto 0.005).
        threshold (float, opcional): Umbral a partir del cual se considera que ha ocurrido un cambio (por defecto 50).
        alpha (float, opcional): Factor de olvido o nivel de confianza, usado para suavizar la actualización del modelo (por defecto 0.9999).

    Returns:
        tuple:
            - deteccion (str): 'Detectado' si se ha identificado drift en alguna de las columnas, 'No detectado' en caso contrario.
            - reporte (dict): Diccionario con el nombre de cada columna como clave y un valor booleano indicando si se detectó drift.
    """
    deteccion = 'No detectado'
    reporte={}
    
    for x in df.columns:
        
        d1=df[x]
        drift=False 
        pageH = PageHinkley(min_instances,delta,threshold,alpha)
        for i in range(len(d1)):
            pageH.add_element(d1[i])
            if pageH.detected_change():
                print('Change has been detected in data: ' + str(d1[i]) + ' - of index: ' + str(i))
                deteccion = 'Detectado'
                drift = True
        reporte.update({x: {"drift_status": drift}})
        
    return (deteccion,reporte)

# MCUSUM

def detect_dataset_drift_mcusum(df, min_inst=100, lambd=0.5):
    """
    Detecta deriva en un conjunto de datos multivariado usando el método MCUSUM.

    Args:
        df (pd.DataFrame): Conjunto de datos multivariado a evaluar.
        min_inst (int, opcional): Número de instancias iniciales consideradas en control. Por defecto 100.
        lambd (float, opcional): Parámetro de tolerancia (slack) del MCUSUM. Controla la sensibilidad. Por defecto 0.5.

    Returns:
        drift (bool): True si se detecta deriva en el conjunto de datos, False en caso contrario.
    """
    # Ejecuta MCUSUM para obtener los valores del estadístico y el límite de control
    y_vals, ucl = mcusum(df, min_inst, lambd)

    # Se detecta deriva si algún valor excede el límite de control
    drift = False
    if max(y_vals) > ucl:
        drift = True

    return drift

# MEWMA

def detect_dataset_drift_mewma(df, min_inst=100, lambd=0.1, alpha=0):
    """
    Detecta deriva en un conjunto de datos multivariado utilizando el método MEWMA (Multivariate Exponentially Weighted Moving Average).

    Args:
        df (pd.DataFrame): Conjunto de datos multivariado a evaluar.
        min_inst (int, opcional): Número de observaciones iniciales consideradas en control. Por defecto 100.
        lambd (float, opcional): Parámetro de suavizado entre 0 y 1. Valores más bajos dan más peso a observaciones pasadas. Por defecto 0.1.
        alpha (float, opcional): Porcentaje de falsos positivos permitido. Se usa para calcular el límite superior de control (UCL). Por defecto 0.

    Returns:
        drift (bool): True si se detecta deriva en los datos, False si no se detecta.
    """
    # Aplica el método MEWMA al conjunto de datos
    mewma_stats, ucl = apply_mewma(df, min_inst, alpha, lambd)

    # Determina si se ha producido deriva al comparar con el límite superior de control
    drift = False
    if max(mewma_stats) > ucl:
        drift = True

    return drift

def detect_dataset_drift_pC_mewma(df, princ_comp, min_inst=100, lambd=0.1, alpha=0):
    """
    Detecta deriva en un conjunto de datos utilizando MEWMA aplicado a componentes principales.

    Args:
        df (pd.DataFrame): Conjunto de datos multivariado a analizar.
        princ_comp (int): Número de componentes principales a retener antes de aplicar MEWMA.
        min_inst (int, opcional): Número de instancias iniciales consideradas en control. Por defecto 100.
        lambd (float, opcional): Parámetro de suavizado (entre 0 y 1). Valores más bajos dan más peso a observaciones antiguas. Por defecto 0.1.
        alpha (float, opcional): Porcentaje de falsos positivos permitido. Se usa para calcular el límite superior de control (UCL). Por defecto 0.

    Returns:
        drift (bool): True si se detecta deriva en el conjunto de datos, False en caso contrario.
    """
    # Aplica MEWMA sobre los datos proyectados en componentes principales
    mewma_stats, ucl = pc_mewma(df, min_inst, princ_comp, alpha, lambd)

    # Se detecta deriva si el valor máximo del estadístico excede el UCL
    drift = False
    if max(mewma_stats) > ucl:
        drift = True

    return drift

# HOTELLING

def detect_dataset_drift_hotelling(df, min_inst=100, alpha=0):
    """
    Detecta deriva en un conjunto de datos multivariado utilizando el estadístico Hotelling T².

    Args:
        df (pd.DataFrame): Conjunto de datos multivariado a evaluar.
        min_inst (int, opcional): Número de observaciones iniciales consideradas en control. Por defecto 100.
        alpha (float, opcional): Porcentaje de falsos positivos permitido, usado para calcular el límite superior de control (UCL). Por defecto 0.

    Returns:
        drift (bool): True si se detecta deriva en los datos, False en caso contrario.
    """
    # Calcula el estadístico Hotelling T² y el límite de control (UCL)
    t2_values, ucl = hotelling_t2(df, min_inst, alpha)

    # Se considera que hay deriva si el valor máximo excede el UCL
    drift = False
    if max(t2_values) > ucl:
        drift = True

    return drift