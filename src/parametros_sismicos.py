"""
LIBRERÍA PARA CALCULAR LAS CARACTERÍSTICAS SÍSMICAS

"""

#######################################################
# Función para asignar la zona sismogénica al Dataframe
#######################################################

import geopandas as gpd
import pandas as pd

def asignar_zona_sismogenica(gdf, shp_path, col_geom="geometry", col_id="ID"):
    """
    Asigna zona sismogénica a cada evento en un GeoDataFrame.
    
    Parámetros
    ----------
    gdf : GeoDataFrame
        DataFrame de eventos con geometría (lon/lat).
    shp_path : str
        Ruta al shapefile de las zonas sismogénicas.
    col_geom : str, opcional
        Nombre de la columna de geometría en gdf (default='geometry').
    col_id : str, opcional
        Nombre de la columna en el shapefile con el identificador de zona.
    
    Devuelve
    --------
    GeoDataFrame con nueva columna 'Zona_ID', sin la columna ID del shapefile.
    """
    
    # Leer shapefile
    zonas = gpd.read_file(shp_path)
    
    # Asegurar que el GeoDataFrame de eventos está en EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Reproyectar las zonas al mismo CRS
    zonas = zonas.to_crs(gdf.crs)
    
    # Spatial join
    gdf_out = gpd.sjoin(gdf, zonas[[col_id, "geometry"]], how="left", predicate="within")
    
    # Crear Zona_ID y eliminar columna ID del shapefile
    gdf_out["Zona_ID"] = gdf_out[col_id]
    gdf_out = gdf_out.drop(columns=[col_id, "index_right"], errors="ignore")
    
    return gdf_out




#######################################################
# Función para buscar la zona sismógenica correspondiente a un evento
#######################################################  

from shapely.geometry import Point

def buscar_zona_sismogenica(lat, lon, zonas, columna_id="ID"):
    """
    Devuelve el número de la zona sismogénica a la que pertenece un punto dado.
    
    Parámetros:
    -----------
    lat : float
        Latitud del punto
    lon : float
        Longitud del punto
    zonas : GeoDataFrame
        GeoDataFrame con las zonas sismogénicas
    columna_id : str
        Nombre de la columna que contiene el número de la zona sismogénica (ej. "ID")
    
    Retorna:
    --------
    int o None
        El número de la zona sismogénica, o None si el punto no está en ninguna.
    """
    
    # Aseguramos que zonas está en WGS84 (EPSG:4326)
    if zonas.crs is not None and zonas.crs.to_epsg() != 4326:
        zonas = zonas.to_crs(epsg=4326)
    
    # Crear el punto (shapely usa (lon, lat))
    punto = Point(lon, lat)
    
    # Buscar si cae en alguna zona
    zona = zonas[zonas.contains(punto)]
    
    if not zona.empty:
        return int(zona[columna_id].values[0])
    else:
        return None



#######################################################
# Función para calcular la magnitud de completitud del 
# catálogo sísmico
####################################################### 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def estimar_mc_optimo(df, col_mag="Mag_mbLgL", bins=0.1, threshold=0.9, min_events=20, plot=True):
    """
    Estima la magnitud de completitud (Mc) óptima para un catálogo sísmico,
    comparando MAXC y GFT, y devuelve directamente un único valor conservador.
    
    Parámetros:
        df : DataFrame con los datos
        col_mag : str, nombre de la columna de magnitud
        bins : float, tamaño de bin en histograma
        threshold : float, bondad de ajuste mínima aceptada en GFT (ej. 0.9 = 90%)
        min_events : int, mínimo de eventos para aplicar GFT
        plot : bool, si True, genera gráfico comparativo
        
    Retorna:
        float : Mc óptimo a usar en el catálogo
    """
    mags = df[col_mag].dropna().values
    mags = np.sort(mags)

    # ---------- Método 1: Máxima Curvatura (MAXC) ----------
    hist, bin_edges = np.histogram(mags, bins=np.arange(mags.min(), mags.max()+bins, bins))
    maxc_index = np.argmax(hist)
    mc_maxc = bin_edges[maxc_index]

    # ---------- Método 2: Goodness-of-Fit Test (GFT) ----------
    Mc_values = np.arange(mags.min(), mags.max(), bins)
    mc_gft = None
    
    for mc in Mc_values:
        subset = mags[mags >= mc]
        if len(subset) < min_events:
            continue
        
        counts = np.array([np.sum(subset >= m) for m in subset])
        slope, intercept, r_value, _, _ = linregress(subset, np.log10(counts))
        
        if r_value**2 >= threshold:
            mc_gft = mc
            break

    # ---------- Determinar Mc óptimo ----------
    if mc_gft is not None:
        mc_optimo = max(mc_maxc, mc_gft)  # conservador
    else:
        mc_optimo = mc_maxc

    # ---------- Gráfico ----------
    if plot:
        plt.figure(figsize=(8,5))
        plt.hist(mags, bins=np.arange(mags.min(), mags.max()+bins, bins), alpha=0.6, label="Frecuencia")
        plt.axvline(mc_maxc, color="blue", linestyle="--", label=f"Mc (MAXC) = {mc_maxc:.2f}")
        if mc_gft:
            plt.axvline(mc_gft, color="red", linestyle="--", label=f"Mc (GFT) = {mc_gft:.2f}")
        plt.axvline(mc_optimo, color="green", linestyle="-", label=f"Mc óptimo = {mc_optimo:.2f}")
        plt.xlabel("Magnitud")
        plt.ylabel("Frecuencia")
        plt.title("Estimación Mc: MAXC vs GFT")
        plt.legend()
        plt.show()

    return mc_optimo

# EJEMPLO DE USO
# mc_opt = estimar_mc_optimo(gdf_eq, col_mag="Mag_mbLgL", bins=0.1, threshold=0.9)
# print("Mc óptimo a usar en el catálogo:", mc_opt)
    

#######################################################
#
# CARACTERISTICAS PARAMETRICAS
#
####################################################### 

#######################################################
#
# 1. Parámetros de la ley de Gutenberg–Richter (GR)
#
# Función para obtener los parámetros a y b de la Ley de Gutemberg Richter
# en una ventana de N eventos
# Actualiza el DataFrame con el resultado
#
####################################################### 

import numpy as np
import pandas as pd
from scipy.stats import linregress

def gutemberg_richter_ventana(df, col_mag="Mag_mbLgL", n_eventos=50, min_mag=None):
    """
    Calcula los parámetros a y b de Gutenberg-Richter en ventanas móviles
    de N eventos previos al evento i.
    
    Devuelve el DataFrame con 4 nuevas columnas:
    - 'a_lsq', 'b_lsq' : regresión lineal
    - 'a_mlk', 'b_mlk' : máxima verosimilitud
    
    Parámetros
    ----------
    df : DataFrame con columna de magnitudes
    col_mag : str
        Nombre de la columna de magnitudes
    n_eventos : int
        Número de eventos previos a considerar
    min_mag : float o None
        Magnitud mínima a considerar. Si None, se toma la mínima de cada ventana.
    """
    df = df.copy()
    mags = df[col_mag].to_numpy()
    
    # Inicializar columnas
    a_lsq, b_lsq, a_mlk, b_mlk = [], [], [], []
    
    for i in range(len(mags)):
        # Ventana de N eventos previos
        if i < n_eventos:
            a_lsq.append(np.nan)
            b_lsq.append(np.nan)
            a_mlk.append(np.nan)
            b_mlk.append(np.nan)
            continue
        
        ventana = mags[i-n_eventos:i]
        
        # Magnitud mínima en la ventana
        Mmin = min_mag if min_mag is not None else ventana.min()
        subset = ventana[ventana >= Mmin]
        
        if len(subset) < 10:  # asegurar estadística mínima
            a_lsq.append(np.nan)
            b_lsq.append(np.nan)
            a_mlk.append(np.nan)
            b_mlk.append(np.nan)
            continue
        
        # --- Método regresión lineal ---
        mag_values = np.sort(subset)
        N = np.array([np.sum(mag_values >= m) for m in mag_values])
        logN = np.log10(N)
        slope, intercept, *_ = linregress(mag_values, logN)
        b_reg = -slope
        a_reg = intercept
        
        # --- Método máxima verosimilitud ---
        M_mean = subset.mean()
        b_mv = np.log10(np.exp(1)) / (M_mean - Mmin)
        a_mv = np.log10(len(subset)) + b_mv * Mmin
        
        a_lsq.append(a_reg)
        b_lsq.append(b_reg)
        a_mlk.append(a_mv)
        b_mlk.append(b_mv)
    
    # Guardar en el DataFrame
    df["a_lsq"] = a_lsq
    df["b_lsq"] = b_lsq
    df["a_mlk"] = a_mlk
    df["b_mlk"] = b_mlk
    
    return df

# EJEMPLO DE USO
#
# gdf_2002_ab = gutemberg_richter_ventana(gdf_2002, col_mag="Mag_mbLgL", n_eventos=50)
#
#

import numpy as np
import pandas as pd

#######################################################
#
# 2. Tasa media de liberación de energía de Benioff
#
# Función para obtener la tasa media de liberación de energía de Benioff
# Actualiza el DataFrame con el resultado
####################################################### 

import pandas as pd
import numpy as np

def calcular_tasa_benioff_ventana_rapida(gdf, col_mag="Mag_mbLgL", fecha_col="FechaHora", T=1.0):
    """
    Calcula la tasa media de liberación de energía de Benioff usando una ventana
    deslizante de duración T (años) de forma vectorizada.
    
    Parámetros
    ----------
    gdf : GeoDataFrame
        DataFrame con columnas de magnitud y fecha/hora.
    col_mag : str
        Nombre de la columna con la magnitud mbLg(L).
    fecha_col : str
        Nombre de la columna datetime.
    T : float
        Duración de la ventana en años.
    
    Devuelve
    -------
    gdf : GeoDataFrame
        Con nueva columna 'Tasa_Benioff' calculada evento a evento.
    """
    gdf = gdf.sort_values(by=fecha_col).copy()
    fechas = gdf[fecha_col].values
    mags = gdf[col_mag].values
    
    # Calcular energía de Benioff según tu fórmula
    energia = 10 ** (1.5 * mags + 4.8)
    
    # Convertir ventana T a días
    T_dias = T * 365.25
    
    # Preparar array para la tasa
    tasa = np.zeros(len(gdf))
    
    # Punteros de inicio y fin de la ventana
    start_idx = 0
    
    for i in range(len(gdf)):
        fecha_actual = fechas[i]
        # Mover el start_idx hasta que las fechas estén dentro de la ventana
        while start_idx < i and (fecha_actual - fechas[start_idx]).astype('timedelta64[D]').astype(float) > T_dias:
            start_idx += 1
        # Sumar energía de la ventana
        E_sum = energia[start_idx:i].sum()
        tasa[i] = np.sqrt(E_sum) / T if E_sum > 0 else 0.0
    
    gdf["Tasa_Benioff"] = tasa
    return gdf



    import pandas as pd


#######################################################
#
# 3. Tiempo de n eventos
#
# Función para obtener el tiempo correspondiente a n eventos
# Actualiza el DataFrame con el resultado
####################################################### 
    

import pandas as pd

def tiempo_desde_evento_n_rapido(gdf, n=10, fecha_col="FechaHora", unidad="days", nueva_col="Tiempo_desde_n"):
    """
    Calcula de manera vectorizada el tiempo transcurrido desde el evento n-ésimo anterior.
    
    Parámetros
    ----------
    gdf : DataFrame o GeoDataFrame
        DataFrame con columna datetime.
    n : int
        Número de eventos anteriores a considerar.
    fecha_col : str
        Nombre de la columna datetime.
    unidad : str
        Unidad de tiempo para el resultado ('days', 'hours', 'minutes', 'seconds').
    nueva_col : str
        Nombre de la nueva columna a crear.
    
    Retorna
    -------
    gdf : DataFrame/GeoDataFrame
        Con nueva columna 'Tiempo_desde_n' (o el nombre que indiques).
    """
    gdf = gdf.sort_values(by=fecha_col).copy()
    
    # Desplazar la columna de fechas n posiciones
    fechas_n_anterior = gdf[fecha_col].shift(n)
    
    # Diferencia de tiempo
    delta = gdf[fecha_col] - fechas_n_anterior
    
    # Convertir a la unidad deseada
    if unidad == "days":
        gdf[nueva_col] = delta.dt.total_seconds() / 86400
    elif unidad == "hours":
        gdf[nueva_col] = delta.dt.total_seconds() / 3600
    elif unidad == "minutes":
        gdf[nueva_col] = delta.dt.total_seconds() / 60
    elif unidad == "seconds":
        gdf[nueva_col] = delta.dt.total_seconds()
    else:
        raise ValueError("Unidad no soportada. Usa 'days', 'hours', 'minutes' o 'seconds'.")
    
    return gdf




#######################################################
#
# 4. Magnitud media en n eventos previos
#
# Función para obtener la magnitud media correspondiente a n eventos previos
# Actualiza el DataFrame con el resultado
####################################################### 
    
    import pandas as pd

def magnitud_media_eventos_previos(gdf, n=10, col_mag="Mag_mbLgL", nueva_col="Mag_media_n"):
    """
    Calcula la magnitud media de los n eventos anteriores para cada evento.
    
    Parámetros
    ----------
    gdf : DataFrame o GeoDataFrame
        DataFrame con la columna de magnitudes.
    n : int
        Número de eventos anteriores a considerar.
    col_mag : str
        Nombre de la columna de magnitud.
    nueva_col : str
        Nombre de la nueva columna a crear.
    
    Retorna
    -------
    gdf : DataFrame/GeoDataFrame
        Con nueva columna 'Mag_media_n' (o el nombre que indiques).
    """
    gdf = gdf.copy()
    
    # Magnitud media de los n eventos anteriores usando shift + rolling
    gdf[nueva_col] = gdf[col_mag].shift(1).rolling(window=n, min_periods=1).mean()
    
    return gdf


#######################################################
#
# 5. Valor medio y desviación estándar de la magnitud en los últimos 30 días
#
# Función para obtener la magnitud media y desviación estandar 
# correspondiente los eventos ocurridos en los n dias previos
# Actualiza el DataFrame con el resultado
####################################################### 

import pandas as pd

def estadisticas_magnitud_ventana(df, fecha_col="FechaHora", mag_col="Mag_mbLgL", dias=30):
    """
    Calcula la media y la desviación estándar de la magnitud en una ventana temporal de n días.
    
    Parámetros
    ----------
    df : DataFrame
        DataFrame con columna de fecha y magnitud.
    fecha_col : str
        Nombre de la columna de fecha.
    mag_col : str
        Nombre de la columna de magnitud.
    dias : int
        Ventana de tiempo en días (por defecto 30).
    
    Retorna
    -------
    df : DataFrame
        DataFrame con dos nuevas columnas:
        - 'Mag_media_ndias'
        - 'Mag_std_ndias'
    """
    df = df.copy()
    
    # Asegurarse de que la fecha esté en datetime y ordenar
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col)
    
    # Convertir la columna de fecha en índice para usar rolling con ventana temporal
    df = df.set_index(fecha_col)
    
    # Calcular media y desviación estándar sobre los últimos n días
    df[f"Mag_media_{dias}d"] = df[mag_col].rolling(f"{dias}D", min_periods=1).mean()
    df[f"Mag_std_{dias}d"]   = df[mag_col].rolling(f"{dias}D", min_periods=1).std()
    
    # Restaurar el índice original
    df = df.reset_index()
    
    return df


#######################################################
#
# GENERALIZACIÓN DEL PARÁMETRO 5
# Valor medio y desviación estándar de cualquier variable en los últimos 30 días
#
# Función para obtener la magnitud media y desviación estandar 
# correspondiente los eventos ocurridos en los n dias previos
# Actualiza el DataFrame con el resultado
####################################################### 

import pandas as pd

def estadisticas_ventana_temporal(df, fecha_col="FechaHora", valor_col="Mag_mbLgL", dias=30):
    """
    Calcula la media y la desviación estándar de una variable en una ventana temporal de n días.
    
    Parámetros
    ----------
    df : DataFrame
        DataFrame con columna de fecha y la variable de interés.
    fecha_col : str
        Nombre de la columna de fecha.
    valor_col : str
        Nombre de la columna numérica sobre la que calcular estadísticas.
    dias : int
        Ventana de tiempo en días (por defecto 30).
    
    Retorna
    -------
    df : DataFrame
        DataFrame con dos nuevas columnas:
        - '<valor_col>_media_nd'
        - '<valor_col>_std_nd'
    """
    df = df.copy()
    
    # Asegurarse de que la fecha esté en datetime y ordenar
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col)
    
    # Convertir la columna de fecha en índice
    df = df.set_index(fecha_col)
    
    # Calcular estadísticas en ventana móvil de n días
    df[f"{valor_col}_media_{dias}d"] = df[valor_col].rolling(f"{dias}D", min_periods=1).mean()
    df[f"{valor_col}_std_{dias}d"]   = df[valor_col].rolling(f"{dias}D", min_periods=1).std()
    
    # Restaurar índice
    df = df.reset_index()
    
    return df

    # EJEMPLO DE USO
    #
    # Para profundidad
    # gdf_2002 = estadisticas_ventana_temporal(gdf_2002, fecha_col="FechaHora", valor_col="Prof_Km", dias=30)

import pandas as pd

#######################################################
#
# GENERALIZACIÓN DEL PARÁMETRO 5
# Calcula valor medio, desviación estándar, mediana, máximo, mínimo
# de cualquier variable en los últimos 30 días
#
# Función para obtener esas estadísticas 
# correspondiente los eventos ocurridos en los n dias previos
# Actualiza el DataFrame con el resultado
####################################################### 

def estadisticas_ventana_temporal(
    df, 
    fecha_col="FechaHora", 
    valor_col="Mag_mbLgL", 
    dias=30, 
    metricas=("mean", "std", "median", "max", "min")
):
    """
    Calcula estadísticas (media, desviación estándar, mediana, máximo, mínimo, etc.) 
    de una variable en una ventana temporal de n días.

    Parámetros
    ----------
    df : DataFrame
        DataFrame con columna de fecha y la variable de interés.
    fecha_col : str
        Nombre de la columna de fecha.
    valor_col : str
        Nombre de la columna numérica sobre la que calcular estadísticas.
    dias : int
        Ventana de tiempo en días (por defecto 30).
    metricas : tuple o lista
        Métricas a calcular (mean, std, median, max, min...).

    Retorna
    -------
    df : DataFrame
        DataFrame con nuevas columnas para cada métrica seleccionada.
    """
    df = df.copy()
    
    # Asegurar formato fecha y ordenar
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col)
    
    # Convertir fecha a índice
    df = df.set_index(fecha_col)

    # Calcular estadísticas en ventana móvil de n días
    for metrica in metricas:
        df[f"{valor_col}_{metrica}_{dias}d"] = (
            df[valor_col].rolling(f"{dias}D", min_periods=1).__getattribute__(metrica)()
        )
    
    # Restaurar índice
    df = df.reset_index()
    return df

    # EJEMPLO DE USO
    #
    # Para magnitud con todas las métricas
    #gdf_2002 = estadisticas_ventana_temporal(
    #    gdf_2002, 
    #    fecha_col="FechaHora", 
    #    valor_col="Mag_mbLgL", 
    #    dias=30, 
    #    metricas=("mean", "std", "median", "max", "min")
    #)

    # Ver columnas nuevas creadas
    #print(gdf_2002.filter(like="Mag_mbLgL").tail(10))


#######################################################
#
# 6a. Cambios en la tasa sísmica (z-value)
#
# Función para obtener la tasa sísmica.  
# Se calcula el z-value en los intervalos (t-1,T) y (T+1,2*T)
# Actualiza el DataFrame con el resultado
####################################################### 

import pandas as pd
import numpy as np

def calcular_z_value_vectorizado(df, fecha_col="FechaHora", T=30, col_out="z_value"):
    """
    Calcula el z-value para cada evento de manera vectorizada usando ventanas deslizantes.
    Excluye siempre el día del evento.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los eventos sísmicos.
    fecha_col : str
        Columna con la fecha (datetime).
    T : int
        Tamaño de la ventana en días (por defecto 30).
    col_out : str
        Nombre de la columna de salida.
    
    Devuelve
    --------
    df : pd.DataFrame con nueva columna `col_out`.
    """
    df = df.copy()
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    # --- 1) Serie diaria de conteo ---
    eventos_por_dia = df.groupby(df[fecha_col].dt.floor("D")).size()
    eventos_por_dia = eventos_por_dia.asfreq("D", fill_value=0)  # rellenar días vacíos

    # --- 2) Rolling windows ---
    # Ventana actual (los T días previos, excluyendo el día del evento → shift(1))
    n1 = eventos_por_dia.rolling(window=T).sum().shift(1)
    s1 = eventos_por_dia.rolling(window=T).var(ddof=1).shift(1)  # varianza diaria

    # Ventana previa (los T días antes de la ventana actual → shift(T+1))
    n2 = eventos_por_dia.rolling(window=T).sum().shift(T+1)
    s2 = eventos_por_dia.rolling(window=T).var(ddof=1).shift(T+1)

    # Tasas (eventos por día)
    R1 = n1 / T
    R2 = n2 / T

    # --- 3) Fórmula z-value ---
    denom = np.sqrt((s1 / T) + (s2 / T))
    z_series = (R1 - R2) / denom

    # --- 4) Mapear de nuevo al dataframe ---
    df[col_out] = df[fecha_col].dt.floor("D").map(z_series)

    return df


#######################################################
#
# 6b. Cambios en la tasa sísmica (beta-value)
#
# Función para obtener la tasa sísmica.  
# Se calcula el beta-value
# Actualiza el DataFrame con el resultado
#######################################################

import pandas as pd
import numpy as np

def calcular_beta_value(df, fecha_col="FechaHora", ventana_dias=30):
    """
    Calcula el β-value para cada evento en un DataFrame, usando rolling windows.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Catálogo sísmico con columna de fechas.
    fecha_col : str
        Nombre de la columna con las fechas (tipo datetime).
    ventana_dias : int
        Duración de la ventana de interés (Δt), en días.

    Devuelve:
    ---------
    df : pandas.DataFrame
        Mismo DataFrame con nuevas columnas 'M_obs' y 'Beta_value'.
    """
    df = df.copy()

    # Asegurar datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Ordenar por fecha
    df = df.sort_values(fecha_col).reset_index(drop=True)

    # Crear serie diaria de conteo de eventos
    conteo_diario = df.groupby(df[fecha_col].dt.floor("D")).size()

    # Parámetros globales
    t_total = (conteo_diario.index.max() - conteo_diario.index.min()).days + 1
    n_total = len(df)
    delta = ventana_dias / t_total

    if delta >= 1:
        raise ValueError("La ventana es mayor o igual a la duración del catálogo.")

    # Conteo acumulado con ventana deslizante
    M_obs_diario = conteo_diario.rolling(f"{ventana_dias}D").sum()

    # Reasignar el número de eventos observados a cada fila
    df["M_obs"] = df[fecha_col].dt.floor("D").map(M_obs_diario)

    # Calcular β-value
    df["Beta_value"] = (df["M_obs"] - n_total * delta) / np.sqrt(n_total * delta * (1 - delta))

    return df

#######################################################
#
# 7. Magnitud máxima en los últimos T dias
#
# La función permite calcular la magnitud máxima para
# varios periodos de tiempo a la vez
#
# Actualiza el DataFrame con el resultado
#######################################################

import pandas as pd
import numpy as np

def agregar_magnitud_max_multiventana(df, fecha_col="FechaHora", mag_col="Mag_mbLgL", ventanas=[30, 90, 180]):
    """
    Añade al DataFrame columnas con la magnitud máxima registrada
    en los últimos T días previos a cada evento, para múltiples ventanas.

    Parámetros
    ----------
    df : pd.DataFrame
        Catálogo con columnas de fecha y magnitud.
    fecha_col : str
        Nombre de la columna con fechas (tipo datetime).
    mag_col : str
        Nombre de la columna con magnitudes.
    ventanas : list
        Lista de ventanas en días.

    Retorna
    -------
    df : pd.DataFrame
        DataFrame actualizado con nuevas columnas 'MagMax_{T}d'.
    """
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    # Serie temporal con índice de fechas
    serie = df.set_index(fecha_col)[mag_col]

    for T in ventanas:
        # Rolling máximo excluyendo el día actual (closed='left')
        mag_max = serie.rolling(f"{T}D", closed="left").max()
        df[f"MagMax_{T}d"] = mag_max.values

    return df

#######################################################
#
# 8. Profundidad media de los eventos ocurridos en los últimos T dias
#
# La función permite calcular la profundidad media para
# varios periodos de tiempo a la vez
#
# Actualiza el DataFrame con el resultado
#######################################################

def agregar_profundidad_media_multiventana(df, fecha_col="FechaHora", prof_col="Prof_Km", ventanas=[30, 90, 180]):
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    serie = df.set_index(fecha_col)[prof_col]

    for T in ventanas:
        prof_media = serie.rolling(f"{T}D", closed="left").mean()
        df[f"ProfMedia_{T}d"] = prof_media.values

    return df

#######################################################
#
# 9. Tiempo desde el último terremoto
#
# Actualiza el DataFrame con el resultado
#######################################################

import pandas as pd

def agregar_tiempo_desde_ultimo(df, fecha_col="FechaHora", unidad="dias"):
    """
    Añade al DataFrame una columna con el tiempo transcurrido desde el último terremoto.

    Parámetros
    ----------
    df : pd.DataFrame
        Catálogo sísmico con columna de fechas.
    fecha_col : str
        Nombre de la columna con la fecha del evento.
    unidad : str
        Unidad de tiempo para la diferencia ('dias', 'horas', 'minutos', 'segundos').

    Retorna
    -------
    df : pd.DataFrame
        DataFrame actualizado con nueva columna 'TiempoDesdeUltimo'.
    """

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    # Diferencia entre eventos consecutivos
    tiempo_diff = df[fecha_col].diff()

    # Convertir a la unidad deseada
    if unidad == "dias":
        tiempo_diff = tiempo_diff.dt.total_seconds() / (24*3600)
    elif unidad == "horas":
        tiempo_diff = tiempo_diff.dt.total_seconds() / 3600
    elif unidad == "minutos":
        tiempo_diff = tiempo_diff.dt.total_seconds() / 60
    elif unidad == "segundos":
        tiempo_diff = tiempo_diff.dt.total_seconds()
    else:
        raise ValueError("Unidad no soportada. Usar 'dias', 'horas', 'minutos' o 'segundos'.")

    # Añadir columna al dataframe
    df["TiempoDesdeUltimo"] = tiempo_diff

    return df


#######################################################
#
# 10. Tiempo transcurrido entre n eventos
#
# Actualiza el DataFrame con el resultado
#######################################################

def agregar_tiempo_ventana_eventos_vector(df, fecha_col="FechaHora", n_eventos=5, unidad="dias"):
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    col_nombre = f"TiempoVentana_{n_eventos}Eventos"

    # Convertir fechas a timestamps (segundos desde epoch)
    ts = df[fecha_col].view("int64") / 1e9  # nanosegundos a segundos

    # Rolling max y min sobre los timestamps
    rolling_max = pd.Series(ts).rolling(window=n_eventos, min_periods=n_eventos).max()
    rolling_min = pd.Series(ts).rolling(window=n_eventos, min_periods=n_eventos).min()

    tiempo_diff = rolling_max - rolling_min  # en segundos

    # Convertir a la unidad deseada
    if unidad == "dias":
        tiempo_diff = tiempo_diff / (24*3600)
    elif unidad == "horas":
        tiempo_diff = tiempo_diff / 3600
    elif unidad == "minutos":
        tiempo_diff = tiempo_diff / 60
    elif unidad == "segundos":
        pass  # ya está en segundos
    else:
        raise ValueError("Unidad no soportada. Usar 'dias', 'horas', 'minutos' o 'segundos'.")

    df[col_nombre] = tiempo_diff.values

    return df



#######################################################
#
# 11. Tiempo medio entre eventos
#
# Actualiza el DataFrame con el resultado
#######################################################  

import pandas as pd

def agregar_tiempo_medio_ventana_eventos(df, fecha_col="FechaHora", n_eventos=5, unidad="dias"):
    """
    Añade al DataFrame una columna con el tiempo medio entre eventos
    considerando solo los últimos n eventos previos a cada evento.

    Parámetros
    ----------
    df : pd.DataFrame
        Catálogo sísmico con columna de fechas.
    fecha_col : str
        Nombre de la columna con la fecha del evento.
    n_eventos : int
        Número de eventos de la ventana hacia atrás.
    unidad : str
        Unidad de tiempo para la diferencia ('dias', 'horas', 'minutos', 'segundos').

    Retorna
    -------
    df : pd.DataFrame
        DataFrame actualizado con nueva columna 'TiempoMedio_{n_eventos}Eventos'.
    """

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    col_nombre = f"TiempoMedio_{n_eventos}Eventos"

    # Convertir fechas a timestamps (segundos desde epoch)
    ts = df[fecha_col].view("int64") / 1e9  # nanosegundos a segundos

    # Diferencias entre eventos consecutivos
    dif = pd.Series(ts).diff()

    # Ventana rolling de n-1 diferencias para calcular promedio de los últimos n eventos
    tiempo_medio = dif.rolling(window=n_eventos-1, min_periods=n_eventos-1).mean()

    # Convertir a la unidad deseada
    if unidad == "dias":
        tiempo_medio = tiempo_medio / (24*3600)
    elif unidad == "horas":
        tiempo_medio = tiempo_medio / 3600
    elif unidad == "minutos":
        tiempo_medio = tiempo_medio / 60
    elif unidad == "segundos":
        pass  # ya está en segundos
    else:
        raise ValueError("Unidad no soportada. Usar 'dias', 'horas', 'minutos' o 'segundos'.")

    df[col_nombre] = tiempo_medio.values

    return df


#######################################################
#
# 12. Coeficiente de variación
#
# Actualiza el DataFrame con el resultado
#######################################################  

import pandas as pd
import numpy as np

def agregar_coeficiente_variacion(df, fecha_col="FechaHora", n_eventos=5, unidad="dias"):
    """
    Añade al DataFrame una columna con el coeficiente de variación (C) 
    de los intervalos de tiempo entre los últimos n eventos.

    Parámetros
    ----------
    df : pd.DataFrame
        Catálogo sísmico con columna de fechas.
    fecha_col : str
        Nombre de la columna con la fecha del evento.
    n_eventos : int
        Número de eventos para calcular la ventana hacia atrás.
    unidad : str
        Unidad de tiempo para los intervalos ('dias', 'horas', 'minutos', 'segundos').

    Retorna
    -------
    df : pd.DataFrame
        DataFrame actualizado con nueva columna 'CoefVar_{n_eventos}Eventos'.
    """

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    col_nombre = f"CoefVar_{n_eventos}Eventos"

    # Convertir fechas a timestamps (segundos desde epoch)
    ts = df[fecha_col].view("int64") / 1e9  # nanosegundos a segundos

    # Diferencias entre eventos consecutivos
    dif = pd.Series(ts).diff()

    # Rolling window de tamaño n-1 para los intervalos
    rolling_window = dif.rolling(window=n_eventos-1, min_periods=n_eventos-1)

    # Media y desviación estándar en cada ventana
    mu = rolling_window.mean()
    sigma = rolling_window.std(ddof=1)

    # Coeficiente de variación
    coef_var = sigma / mu

    # Convertir a la unidad deseada
    if unidad == "dias":
        coef_var = coef_var  # adimensional, no depende de la unidad
    elif unidad in ["horas", "minutos", "segundos"]:
        coef_var = coef_var  # coeficiente adimensional
    else:
        raise ValueError("Unidad no soportada. Usar 'dias', 'horas', 'minutos' o 'segundos'.")

    df[col_nombre] = coef_var.values

    return df



#######################################################
#
# 13. Número de terremotos en los últimos 30 días
#
# Actualiza el DataFrame con el resultado
####################################################### 

import pandas as pd

def contar_eventos_ultimos_T_dias(df, fecha_col="FechaHora", T=30):
    """
    Añade al DataFrame una columna con el número de terremotos ocurridos
    en los últimos T días para cada evento.

    Parámetros
    ----------
    df : pd.DataFrame
        Catálogo sísmico con columna de fechas.
    fecha_col : str
        Nombre de la columna con la fecha del evento.
    T : int o float
        Tamaño de la ventana en días.

    Retorna
    -------
    df : pd.DataFrame
        DataFrame actualizado con nueva columna 'NumEventosUltimos_{T}dias'.
    """

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(by=fecha_col).reset_index(drop=True)

    col_nombre = f"NumEventosUltimos_{T}dias"

    # Convertimos fechas a índice temporal para usar rolling por tiempo
    df.set_index(fecha_col, inplace=True)

    # Crear serie de 1's para contar eventos
    s = pd.Series(1, index=df.index)

    # Ventana de T días, sumando los 1's para contar eventos
    df[col_nombre] = s.rolling(f"{T}D").sum().values

    # Restaurar el índice original
    df.reset_index(inplace=True)

    return df


#######################################################
#
# CARACTERISTICAS NO PARAMETRICAS
#
####################################################### 

#######################################################
#
# 1. Probabilidad de ocurrencia de un terremoto
#
# Actualiza el DataFrame con el resultado
####################################################### 

import numpy as np
import pandas as pd

def agregar_prob_ocurrencia(df, b_cols, M_list=6.0, Mmin=4.0, sufijo="prob"):
    """
    Calcula la probabilidad de ocurrencia de eventos de magnitud >= M
    según la Ley de Gutenberg-Richter, vectorizada y flexible.

    Fórmula: P = exp(-(M - Mmin) * b * ln(10))
             ln(10) = 1 / log(e)

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas de b-values.
    b_cols : list
        Lista con nombres de las columnas que contienen b-values.
    M_list : float o list[float], opcional
        Magnitud o lista de magnitudes umbral (default = 6.0).
    Mmin : float, opcional
        Magnitud mínima de completitud del catálogo (default = 4.0).
    sufijo : str, opcional
        Sufijo para nombrar las nuevas columnas (default = "prob").

    Retorna
    -------
    pd.DataFrame
        DataFrame original + columnas nuevas con probabilidades.
    """
    # Asegurar que M_list sea iterable
    if np.isscalar(M_list):
        M_list = [M_list]

    ln10 = np.log(10)
    delta_M = np.array(M_list) - Mmin  # shape (len(M_list),)

    # Extraer b-values
    B = df[b_cols].to_numpy()  # shape (n, len(b_cols))

    # Broadcasting para aplicar todas las magnitudes
    P = np.exp(-np.expand_dims(B, axis=2) * delta_M * ln10)
    # shape (n, len(b_cols), len(M_list))

    # Convertir a DataFrame plano
    colnames = [f"{col}_{sufijo}_M{M}" for col in b_cols for M in M_list]
    P_2d = P.reshape(len(df), -1)
    df_probs = pd.DataFrame(P_2d, columns=colnames, index=df.index)

    return pd.concat([df, df_probs], axis=1)

# EJEMPLO DE USO
# Un solo umbral
# df1 = agregar_prob_ocurrencia(df, ["b_lsq", "b_mlk"], M_list=6, Mmin=4)

# Varios umbrales
# df2 = agregar_prob_ocurrencia(df, ["b_lsq", "b_mlk"], M_list=[5, 6, 7], Mmin=4)


#######################################################
#
# 2a. Desviación de la Ley de Gutenberg-Richter
# Utiliza una ventana de T dias previos al evento seleccionado
#
# Actualiza el DataFrame con el resultado
#######################################################

import numpy as np
import pandas as pd
from numba import njit

# ---------------------------------------------------
# Función Numba: cálculo de eta en una ventana temporal
# ---------------------------------------------------
@njit
def calcular_eta_numba(fecha_sec, mags, a_vals, b_vals, ventana_segundos):
    n = len(fecha_sec)
    eta = np.full(n, np.nan)  # inicializamos con NaN
    
    for i in range(n):
        t0 = fecha_sec[i] - ventana_segundos
        t1 = fecha_sec[i]
        
        # seleccionar eventos dentro de la ventana
        mask = (fecha_sec >= t0) & (fecha_sec <= t1)
        m_window = mags[mask]
        
        if len(m_window) > 1:
            n_eventos = len(m_window)
            
            # N observado acumulado (log10)
            N_obs = np.arange(n_eventos, 0, -1)  # cuenta descendente
            logN = np.log10(N_obs)
            
            # recta ajustada: a - bM
            a = a_vals[i]
            b = b_vals[i]
            M = np.sort(m_window)[::-1]  # magnitudes descendentes
            ajuste = a - b * M
            
            # eta
            diff = logN - ajuste
            eta[i] = np.sum(diff**2) / (n_eventos - 1)
        else:
            eta[i] = np.nan
    return eta


# ---------------------------------------------------
# Función envoltorio para dataframe
# ---------------------------------------------------
def eta_numba_vector(df, fecha_col, col_mag, a_lsq, b_lsq, a_mlk, b_mlk, dias=30):
    # Convertir fechas a segundos (int64)
    fecha_sec = df[fecha_col].astype("int64").to_numpy() // 10**9
    mags = df[col_mag].to_numpy(dtype=np.float64)
    
    ventana_segundos = dias * 24 * 3600
    
    # LSQ
    a_vals = df[a_lsq].to_numpy(dtype=np.float64)
    b_vals = df[b_lsq].to_numpy(dtype=np.float64)
    df["eta_T_lsq"] = calcular_eta_numba(fecha_sec, mags, a_vals, b_vals, ventana_segundos)
    
    # MLK
    a_vals = df[a_mlk].to_numpy(dtype=np.float64)
    b_vals = df[b_mlk].to_numpy(dtype=np.float64)
    df["eta_T_mlk"] = calcular_eta_numba(fecha_sec, mags, a_vals, b_vals, ventana_segundos)
    
    return df


# EJEMPLO DE USO
#gdf_eq = eta_numba_vector(
#    gdf_eq,
#    fecha_col="FechaHora",
#    col_mag="Mag_mbLgL",
#    a_lsq="a_lsq",
#    b_lsq="b_lsq",
#    a_mlk="a_mlk",
#    b_mlk="b_mlk",
#    dias=30
#)

#gdf_eq[["FechaHora", "eta_lsq", "eta_mlk"]].tail()


#######################################################
#
# 2b. Desviación de la Ley de Gutenberg-Richter
# Utiliza una ventana de N eventos previos al evento seleccionado
#
# Actualiza el DataFrame con el resultado
#######################################################

import numpy as np
import pandas as pd

def eta_ventana_eventos(df, col_mag="Mag_mbLgL", a_lsq="a_lsq", b_lsq="b_lsq",
                        a_mlk="a_mlk", b_mlk="b_mlk", n_eventos=50):
    """
    Calcula eta de Gutenberg-Richter para cada evento usando los últimos n_eventos previos.

    Parámetros:
        df : pd.DataFrame
            DataFrame con magnitudes y parámetros a, b.
        col_mag : str
            Nombre de la columna de magnitudes.
        a_lsq, b_lsq : str
            Columnas con parámetros LSQ.
        a_mlk, b_mlk : str
            Columnas con parámetros MLK.
        n_eventos : int
            Número de eventos previos para calcular eta.

    Retorna:
        df actualizado con columnas 'eta_N_lsq' y 'eta_N_mlk'
    """
    df = df.copy()
    mags = df[col_mag].values
    n_total = len(df)
    
    eta_lsq_arr = np.full(n_total, np.nan)
    eta_mlk_arr = np.full(n_total, np.nan)
    
    for i in range(n_eventos, n_total):
        window_mags = mags[i-n_eventos:i]
        window_n = len(window_mags)
        logN = np.log10(np.array([np.sum(window_mags >= M) for M in window_mags]))
        
        # Eta LSQ
        eta_lsq = ((logN - (df[a_lsq].values[i-n_eventos:i] + 
                            df[b_lsq].values[i-n_eventos:i] * window_mags))**2).sum() / (window_n - 1)
        # Eta MLK
        eta_mlk = ((logN - (df[a_mlk].values[i-n_eventos:i] + 
                            df[b_mlk].values[i-n_eventos:i] * window_mags))**2).sum() / (window_n - 1)
        
        eta_lsq_arr[i] = eta_lsq
        eta_mlk_arr[i] = eta_mlk
    
    df["eta_N_lsq"] = eta_lsq_arr
    df["eta_N_mlk"] = eta_mlk_arr
    
    return df

# EJEMPLO DE USO
#gdf_eq = eta_ventana_eventos(
#    gdf_eq, 
#    col_mag="Mag_mbLgL",
#    a_lsq="a_lsq",
#    b_lsq="b_lsq",
#    a_mlk="a_mlk",
#    b_mlk="b_mlk",
#    n_eventos=50
#)

#gdf_eq[["eta_lsq", "eta_mlk"]].tail()


#######################################################
#
# 3a. Desviación estándar del parámetro b
# Utiliza una ventana de T dias previos al evento seleccionado
#
# Actualiza el DataFrame con el resultado
#######################################################

import numpy as np
from numba import njit

# ----------------------------------------------
# Función Numba para calcular sigma_b en ventana de tiempo
# ----------------------------------------------
@njit
def sigma_b_numba_tiempo(fecha_sec, mags, b_vals, ventana_segundos):
    n = len(fecha_sec)
    sigma_b = np.full(n, np.nan)
    
    for i in range(n):
        t0 = fecha_sec[i] - ventana_segundos
        t1 = fecha_sec[i]

        # seleccionar magnitudes dentro de la ventana
        mask = (fecha_sec >= t0) & (fecha_sec <= t1)
        window = mags[mask]

        if len(window) > 1:
            mean_m = np.mean(window)
            n_ev = len(window)
            var_m = np.sum((window - mean_m) ** 2) / (n_ev * (n_ev - 1))

            b = b_vals[i]
            sigma_b[i] = 2.3 * (b ** 2) * np.sqrt(var_m)
        else:
            sigma_b[i] = np.nan
    
    return sigma_b


# ----------------------------------------------
# Envoltorio para DataFrame
# ----------------------------------------------
def agregar_sigma_b_tiempo(df, fecha_col, col_mag, b_lsq, b_mlk, dias=30):
    # convertir a arrays NumPy explícitamente
    fecha_sec = df[fecha_col].astype("int64").to_numpy() // 10**9
    mags = df[col_mag].to_numpy(dtype=np.float64)
    ventana_segundos = dias * 24 * 3600
    
    # LSQ
    b_vals = df[b_lsq].to_numpy(dtype=np.float64)
    df[f"sigma_b_lsq_{dias}d"] = sigma_b_numba_tiempo(fecha_sec, mags, b_vals, ventana_segundos)

    # MLK
    b_vals = df[b_mlk].to_numpy(dtype=np.float64)
    df[f"sigma_b_mlk_{dias}d"] = sigma_b_numba_tiempo(fecha_sec, mags, b_vals, ventana_segundos)

    return df


# EJEMPLO DE USO
# Por ventana de días
# gdf_eq = agregar_sigma_b_tiempo(gdf_eq, fecha_col="FechaHora", col_mag="Mag_mbLgL",
#                                b_lsq="b_lsq", b_mlk="b_mlk", dias=30)
#
# gdf_eq.tail()[["FechaHora", "sigma_b_lsq_30d", "sigma_b_mlk_30d"]]

#######################################################
#
# 3B. Desviación estándar del parámetro b
# Utiliza una ventana de N eventos previos al evento seleccionado
#
# Actualiza el DataFrame con el resultado
#######################################################


import numpy as np
from numba import njit

# ----------------------------------------------
# Función Numba: desviación estándar de b en ventana de N eventos
# ----------------------------------------------
@njit
def sigma_b_numba_eventos(mags, b_vals, n_eventos):
    n = len(mags)
    sigma_b = np.full(n, np.nan)

    for i in range(n):
        if i >= n_eventos - 1:
            window = mags[i - n_eventos + 1 : i + 1]

            if len(window) > 1:
                mean_m = np.mean(window)
                n_ev = len(window)
                var_m = np.sum((window - mean_m) ** 2) / (n_ev * (n_ev - 1))

                b = b_vals[i]
                sigma_b[i] = 2.3 * (b ** 2) * np.sqrt(var_m)
            else:
                sigma_b[i] = np.nan
        else:
            sigma_b[i] = np.nan

    return sigma_b


# ----------------------------------------------
# Envoltorio para DataFrame
# ----------------------------------------------
def agregar_sigma_b_eventos(df, col_mag, b_lsq, b_mlk, n_eventos=50):
    # convertir a arrays NumPy
    mags = df[col_mag].to_numpy(dtype=np.float64)

    # LSQ
    b_vals = df[b_lsq].to_numpy(dtype=np.float64)
    df[f"sigma_b_lsq_{n_eventos}ev"] = sigma_b_numba_eventos(mags, b_vals, n_eventos)

    # MLK
    b_vals = df[b_mlk].to_numpy(dtype=np.float64)
    df[f"sigma_b_mlk_{n_eventos}ev"] = sigma_b_numba_eventos(mags, b_vals, n_eventos)

    return df


# EJEMPLO DE USO
# Por número de eventos
# gdf_eq = agregar_sigma_b_eventos(gdf_eq, col_mag="Mag_mbLgL", b_lsq="b_lsq", b_mlk="b_mlk", n_eventos=10)
#
# gdf_eq.tail()[["FechaHora", "sigma_b_lsq_30d", "sigma_b_mlk_30d"]]
#



########################################################
# 4a. Déficit de magnitud
# Utiliza el evento actual 
#
# Actualiza el DataFrame con el resultado
#######################################################

def agregar_deficit_evento(df, col_mag="Mag", 
                           col_a1="a_lsq", col_b1="b_lsq", 
                           col_a2="a_mlk", col_b2="b_mlk"):
    """
    Calcula el déficit de magnitud evento a evento:
        Mdef = M_evento - (a / b)
    Genera dos columnas: Mdef_evento_lsq y Mdef_evento_mlk
    """
    df["Mdef_evento_lsq"] = df[col_mag] - (df[col_a1] / df[col_b1])
    df["Mdef_evento_mlk"] = df[col_mag] - (df[col_a2] / df[col_b2])
    return df


########################################################
# 4b. Déficit de magnitud
# Utiliza una ventana de T días previos al evento seleccionado
#
# Actualiza el DataFrame con el resultado
#######################################################

import pandas as pd

def agregar_deficit_ventana_tiempo(df, col_mag="Mag", fecha_col="FechaHora", T=365,
                                   col_a1="a_lsq", col_b1="b_lsq", 
                                   col_a2="a_mlk", col_b2="b_mlk"):
    """
    Calcula el déficit de magnitud en una ventana temporal de T días.
    Genera dos columnas: Mdef_Tdias_lsq y Mdef_Tdias_mlk
    """
    df = df.sort_values(fecha_col).copy()

    # Mmax en ventana T días
    rolling_max = df.set_index(fecha_col)[col_mag].rolling(f"{T}D").max().reset_index(drop=True)

    df["Mdef_Tdias_lsq"] = rolling_max - (df[col_a1] / df[col_b1])
    df["Mdef_Tdias_mlk"] = rolling_max - (df[col_a2] / df[col_b2])
    return df



########################################################
# 4c. Déficit de magnitud
# Utiliza una ventana de N eventos previos al evento seleccionado
#
# Actualiza el DataFrame con el resultado
#######################################################

def agregar_deficit_ventana_eventos(df, col_mag="Mag", fecha_col="FechaHora", n_eventos=50,
                                    col_a1="a_lsq", col_b1="b_lsq", 
                                    col_a2="a_mlk", col_b2="b_mlk"):
    """
    Calcula el déficit de magnitud en una ventana de N eventos.
    Genera dos columnas: Mdef_Neventos_lsq y Mdef_Neventos_mlk
    """
    df = df.sort_values(fecha_col).copy()

    # Mmax en ventana de n_eventos
    rolling_max = df[col_mag].rolling(window=n_eventos, min_periods=1).max()

    df["Mdef_Neventos_lsq"] = rolling_max - (df[col_a1] / df[col_b1])
    df["Mdef_Neventos_mlk"] = rolling_max - (df[col_a2] / df[col_b2])
    return df

# EJEMPLO DE USO
#
# gdf = agregar_deficit_evento(gdf, col_mag="Mag_mbLgL")
# gdf = agregar_deficit_ventana_tiempo(gdf, col_mag="Mag_mbLgL", fecha_col="FechaHora", T=180)
# gdf = agregar_deficit_ventana_eventos(gdf, col_mag="Mag_mbLgL", fecha_col="FechaHora", n_eventos=100)
#


########################################################
# 4_bis. Déficit de magnitud (MÉTODO DE KIJKO)
#
# Utiliza el evento actual
# Utiliza una ventana de T días previos al evento seleccionado
# Utiliza una ventana de N eventos previos al evento seleccionado
#
# 
# Actualiza el DataFrame con el resultado
#######################################################

import pandas as pd
import numpy as np

def agregar_mmax_kijko_vector_optim(df, col_mag="Mag", fecha_col="FechaHora",
                                     b_cols=["b_lsq", "b_mlk"], T_list=[5], N_list=[10]):
    """
    Versión ultra-optimizada de Mmax Kijko.
    Crea todas las columnas de golpe para todos los b y ventanas.

    Parámetros:
    - col_mag: columna de magnitudes.
    - fecha_col: columna con fechas.
    - b_cols: lista con columnas de b.
    - T_list: lista de ventanas temporales en días (puede ser vacía).
    - N_list: lista de ventanas de N eventos (puede ser vacía).
    """
    mags = df[col_mag].to_numpy()
    n_total = len(mags)

    # Convertir fechas a segundos si se usan ventanas temporales
    if len(T_list) > 0:
        fechas_sec = pd.to_datetime(df[fecha_col]).astype(np.int64) // 10**9

    for b_name in b_cols:
        b_vals = df[b_name].to_numpy()
        mags_pow = 10 ** (b_vals * mags)
        cum_mags = np.concatenate(([0], np.cumsum(mags_pow)))  # padding 0

        # --- Evento completo ---
        Mmax_evento = np.log10(cum_mags[1:] / 1) / b_vals
        df[f"Mmax_kijko_evento_{b_name}"] = Mmax_evento

        # --- Ventanas temporales ---
        for T in T_list:
            ventana_seg = T * 86400
            start_idx = np.searchsorted(fechas_sec, fechas_sec - ventana_seg, side='left')
            delta_cum = cum_mags[1 + np.arange(n_total)] - cum_mags[start_idx]
            Mmax_T = np.log10(delta_cum) / b_vals
            df[f"Mmax_kijko_T{T}d_{b_name}"] = Mmax_T

        # --- Ventanas de N eventos ---
        for N in N_list:
            start_idx_ev = np.maximum(0, np.arange(n_total) - N)
            delta_cum_ev = cum_mags[1 + np.arange(n_total)] - cum_mags[start_idx_ev]
            Mmax_N = np.log10(delta_cum_ev) / b_vals
            df[f"Mmax_kijko_{N}ev_{b_name}"] = Mmax_N

    return df

# -------------------------
# EJEMPLO DE USO
# -------------------------

# Calcular todas las columnas de golpe
# df = agregar_mmax_kijko_vector_optim(df, col_mag="Mag_mbLgL",
#                                      b_cols=["b_lsq", "b_mlk"],
#                                      T_list=[5], N_list=[10])

# print(df.tail())



########################################################
# 5. Tiempo de recurrencia total
# La magnitud va desde 4.0 a 6.0 en intervalos de 0.1
# a y b pueden tomar dos valores diferentes
#
# Actualiza el DataFrame con el resultado
#######################################################

import numpy as np
import pandas as pd

def agregar_tiempo_recurrencia(df, fecha_col="FechaHora", 
                               col_a1="a_lsq", col_b1="b_lsq", 
                               col_a2="a_mlk", col_b2="b_mlk",
                               Mmin=4.0, Mmax=6.0, step=0.1,
                               unidad="años"):
    """
    Calcula el tiempo de recurrencia sísmico T(Mth) para un rango de magnitudes umbral.
    Añade al dataframe columnas nuevas: Trec_Mxx_lsq y Trec_Mxx_mlk (xx=4.0,4.1,...6.0)
    
    Parámetros:
    - fecha_col: columna con fechas
    - unidad: 'dias', 'meses' o 'años' para TCatálogo
    """
    df = df.copy()

    # 1. Duración del catálogo
    Tcatalogo_dias = (df[fecha_col].max() - df[fecha_col].min()).days
    if unidad == "años":
        Tcatalogo = Tcatalogo_dias / 365.25
    elif unidad == "meses":
        Tcatalogo = Tcatalogo_dias / 30.44
    else:
        Tcatalogo = Tcatalogo_dias  # en días

    # 2. Rango de magnitudes umbral
    Mth_vals = np.round(np.arange(Mmin, Mmax + step, step), 1)  # [4.0,4.1,...6.0]

    # 3. Broadcasting vectorizado
    a_vals = df[[col_a1, col_a2]].to_numpy()  # (N,2)
    b_vals = df[[col_b1, col_b2]].to_numpy()  # (N,2)

    # Expand dims para vectorizar: (N,2,1) y (1,M)
    a_exp = a_vals[:, :, None]  # (N,2,1)
    b_exp = b_vals[:, :, None]  # (N,2,1)
    M_exp = Mth_vals[None, None, :]  # (1,1,M)

    # Fórmula T(Mth)
    Trec = Tcatalogo / (10 ** (a_exp - b_exp * M_exp))  # (N,2,M)

    # 4. Generar nombres de columnas
    colnames = []
    for mb in Mth_vals:
        colnames.append(f"Trec_M{mb:.1f}_lsq")
    for mb in Mth_vals:
        colnames.append(f"Trec_M{mb:.1f}_mlk")

    # 5. Aplanar (N,2,M) → (N,2M)
    Trec_flat = np.concatenate([Trec[:,0,:], Trec[:,1,:]], axis=1)  # (N, 2M)

    # 6. Añadir al dataframe
    df[colnames] = Trec_flat

    return df

# EJEMPLO DE USO
# Supongamos que ya tienes gdf con columnas "FechaHora", "a_lsq", "b_lsq", "a_mlk", "b_mlk"
# gdf = agregar_tiempo_recurrencia(gdf, fecha_col="FechaHora", unidad="años")
#
# Ver columnas nuevas
# print(gdf.filter(like="Trec_").head())


########################################################
# 6. Incremento del valor b
#
# Actualiza el DataFrame con el resultado
#######################################################

import pandas as pd
import numpy as np

def agregar_incrementos_b(df, col_b1="b_lsq", col_b2="b_mlk"):
    """
    Calcula los incrementos del parámetro b en ventanas temporales definidas.
    Crea nuevas columnas en el dataframe para b_lsq y b_mlk.
    """
    df = df.copy()
    
    # Definir pares de ventanas
    pares = [(0,2),(2,4),(4,6),(6,8),(8,10),
             (0,4),(4,8),(8,12),(12,16),(16,20)]
    
    # Seleccionar columnas de b
    bcols = [col_b1, col_b2]
    
    for col in bcols:
        arr = df[col].to_numpy()
        for p1, p2 in pares:
            # shift usando np.roll, pero mejor con pandas shift para evitar wrap-around
            serie = df[col]
            val1 = serie.shift(p1)
            val2 = serie.shift(p2)
            df[f"d{col}_{p1}_{p2}"] = val1 - val2
    
    return df

# EJEMPLO DE USO
#
# gdf = agregar_incrementos_b(gdf, col_b1="b_lsq", col_b2="b_mlk")
#
# Ver las nuevas columnas
# print(gdf.filter(like="db_").head())


########################################################
# 7. Coeficiente de agrupacion
#
# Actualiza el DataFrame con el resultado
#######################################################

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

def calcular_ccluster_aprox(df, fecha_col="FechaHora",
                            lat_col="Latitud", lon_col="Longitud",
                            Tp_dias=30, eps_km=50,
                            nueva_col="Ccluster",
                            chunk_size=10000):
    """
    Aproximación rápida del coeficiente de agrupamiento (Ccluster)
    con barra de progreso. Usa densidad relativa de vecinos
    en un radio eps_km y media temporal en ventana Tp.

    Parámetros:
        df : DataFrame con columnas de fecha, latitud y longitud
        fecha_col : str, columna de fechas
        lat_col, lon_col : str, columnas de coordenadas
        Tp_dias : int, ventana temporal en días
        eps_km : float, radio de vecindad en km
        nueva_col : str, nombre de la columna resultado
        chunk_size : int, tamaño de lote para actualizar progreso
    """
    df = df.sort_values(fecha_col).reset_index(drop=True).copy()
    fechas = pd.to_datetime(df[fecha_col])
    coords = np.radians(df[[lat_col, lon_col]].to_numpy())

    # 1) Contar vecinos en eps_km
    tree = BallTree(coords, metric="haversine")
    neigh_counts = tree.query_radius(coords, r=eps_km/6371.0, count_only=True) - 1

    # 2) Normalización a [0,1]
    C_local = neigh_counts / neigh_counts.max()

    # 3) Media móvil temporal con tqdm
    s = pd.Series(C_local, index=fechas)
    result = np.empty(len(s))
    window = pd.Timedelta(days=Tp_dias)

    for start in tqdm(range(0, len(s), chunk_size),
                      desc="Ccluster chunks"):
        end = min(start + chunk_size, len(s))
        for i in range(start, end):
            t = fechas.iloc[i]
            mask = (fechas <= t) & (fechas >= t - window)
            result[i] = C_local[mask].mean() if mask.any() else np.nan

    df[nueva_col] = result
    return df


# EJEMPLO DE USO
#
# gdf_out = calcular_ccluster_aprox(
#    gdf_2002,
#    fecha_col="FechaHora",
#    lat_col="Latitud",
#    lon_col="Longitud",
#    Tp_dias=30,
#    eps_km=50,
#    chunk_size=5000
# )
#
#


########################################################
# DATAFRAME CON EL VECTOR DE SALIDA
#
# Máxima amplitud detectada en los siguientes 30 días 
#######################################################

import pandas as pd
import numpy as np

def max_mag_next_30d_fast(df, fecha_col="FechaHora", mag_col="Mag", nueva_col="Mmax_next30d"):
    """
    Calcula de forma eficiente la magnitud máxima del siguiente evento
    dentro de los 30 días posteriores a cada terremoto.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con columnas de fecha y magnitud
    fecha_col : str
        Nombre de la columna de fechas
    mag_col : str
        Nombre de la columna de magnitudes
    nueva_col : str
        Nombre de la nueva columna con el resultado

    Retorna:
    --------
    df_out : pd.DataFrame
        Copia del DataFrame original con una nueva columna
        que contiene la magnitud máxima dentro de los 30 días posteriores.
    """
    df = df.copy()
    df = df.sort_values(fecha_col).reset_index(drop=True)
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Creamos una serie con magnitudes indexada por fecha
    s = pd.Series(df[mag_col].values, index=df[fecha_col])

    # Calculamos el máximo en los 30 días hacia adelante con rolling
    # Usamos 'forward-looking' con closed='right'
    max_futuro = (
        s[::-1]  # invertir para hacer ventana hacia adelante
        .rolling("30D", closed="right")
        .max()[::-1]
    )

    # Desplazar 1 para que no cuente el propio evento
    max_futuro = max_futuro.shift(-1)

    # Añadir al dataframe
    df[nueva_col] = max_futuro.values
    return df


# EJEMPLO DE USO
#
# df_out = max_mag_next_30d_fast(df, fecha_col="FechaHora", mag_col="Mag")
# print(df_out)
#
#