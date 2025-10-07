
"""
LIBRERÍA CON ALGUNAS FUNCIONES GENÉRICAS

"""

#######################################################
# Función para normalizar los datos del DataFrame
#######################################################

import unicodedata
import re

def normalizar_columnas(df):
    """
    Normaliza los nombres de columnas de un DataFrame:
    - Elimina acentos
    - Reemplaza espacios por '_'
    - Elimina caracteres no alfanuméricos
    """
    nuevas_cols = []
    for col in df.columns:
        # Quitar acentos
        col_sin_acentos = ''.join(
            c for c in unicodedata.normalize('NFKD', col)
            if not unicodedata.combining(c)
        )
        # Reemplazar espacios por guiones bajos
        col_sin_espacios = col_sin_acentos.strip().replace(" ", "_")
        # Quitar caracteres raros (solo letras, números y '_')
        col_limpio = re.sub(r"[^0-9a-zA-Z_]", "", col_sin_espacios)
        nuevas_cols.append(col_limpio)
    
    df.columns = nuevas_cols
    return df

#######################################################
# Unificación de magnitudes a mbLg (L)
#######################################################

import numpy as np
import pandas as pd

# --- Regresiones ---
def mb_to_mw(mb): return -1.576 + 1.222 * mb
def mblg_pre2002_to_mw(mblg): return 0.258 + 0.980 * mblg
def mblg_post2002_to_mw(mblg): return 0.644 + 0.844 * mblg
def mw_to_mblg_pre2002(mw): return (mw - 0.258) / 0.980
def mw_to_mblg_post2002(mw): return (mw - 0.644) / 0.844

def homogenizar_dataframe(df, col_tipo="Tipo_Mag", col_mag="Mag_num", col_fecha="FechaHora"):
    """
    Convierte todas las magnitudes del DataFrame al sistema homogéneo mbLg(L),
    de manera vectorizada y devuelve un resumen.
    """
    df = df.copy()
    fechas = pd.to_datetime(df[col_fecha])
    corte = pd.Timestamp("2002-03-01")

    # Asegurar magnitudes numéricas
    df[col_mag] = pd.to_numeric(df[col_mag], errors="coerce")
    
    # Crear columna destino vacía
    df["Mag_mbLgL"] = np.nan
    
    # --- Casos tipo 2 (mb V-C) ---
    mask = df[col_tipo] == 2
    if mask.any():
        mw = mb_to_mw(df.loc[mask, col_mag])
        pre = mask & (fechas < corte)
        post = mask & (fechas >= corte)
        df.loc[pre, "Mag_mbLgL"] = mw_to_mblg_pre2002(mw.loc[pre])
        df.loc[post, "Mag_mbLgL"] = mw_to_mblg_post2002(mw.loc[post])

    # --- Casos tipo 3 (MbLg M-MS) ---
    mask = df[col_tipo] == 3
    if mask.any():
        pre = mask & (fechas < corte)
        post = mask & (fechas >= corte)
        df.loc[pre, "Mag_mbLgL"] = mw_to_mblg_pre2002(mblg_pre2002_to_mw(df.loc[pre, col_mag]))
        df.loc[post, "Mag_mbLgL"] = mw_to_mblg_post2002(mblg_post2002_to_mw(df.loc[post, col_mag]))

    # --- Casos tipo 4 (mbLgL ya está en destino) ---
    mask = df[col_tipo] == 4
    df.loc[mask, "Mag_mbLgL"] = df.loc[mask, col_mag]

    # --- Casos tipo 5 y 6 (Mw ya dado) ---
    mask = df[col_tipo].isin([5, 6])
    if mask.any():
        pre = mask & (fechas < corte)
        post = mask & (fechas >= corte)
        df.loc[pre, "Mag_mbLgL"] = mw_to_mblg_pre2002(df.loc[pre, col_mag])
        df.loc[post, "Mag_mbLgL"] = mw_to_mblg_post2002(df.loc[post, col_mag])

    # --- Resumen ---
    total = len(df)
    n_ok = df["Mag_mbLgL"].notna().sum()
    n_nan = df["Mag_mbLgL"].isna().sum()
    print(f"[REPORTE] Total={total}, OK={n_ok}, NaN={n_nan} ({100*n_nan/total:.2f}%)")

    return df


######################################################
# Validar si latitud y longitud coinciden con Points de geopandas
#######################################################

def validar_geometry(gdf, col_lon="Longitud", col_lat="Latitud", tol=1e-8):
    """
    Valida que la geometría POINT(x,y) de un GeoDataFrame coincide con las columnas Longitud y Latitud.
    
    Parámetros:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame que contiene la columna 'geometry' y las columnas de coordenadas.
    col_lon : str, opcional
        Nombre de la columna de longitud. Por defecto "Longitud".
    col_lat : str, opcional
        Nombre de la columna de latitud. Por defecto "Latitud".
    tol : float, opcional
        Tolerancia de diferencia entre coordenadas (por redondeo). Por defecto 1e-8.
    
    Retorna:
    --------
    diff_df : DataFrame
        Subconjunto de filas donde geometry no coincide con Lat/Lon.
        Vacío si todo está correcto.
    """
    # Extraer coordenadas de geometry
    coords = gdf.geometry.apply(lambda p: (p.x, p.y))
    
    # Comparar con las columnas originales
    diffs = gdf[
        (abs(coords.apply(lambda xy: xy[0]) - gdf[col_lon]) > tol) |
        (abs(coords.apply(lambda xy: xy[1]) - gdf[col_lat]) > tol)
    ]
    
    if diffs.empty:
        print("✅ Todas las geometrías coinciden con Longitud y Latitud.")
    else:
        print(f"⚠️ Se encontraron {len(diffs)} discrepancias.")
    
    return diffs

#######################################################
# Ejemplo de llamada a la función

# Validar GeoDataFrame
#diferencias = uti.validar_geometry(gdf_2002)

# Si hay diferencias, ver cuáles son
#if not diferencias.empty:
#    print(diferencias[["Longitud", "Latitud", "geometry"]].head())


######################################################
# Calcular la magnitud de completitud del catálogo
#######################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def estimar_mc_optimo(df, col_mag="Mag_mbLgL", bins=0.1, threshold=0.9, min_events=20, plot=False):
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
#
# mc_opt = estimar_mc_optimo(gdf_eq, col_mag="Mag_mbLgL", bins=0.1, threshold=0.9)
# print("Mc óptimo a usar en el catálogo:", mc_opt)
#


#######################################################
#
# Parámetros de la ley de Gutenberg–Richter (GR)
#
# Función para obtener los parámetros a y b de la Ley de Gutemberg Richter
# Actualiza el DataFrame con el resultado
####################################################### 

import numpy as np
import pandas as pd
from scipy.stats import linregress

def gutemberg_richter(df, col_mag="Mag_mbLgL", min_mag=None):
    """
    Calcula los parámetros a y b de la Ley de Gutenberg-Richter
    mediante Regresión Lineal y Máxima Verosimilitud,
    y añade cuatro columnas al DataFrame:
    - 'a_reg', 'b_reg', 'a_mv', 'b_mv'
    
    Parámetros:
        df : DataFrame con columna de magnitudes
        col_mag : str, nombre de la columna de magnitudes
        min_mag : float o None, magnitud mínima para incluir en el cálculo
    """
    df = df.copy()
    
    # Seleccionar magnitudes ≥ min_mag
    if min_mag is None:
        min_mag = df[col_mag].min()
    mags = df[col_mag][df[col_mag] >= min_mag]

    # --- Método Regresión Lineal ---
    # Frecuencia acumulada (N ≥ M)
    mag_values = np.sort(mags)
    N = np.array([np.sum(mag_values >= m) for m in mag_values])
    logN = np.log10(N)
    
    # Regresión lineal log10(N) vs M
    slope, intercept, r_value, p_value, std_err = linregress(mag_values, logN)
    b_reg = -slope
    a_reg = intercept

    # --- Método Máxima Verosimilitud ---
    M_mean = mags.mean()
    b_mv = np.log10(np.exp(1)) / (M_mean - min_mag)
    # Para a_mv usamos log10(N) = a - b Mmin
    a_mv = np.log10(len(mags)) + b_mv * min_mag

    # --- Añadir columnas al DataFrame ---
    df["a_lsq_global"] = a_reg
    df["b_lsq_global"] = b_reg
    df["a_mlk_global"] = a_mv
    df["b_mlk_global"] = b_mv
    
    return df, {"a_lsq_global": a_reg, "b_lsq_global": b_reg, "a_mlk_global": a_mv, "b_mlk_global": b_mv}

# EJEMPLO DE USO
#
# gdf_2002_1,parametros_GR=par.gutemberg_richter(gdf_2002, col_mag="Mag_mbLgL", min_mag=None)
#
#
