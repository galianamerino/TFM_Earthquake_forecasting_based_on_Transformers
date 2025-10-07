
"""
LIBRERÍA CON VARIAS FUNCIONES PARA REPRESENTAR GRÁFICAMENTE LOS DATOS

"""

#######################################################
# Función para representar el número de terremotos por año
#######################################################

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_terremotos_por_anio_completo(df, fecha_col="FechaHora", output_base=None):
    """
    Representa gráficamente el número de terremotos por año,
    excluyendo los años a los que les falta algún mes.
    
    Parámetros:
        df : DataFrame con columna de fechas
        fecha_col : str, nombre de la columna datetime (default="FechaHora")
        output_base : str o None, ruta/nombre base para guardar el gráfico
                      (sin extensión). Se guardarán PNG y PDF automáticamente.
    """
    # Asegurar que la columna es datetime
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")

    # Extraer año y mes
    df["Año"] = df[fecha_col].dt.year
    df["Mes"] = df[fecha_col].dt.month

    # Detectar años completos (con 12 meses presentes)
    meses_por_anio = df.groupby("Año")["Mes"].nunique()
    anios_completos = meses_por_anio[meses_por_anio == 12].index

    # Filtrar solo esos años completos
    df_filtrado = df[df["Año"].isin(anios_completos)]

    # Contar terremotos por año
    conteo = df_filtrado.groupby("Año").size()

    # --- Plot ---
    plt.figure(figsize=(10,5))
    conteo.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Número de terremotos por año (años completos)")
    plt.ylabel("Número de terremotos")
    plt.xlabel("Año")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # --- Guardar o mostrar ---
    if output_base:
        # Crear carpeta si no existe
        os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
        # Guardar PNG
        plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
        # Guardar PDF
        plt.savefig(f"{output_base}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return conteo


#######################################################
# Función para representar el número de terremotos por año
# No necesita que todos los meses hayan eventos 
#######################################################

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_terremotos_por_anio(df,
                            fecha_col="FechaHora",
                            output_base=None,
                            include_all_years=True,
                            figsize=(10,5),
                            titulo="Número de terremotos por año"):
    """
    Representa el número de terremotos por año.
    - Si la columna `fecha_col` no existe, intenta crearla a partir de 'Fecha' y 'Hora'.
    - Cuenta todos los eventos por año (no requiere que cada año tenga los 12 meses).
    - Si include_all_years=True, rellena con 0 los años que no tengan eventos en el rango.
    - Si output_base se pasa (ruta sin extensión), guarda PNG y PDF.
    
    Retorna:
        conteo (pd.Series): índice = año, valores = número de eventos
    """
    # trabajo sobre copia
    df = df.copy()

    # si no existe la columna fecha_col, intentar componerla desde Fecha + Hora o desde 'Fecha'
    if fecha_col not in df.columns:
        if "Fecha" in df.columns and "Hora" in df.columns:
            df[fecha_col] = pd.to_datetime(df["Fecha"].astype(str).str.strip() + " " +    df_filtrado["Hora"].astype(str).str.strip(),
                                           dayfirst=True, errors="coerce")
        elif "Fecha" in df.columns:
            df[fecha_col] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
        else:
            raise ValueError(f"No existe columna '{fecha_col}' ni 'Fecha' en el DataFrame. "
                             "Pasa la columna correcta con fechas o añade 'Fecha'/'Hora'.")

    # asegurar datetime (coerce para transformar valores inválidos a NaT)
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce", dayfirst=True)

    # quitar filas sin fecha válida
    n_before = len(df)
    df = df[df[fecha_col].notna()]
    n_after = len(df)
    if n_after == 0:
        print("⚠️ No hay fechas válidas tras el parseo. No se puede generar la gráfica.")
        return pd.Series(dtype=int)

    if n_after < n_before:
        print(f"Se han eliminado {n_before-n_after} filas con fecha inválida (NaT).")

    # crear columna año
    df["Año"] = df[fecha_col].dt.year

    # Conteo por año (incluye años con 0 solo si reindexamos después)
    conteo = df.groupby("Año").size().sort_index()

    # si queremos incluir todos los años del rango (incluir años sin eventos)
    if include_all_years and not conteo.empty:
        años_completos = range(int(conteo.index.min()), int(conteo.index.max()) + 1)
        conteo = conteo.reindex(años_completos, fill_value=0)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # barra
    conteo.plot(kind="bar", color="steelblue", edgecolor="black", ax=ax)

    
    plt.ylim(0, 700)

    # etiquetas y formato
    ax.set_title(titulo, fontsize=16)
    ax.set_ylabel("Número de terremotos", fontsize=14)
    ax.set_xlabel("Año", fontsize=14)

    plt.tick_params(axis='both', labelsize=12)

    # formatear etiquetas x si hay muchas barras (mostrar solo algunas etiquetas)
    xticks = ax.get_xticks()
    n_years = len(conteo)
    if n_years > 20:
        # mostrar ~12 etiquetas equiespaciadas para que no se amontonen
        step = max(1, int(n_years // 12))
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_visible(i % step == 0)
    ax.set_xticklabels([str(int(x)) for x in conteo.index], rotation=45, ha="right")

    plt.tight_layout()

    # --- Guardado o mostrar ---
    if output_base:
        # preparar carpeta
        folder = os.path.dirname(output_base)
        if folder:
            os.makedirs(folder, exist_ok=True)
        # guardar PNG y PDF
        plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_base}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráficos guardados en: {output_base}.png  y  {output_base}.pdf")
    else:
        plt.show()

    return conteo


#######################################################
# Función para analizar estadisticamente los dstos de magnitud del catálogo
# Representa Histograma con KDE, Box Plot, Q-Q Plot y Violin Plot en una matriz 2x2
#######################################################

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

def graficos_magnitudes_mejorado(df, col_mag="Mag_mbLgL", output_file=None):
    """
    Genera 4 gráficos de la columna de magnitudes con estadísticos y anotaciones:
    Histograma con KDE, Box Plot, Q-Q Plot y Violin Plot en una matriz 2x2.
    
    Parámetros:
        df : DataFrame con la columna de magnitudes
        col_mag : str, nombre de la columna con magnitudes
        output_file : str o None, ruta base para guardar imagen (PNG/PDF)
    """
    data = df[col_mag].dropna()

    # Estadísticos
    media = data.mean()
    mediana = data.median()
    minimo = data.min()
    maximo = data.max()
    std = data.std()
    
    fig, axes = plt.subplots(2, 2, figsize=(14,10))

    # --- Histograma + KDE ---
    sns.histplot(data, kde=True, bins=30, ax=axes[0,0], color="skyblue")
    axes[0,0].set_title("Histograma con curva de densidad estimada", fontsize=16)
    axes[0,0].set_xlabel("Magnitud (Mag_mbLgL)", fontsize=14)
    axes[0,0].set_ylabel("Cuentas", fontsize=14)
    axes[0,0].tick_params(axis='both', labelsize=12)
    axes[0,0].axvline(media, color='red', linestyle='--', label=f"Media={media:.2f}")
    axes[0,0].axvline(mediana, color='green', linestyle='--', label=f"Mediana={mediana:.2f}")
    axes[0,0].legend()

    
    # --- Box Plot ---
    sns.boxplot(x=data, ax=axes[0,1], color="lightgreen")
    axes[0,1].set_title("Diagrama de caja", fontsize=16)
    axes[0,1].set_xlabel("Magnitud (Mag_mbLgL)", fontsize=14)
    axes[0,1].tick_params(axis='both', labelsize=12)
    axes[0,1].text(0.05, 0.95, f"Min={minimo:.2f}\nMax={maximo:.2f}\nMedia={media:.2f}\nMediana={mediana:.2f}\nStd={std:.2f}", 
                   transform=axes[0,1].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # --- Q-Q Plot ---
    stats.probplot(data, dist="norm", plot=axes[1,0])
    axes[1,0].set_title("Diagrama Q-Q (vs Normal)", fontsize=16)
    axes[1,0].set_xlabel("Valores teóricos", fontsize=14)
    axes[1,0].set_ylabel("Valores ordenados", fontsize=14)
    axes[1,0].tick_params(axis='both', labelsize=12)
    axes[1,0].text(0.05, 0.95, f"Media={media:.2f}\nStd={std:.2f}", transform=axes[1,0].transAxes,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # --- Violin Plot ---
    sns.violinplot(x=data, ax=axes[1,1], color="lightcoral")
    axes[1,1].set_title("Diagrama de violin", fontsize=16)
    axes[1,1].set_xlabel("Magnitud (Mag_mbLgL)", fontsize=14)
    axes[1,1].tick_params(axis='both', labelsize=12)
    axes[1,1].text(0.05, 0.95, f"Media={media:.2f}\nMediana={mediana:.2f}\nStd={std:.2f}",
                    transform=axes[1,1].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Guardar o mostrar
    if output_file:
        import os
        folder = os.path.dirname(output_file)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{output_file}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



#######################################################
# Función para representar una de las columnas 
# del Dataframe
#######################################################

import matplotlib.pyplot as plt
import pandas as pd

def plot_evolucion_multi(df, fecha_col="FechaHora", param_cols=None, 
                         figsize=(12, 5), output_file=None, titulo=None):
    """
    Representa la evolución temporal de una o varias columnas de un DataFrame.

    Parámetros
    ----------
    df : DataFrame
        Datos con al menos una columna de fechas y las de parámetros.
    fecha_col : str
        Nombre de la columna con fechas.
    param_cols : list[str]
        Lista de nombres de columnas con los parámetros a graficar.
    titulo : str, opcional
        Título del gráfico.
    ylabel : str, opcional
        Etiqueta del eje Y.
    figsize : tuple
        Tamaño de la figura.
    """
    if param_cols is None:
        raise ValueError("Debes indicar al menos una columna en param_cols")

    fechas = pd.to_datetime(df[fecha_col])

    plt.figure(figsize=figsize)
    for col in param_cols:
        plt.plot(fechas, df[col], marker="o", markersize=2, linestyle="-", label=col)

    plt.xlabel("Fecha", fontsize=14)
    plt.ylabel("Valor", fontsize=14)
    plt.title(f"{titulo}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()


    # Guardar 
    if output_file:
        import os
        folder = os.path.dirname(output_file)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{output_file}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches="tight")

    
    plt.show()


# EJEMPLO DE USO
# Comparar evolución de b_lsq y b_mlk
# plot_evolucion_multi(
#    gdf_2002, 
#    fecha_col="FechaHora", 
#    param_cols=["b_lsq", "b_mlk"], 
#    titulo="Evolución comparativa de b-values"
# )
