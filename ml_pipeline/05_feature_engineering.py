"""
Creamos nueva variables a partir de las variables existente
Python lee de derecha a izquierda
columna ya existent es lado derecho: antiguedad_laboral_meses
columa nueva o la que crea python es: antiguedad_laboral_anios
"""
import pandas as pd
from config.path import CLEANING_PATH, FEATURED_PATH

def feature_data(df):
    #Copiamos el DataFrame original
    df_fe = df.copy()

 # -----------------------------------------
    # 1. Antigüedad laboral en años
    # -----------------------------------------

    df_fe["antiguedad_laboral_anios"] = (
        df_fe["antiguedad_laboral_meses"] / 12
    ).round(2)


    # -----------------------------------------
    # 2. Monto solicitado respecto al ingreso
    # -----------------------------------------

    df_fe["monto_solicitado_ingreso"] = (
        df_fe["monto_solicitado"] /
        df_fe["ingresos_mensuales"]
    ).round(3)


    # -----------------------------------------
    # 3. Cuota respecto al ingreso
    # -----------------------------------------

    df_fe["cuota_ingreso"] = (
        df_fe["cuota_mensual_estimada"] /
        df_fe["ingresos_mensuales"]
    ).round(3)


    # -----------------------------------------
    # 4. Tiene historial de pagos tardíos
    # -----------------------------------------

    df_fe["tiene_pagos_tardios"] = (
        df_fe["historial_pagos_tardios"] > 0
    ).astype(int)                                   # Convierte de booleano a nro --> True a 1


    # -----------------------------------------
    # Guardar resultado
    # -----------------------------------------

    df_fe.to_csv(FEATURED_PATH, index=False)

    print("✅ Feature Engineering completado.")
    print(f"📥 Entrada: {CLEANING_PATH}")
    print(f"📤 Salida: {FEATURED_PATH}")
    print(f"📊 Filas: {df_fe.shape[0]}")
    print(f"📊 Columnas: {df_fe.shape[1]}")

    return df_fe


if __name__ == "__main__":

    df = pd.read_csv(CLEANING_PATH)

    feature_data(df)