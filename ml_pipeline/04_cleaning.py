"""
Limpieza Basica: duplicados - nulos
"""

import pandas as pd
from config.path import EDA_PATH, CLEANING_PATH

def clean_data(df):

    # Buscar Duplicados
    print("--- Find Duplicates----")
    df_duplicate = df.duplicated()                                  # detectar que filas estas duplicados
    print(f"Number of duplicate rows: {df_duplicate.sum()}")

    # Buscar valores nulos
    print("--- Find Missing/Null values ---")
    missing_values = df.isnull().sum()  # cantidad de nulos por columna
    print(f"Missing values: {missing_values}")

    # Guarda el Dataset Limpio
    df.to_csv(CLEANING_PATH, index=False)
    print("Cleaned dataset saved successfully.")
    return df

if __name__ == "__main__":
    df = pd.read_csv(EDA_PATH)
    clean_data(df)
