"""
En esta etapa observamos que tenemos:
- variables categoricas
- vaiables numericas
- variable objetivo (Targed)
El objetivo es preparar las variables predictoras (X) y la variable objeto (Y) en un formato
adecuado para el entrenamiento del modelo ML
- Defalut --> 0 , 1
"""
import pandas as pd
from sklearn.model_selection import train_test_split 
# Herramienta de Scikit-learn que divide los datos en entrenamiento y prueba.

from config.path import FEATURED_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH

def preprocess_data(df):   
    # Copiar el DataFrame original
    df_pp = df.copy()

    # Separo mis Variables X e Y
    X = df_pp.drop(columns=["default"])      # Toma el DataFrame y elimina la columna default 
    Y = df_pp["default"]                # Toma la columa default y guardalo en Y

    ## DIVISION DE TRAIN / TEST
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.20,         # 20% van a test y el -80% van a entrenamiento
        random_state=42,        # Obtenemos la misma particion al correr el modelo
        stratify=Y              # mantiene la proporcion default en ambos 
    )

    # Reiniciar indices
    X_train = X_train.reset_index(drop=True) # Elimina el indice anteior despues de reiniciar los indices
    X_test = X_test.reset_index(drop=True)

    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    # Unir Predictoras (X) + OBJETIVO (Y)
    train_df = pd.concat([X_train, Y_train], axis=1)    # Uni x_train y Y_train horizontalmente como columnas
    test_df =  pd.concat([X_test, Y_test], axis=1)

    # Guardamos el datasets
    train_df.to_csv(PREPROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PREPROCESSED_TEST_PATH, index=False)

    # Analizar la Salida
    print("✅ Preprocessing completado.")
    print(f"📥 Entrada: {FEATURED_PATH}")
    print(f"📤 Train: {PREPROCESSED_TRAIN_PATH}")
    print(f"📤 Test: {PREPROCESSED_TEST_PATH}")

    print(f"📊 Train: {train_df.shape[0]} filas")
    print(f"📊 Test: {test_df.shape[0]} filas")

    print("\nDistribución del Target:")
    print(train_df["default"].value_counts(normalize=True))


    return train_df, test_df


if __name__ == "__main__":

    df = pd.read_csv(FEATURED_PATH)

    preprocess_data(df)