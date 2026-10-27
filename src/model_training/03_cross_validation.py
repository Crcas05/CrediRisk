"""
Validacion Cruzada 
- Analisis que modelo es estable entrenado
"""
import pandas as pd
import joblib

from sklearn.model_selection import cross_val_score, StratifiedKFold
from config.path import PREPROCESSED_TRAIN_PATH, MODEL_PATH

def cv_data(X_train, Y_train):
    # Cargamos la Data entrenada
    pipeline = joblib.load(MODEL_PATH)

    # Creamos la validacion cruzada
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Ejecutamos validacion cruzada
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        Y_train,
        cv=strat_cv,
        scoring="recall"
    )

    print("Cross Validation Recall:")
    print(cv_scores)
    print(f"Recall promedio: {cv_scores.mean() * 100:.2f}%")
    return(cv_scores)

if __name__ == "__main__":

    # Cargar datos de entrenamiento
    df_train = pd.read_csv( PREPROCESSED_TRAIN_PATH)
    # Separar variables
    X_train = df_train.drop(columns=["default", "cliente_id"])
    Y_train = df_train["default"]

    #Ejecutar Cross 
    cv_data(X_train, Y_train)

    