
"""
Entrenamiento de XGBoost con seguimiento de experimentos en MLflow.
Predice el default de nuevos clientes.
"""
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from config.path import PREPROCESSED_TRAIN_PATH, MODEL_PATH

# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://localhost:5555"
EXPERIMENT_NAME = "Riesgo Crediticio - XGBoost"
REGISTERED_MODEL_NAME = "riesgo_crediticio_xgboost"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def train_data(df):
    df_train = df.copy()

    # Separar variables predictoras y objetivo
    X = df_train.drop(columns=["default", "cliente_id"])
    Y = df_train["default"]

    # Variables categóricas
    CATEGORICAL_COLS = [
        "estado_civil",
        "nivel_educativo",
        "tipo_empleo",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLS,
            )
        ],
        remainder="passthrough",
    )

    # Hiperparámetros de XGBoost
    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Entrenar y registrar en MLflow
    with mlflow.start_run(run_name="entrenamiento_xgboost"):
        mlflow.log_param("modelo", "XGBoost")
        mlflow.log_params(params)
        mlflow.log_param("n_filas_entrenamiento", len(X))
        mlflow.log_param("n_variables_predictoras", X.shape[1])

        pipeline.fit(X, Y)

        # Conservar el modelo local para la API existente
        joblib.dump(pipeline, MODEL_PATH)

        # Registrar el pipeline entrenado y crear una versión
        # en MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="modelo_xgboost",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X.head(5),
            serialization_format="cloudpickle"
        )

        print("✅ Entrenamiento y registro en MLflow completados.")
        print(f"📥 Datos: {PREPROCESSED_TRAIN_PATH}")
        print(f"💾 Modelo local: {MODEL_PATH}")
        print(f"🧪 Experimento: {EXPERIMENT_NAME}")
        print(f"📦 Modelo registrado: {REGISTERED_MODEL_NAME}")

    return True


if __name__ == "__main__":
    df = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    train_data(df)
