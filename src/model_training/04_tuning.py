
"""
Ajuste de hiperparámetros de XGBoost y registro en MLflow.
Proyecto: Riesgo Crediticio.
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score

from config.path import (
    PREPROCESSED_TRAIN_PATH,
    PREPROCESSED_TEST_PATH,
    MODEL_PATH,
    TUNED_MODEL_PATH,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

TRACKING_URI = "http://localhost:5555"
EXPERIMENT_NAME = "Riesgo Crediticio - XGBoost"

# Registro nuevo y limpio: su primera versión será v1.
REGISTERED_MODEL_NAME = "riesgo_crediticio_xgboost_v2"

AUTHOR = "Cristhian Castro"
PROJECT_NAME = "Riesgo Crediticio"
ALGORITHM = "XGBoost"
MODEL_TYPE = "XGBClassifier"


# ============================================================
# CONFIGURAR MLFLOW
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=TRACKING_URI)


# ============================================================
# AJUSTE DE HIPERPARÁMETROS
# ============================================================

def tuning_data(X_train, y_train, X_test, y_test):
    """
    Ajusta los hiperparámetros, evalúa el mejor modelo
    y registra el resultado en MLflow.
    """

    # --------------------------------------------------------
    # Cargar el modelo base
    # --------------------------------------------------------

    base_pipeline = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Definir los hiperparámetros
    # --------------------------------------------------------

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    # --------------------------------------------------------
    # Entrenar y seleccionar el mejor modelo
    # --------------------------------------------------------

    print("\nIniciando ajuste de hiperparámetros...")

    grid.fit(X_train, y_train)

    tuned_pipeline = grid.best_estimator_

    # --------------------------------------------------------
    # Evaluar el modelo
    # --------------------------------------------------------

    y_pred = tuned_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Guardar el modelo local para la API
    # --------------------------------------------------------

    TUNED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(tuned_pipeline, TUNED_MODEL_PATH)

    # --------------------------------------------------------
    # Registrar experimento y modelo en MLflow
    # --------------------------------------------------------

    with mlflow.start_run(run_name="tuning_xgboost") as run:

        run_id = run.info.run_id

        # Tags de la ejecución
        mlflow.set_tags({
            "autor": AUTHOR,
            "proyecto": PROJECT_NAME,
            "algoritmo": ALGORITHM,
            "tipo_modelo": MODEL_TYPE,
        })

        # Parámetros
        mlflow.log_param("modelo_base", str(MODEL_PATH))
        mlflow.log_param("scoring", "recall")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_params(grid.best_params_)

        # Métricas
        mlflow.log_metric(
            "best_recall_cv",
            float(grid.best_score_),
        )
        mlflow.log_metric(
            "test_accuracy",
            float(accuracy),
        )
        mlflow.log_metric(
            "test_recall",
            float(recall),
        )

        # ----------------------------------------------------
        # Registrar el modelo en el registro nuevo
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=tuned_pipeline,
            name="tuned_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )

        # Registrar el archivo local como artefacto
        mlflow.log_artifact(str(TUNED_MODEL_PATH))

        # ----------------------------------------------------
        # Descripción automática del modelo
        # ----------------------------------------------------

        description = f"""
Modelo de predicción de riesgo crediticio.

Autor: {AUTHOR}
Proyecto: {PROJECT_NAME}
Algoritmo: {ALGORITHM}
Tipo de modelo: {MODEL_TYPE}

Objetivo: predecir el incumplimiento crediticio (default).

Método de optimización: GridSearchCV.
Validación: StratifiedKFold con 5 particiones.
Métrica de optimización: recall.

Mejores hiperparámetros:
{grid.best_params_}

Resultados:
- Recall de validación cruzada: {grid.best_score_:.4f}
- Accuracy de prueba: {accuracy:.4f}
- Recall de prueba: {recall:.4f}

Ruta local del modelo: {TUNED_MODEL_PATH}
Ejecución MLflow: {run_id}
""".strip()

        # ----------------------------------------------------
        # Identificar únicamente la versión de esta ejecución
        # ----------------------------------------------------

        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        matching_versions = [
            version
            for version in versions
            if version.run_id == run_id
        ]

        if not matching_versions:
            raise RuntimeError(
                "No se encontró la versión del modelo "
                "correspondiente a la ejecución actual."
            )

        current_version = max(
            matching_versions,
            key=lambda version: int(version.version),
        )

        # Descripción de la versión registrada
        client.update_model_version(
            name=REGISTERED_MODEL_NAME,
            version=current_version.version,
            description=description,
        )

        # ----------------------------------------------------
        # Tags descriptivos como los del profesor
        # ----------------------------------------------------

        version_tags = {
            "autor": AUTHOR,
            "proyecto": PROJECT_NAME,
            "algoritmo": ALGORITHM,
            "model_type": MODEL_TYPE,
            "hyperparameters": str(grid.best_params_),
            "model_path": "inference/artifacts/tuned_model.pkl",
            "scoring": "recall",
            "cv_folds": "5",
        }

        for key, value in version_tags.items():
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=current_version.version,
                key=key,
                value=value,
            )

        # ----------------------------------------------------
        # Resumen final
        # ----------------------------------------------------

        print("\n" + "=" * 55)
        print("RESULTADOS DEL AJUSTE")
        print("=" * 55)

        print(f"Autor: {AUTHOR}")
        print(f"Experimento: {EXPERIMENT_NAME}")
        print(f"Algoritmo: {ALGORITHM}")
        print(f"Mejores parámetros: {grid.best_params_}")
        print(f"Recall CV: {grid.best_score_:.4f}")
        print(f"Accuracy de prueba: {accuracy:.4f}")
        print(f"Recall de prueba: {recall:.4f}")
        print(f"Ruta del modelo: {TUNED_MODEL_PATH}")
        print(f"Registro: {REGISTERED_MODEL_NAME}")
        print(f"Versión: {current_version.version}")
        print(f"Run ID: {run_id}")
        print("Descripción y Tags registrados correctamente.")
        print("=" * 55)

    return tuned_pipeline


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("Cargando datos preparados...")

    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    # Separar variables predictoras y objetivo
    X_train = df_train.drop(
        columns=["default", "cliente_id"]
    )
    y_train = df_train["default"]

    X_test = df_test.drop(
        columns=["default", "cliente_id"]
    )
    y_test = df_test["default"]

    print(f"Filas de entrenamiento: {len(X_train)}")
    print(f"Filas de prueba: {len(X_test)}")

    tuning_data(
        X_train,
        y_train,
        X_test,
        y_test,
    )

