
"""
Evaluación de XGBoost con registro de métricas y artefactos en MLflow.
"""
import pandas as pd
import joblib
import mlflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from config.path import (
    PREPROCESSED_TRAIN_PATH,
    PREPROCESSED_TEST_PATH,
    MODEL_PATH,
    ARTIFACT_DIR,
)

MLFLOW_TRACKING_URI = "http://localhost:5555"
EXPERIMENT_NAME = "Riesgo Crediticio - XGBoost"


def evaluate_data(X_train, Y_train, X_test, Y_test):
    # Conectar con MLflow y seleccionar el experimento
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Cargar el modelo entrenado
    pipeline = joblib.load(MODEL_PATH)

    # Realizar predicciones
    Y_pred = pipeline.predict(X_test)
    Y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calcular métricas reales
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    roc_auc = roc_auc_score(Y_test, Y_prob)

    train_score = pipeline.score(X_train, Y_train)
    test_score = pipeline.score(X_test, Y_test)

    conf_matrix = confusion_matrix(Y_test, Y_pred)
    class_report = classification_report(
        Y_test, Y_pred, zero_division=0
    )

    # Mostrar resultados en la terminal
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Train Score: {train_score * 100:.2f}%")
    print(f"Test Score: {test_score * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Calcular importancia de variables
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out(X_train.columns)
    )
    features_importance = (
        pipeline.named_steps["classifier"].feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": features_importance,
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    # Generar gráfico de importancia
    importance_path = ARTIFACT_DIR / "feature_importance.png"

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(importance_path)
    plt.close()

    # Generar matriz de confusión
    confusion_path = ARTIFACT_DIR / "confusion_matrix.png"

    ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred)
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_path)
    plt.close()

    # Generar curva ROC
    roc_path = ARTIFACT_DIR / "roc_curve.png"

    RocCurveDisplay.from_predictions(Y_test, Y_prob)
    plt.title("XGBoost ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Registrar métricas y artefactos en una ejecución de evaluación
    with mlflow.start_run(run_name="evaluacion_xgboost"):
        mlflow.log_metrics({
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "train_score": float(train_score),
            "test_score": float(test_score),
        })

        # Guardar informe de clasificación
        report_path = ARTIFACT_DIR / "classification_report.txt"
        report_path.write_text(class_report, encoding="utf-8")

        # Guardar matriz de confusión en CSV
        matrix_path = ARTIFACT_DIR / "confusion_matrix.csv"
        pd.DataFrame(conf_matrix).to_csv(matrix_path, index=False)

        # Subir los archivos generados a MLflow
        mlflow.log_artifact(str(importance_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(matrix_path))

        print("\n✅ Métricas y artefactos registrados en MLflow.")
        print("Run ID:", mlflow.active_run().info.run_id)

    return recall


if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    X_train = df_train.drop(columns=["default", "cliente_id"])
    Y_train = df_train["default"]

    X_test = df_test.drop(columns=["default", "cliente_id"])
    Y_test = df_test["default"]

    evaluate_data(X_train, Y_train, X_test, Y_test)
