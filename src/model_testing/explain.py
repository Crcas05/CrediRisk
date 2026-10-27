"""
Explicabilidad del modelo XGBoost con SHAP.

Explica:
- qué variables son más importantes globalmente
- qué variables influyeron en la predicción de un cliente
"""

import numpy as np
import pandas as pd
import joblib
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.path import (PREPROCESSED_TRAIN_PATH,PREPROCESSED_TEST_PATH,MODEL_PATH,ARTIFACT_DIR)

def get_feature_names(pipeline, original_columns):
    """Obtiene los nombres de features después del preprocessing."""
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out(
        original_columns
    )

    return [name.split("__", 1)[-1] for name in feature_names]


def build_explainer(pipeline, X_background, feature_names):
    """Crea un SHAP TreeExplainer para XGBoost."""
    classifier = pipeline.named_steps["classifier"]

    return shap.TreeExplainer(classifier)


def explain_global(pipeline, explainer, X_sample, feature_names, top_n=10):
    """Muestra las features con mayor impacto promedio."""

    preprocessor = pipeline.named_steps["preprocessor"]
    sample = preprocessor.transform(X_sample)

    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)

    ranking = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    print("=== Importancia global de features ===")

    for name, value in ranking:
        print(f"{name:<35} {value:.4f}")

    plt.figure(figsize=(9, 6))

    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plot_path = ARTIFACT_DIR / "shap_summary.png"

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\nGráfico guardado en: {plot_path}")


def explain_instance(pipeline, explainer, feature_names, record, top_n=5):
    """Explica la predicción de un cliente concreto."""

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    df_input = pd.DataFrame([record])
    transformed = preprocessor.transform(df_input)

    p_default = classifier.predict_proba(transformed)[0][1]

    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values[0]

    base_value = explainer.expected_value

    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    contributions = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    print("\n=== Por qué el modelo predijo esto ===")
    print(f"Probabilidad de default: {p_default:.4f}")
    print(f"Valor base: {base_value:.4f}")

    for name, value in contributions:
        direction = (
            "aumenta el riesgo de incumplimiento"
            if value > 0
            else "reduce el riesgo de incumplimiento"
        )

        print(
            f"  {name:<35} "
            f"{value:+.4f} -> {direction}"
        )


if __name__ == "__main__":

    pipeline = joblib.load(MODEL_PATH)

    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    X_train = df_train.drop(
        columns=["default", "cliente_id"]
    )

    X_test = df_test.drop(
        columns=["default", "cliente_id"]
    )

    feature_names = get_feature_names(
        pipeline,
        X_train.columns.tolist()
    )

    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42
    )

    explainer = build_explainer(
        pipeline,
        background,
        feature_names
    )

    explain_global(
        pipeline,
        explainer,
        X_test,
        feature_names
    )

    sample_record = X_test.iloc[0].to_dict()

    explain_instance(
        pipeline,
        explainer,
        feature_names,
        sample_record
    )