"""
Carga el modelo entrenado (tuned_model.pkl) y expone el método predict()
para la API.

El modelo contiene el preprocesamiento y XGBoost dentro de un Pipeline.
"""

import pandas as pd
import joblib
from pathlib import Path


FEATURE_ORDER = [
    "edad",
    "ingresos_mensuales",
    "score_buro",
    "estado_civil",
    "nivel_educativo",
    "tipo_empleo",
    "antiguedad_laboral_meses",
    "deudas_actuales",
    "monto_solicitado",
    "cuota_mensual_estimada",
    "historial_pagos_tardios",
    "ratio_deuda_ingreso",
    "num_creditos_activos",
    "num_dependientes",
    "plazo_meses",
    "antiguedad_laboral_anios",
    "monto_solicitado_ingreso",
    "cuota_ingreso",
    "tiene_pagos_tardios"
]


THRESHOLD = 0.50


class Predictor:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parent.parent
            / "artifacts"
            / "tuned_model.pkl"
        )

        try:
            model = joblib.load(model_path)

            self.predict_fn = model.predict
            self.predict_proba_fn = model.predict_proba

        except FileNotFoundError:
            raise RuntimeError(
                f"Modelo no encontrado en {model_path}."
            )

        except AttributeError:
            raise RuntimeError(
                f"El modelo cargado desde {model_path} "
                "no contiene los métodos necesarios."
            )

    def is_loaded(self) -> bool:
        return self.predict_fn is not None

    def predict(self, features: dict) -> dict:

        df = pd.DataFrame(
            [[features[k] for k in FEATURE_ORDER]],
            columns=FEATURE_ORDER
        )

        prediction = self.predict_fn(df)[0]

        probs = self.predict_proba_fn(df)[0]

        p_no_default = float(probs[0])
        p_default = float(probs[1])

        return {
            "prediction": int(p_default >= THRESHOLD),
            "p_default": round(p_default, 4),
            "p_no_default": round(p_no_default, 4),
            "risk": self._tier(p_default),
            "threshold": THRESHOLD
        }

    def _tier(self, prob: float) -> str:

        if prob >= 0.65:
            return "HIGH"

        if prob >= 0.45:
            return "MEDIUM"

        if prob >= 0.25:
            return "LOW"

        return "VERY_LOW"