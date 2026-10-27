"""
Script de prueba manual: pide los atributos por consola y muestra la predicción.
Útil para validar el modelo entrenado (tuned_model.pkl).
"""

import pandas as pd
import joblib

from config.path import TUNED_MODEL_PATH

def test(input_data: dict):
    """Leyendo el modelo"""
    # Cargar modelo
    try:
        model = joblib.load(TUNED_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Convertir input a DataFrame
    df_input = pd.DataFrame([input_data])

    # Predicción
    prediction = model.predict(df_input)[0]

    # Probabilidad de default
    probability = model.predict_proba(df_input)[0]
    p_default = probability[1]

    print("prediction value:", prediction)

    print(
        "⚠️ Default"
        if prediction == 1
        else "✅ No Default"
    )

    # Nivel de riesgo
    if p_default >= 0.65:
        print(f"⚠️ Risk: High ({p_default:.4f})")
    elif p_default >= 0.45:
        print(f"⚠️ Risk: Medium ({p_default:.4f})")
    elif p_default >= 0.25:
        print(f"⚠️ Risk: Low ({p_default:.4f})")
    else:
        print(f"⚠️ Risk: Very Low ({p_default:.4f})")

    return probability, prediction

if __name__ == "__main__":
    print("=" * 60)
    print("Credit Risk Prediction App")
    print("=" * 60)
    print("Ingrese valores válidos del cliente:")
    print("-" * 60)

    # Datos del cliente
    edad = int(input("Edad: "))
    ingresos_mensuales = float(input("Ingresos mensuales: "))
    score_buro = float(input("Score de buró: "))
    estado_civil = input("Estado civil: ")
    nivel_educativo = input( "Nivel educativo: ")

    tipo_empleo = input("Tipo de empleo: ")

    antiguedad_laboral_meses = int(input("Antigüedad laboral (meses): "))

    deudas_actuales = float(input("Deudas actuales: "))

    monto_solicitado = float(input("Monto solicitado: "))

    cuota_mensual_estimada = float(input("Cuota mensual estimada: "))

    historial_pagos_tardios = int(input("Historial de pagos tardíos: ") )

    num_creditos_activos = int(input("Número de créditos activos: "))

    num_dependientes = int(input("Número de dependientes: "))

    plazo_meses = int(input("Plazo del crédito (meses): "))

    ratio_deuda_ingreso = float(input("Ratio deuda/ingreso: "))

    # Derived features
    antiguedad_laboral_anios = round(
        antiguedad_laboral_meses / 12,
        2
    )

    monto_solicitado_ingreso = round(
        monto_solicitado / ingresos_mensuales,
        3
    )

    cuota_ingreso = round(
        cuota_mensual_estimada / ingresos_mensuales,
        3
    )

    tiene_pagos_tardios = (
        1 if historial_pagos_tardios > 0 else 0
    )

    # Construir input
    input_record = {

        "edad": edad,

        "ingresos_mensuales":
            ingresos_mensuales,

        "score_buro":
            score_buro,

        "estado_civil":
            estado_civil,

        "nivel_educativo":
            nivel_educativo,

        "tipo_empleo":
            tipo_empleo,

        "antiguedad_laboral_meses":
            antiguedad_laboral_meses,

        "deudas_actuales":
            deudas_actuales,

        "monto_solicitado":
            monto_solicitado,

        "cuota_mensual_estimada":
            cuota_mensual_estimada,

        "historial_pagos_tardios":
            historial_pagos_tardios,


        "ratio_deuda_ingreso":
            ratio_deuda_ingreso,

        "num_creditos_activos":
            num_creditos_activos,

        "num_dependientes":
            num_dependientes,

        "plazo_meses":
            plazo_meses,


        "antiguedad_laboral_anios":
            antiguedad_laboral_anios,

        "monto_solicitado_ingreso":
            monto_solicitado_ingreso,

        "cuota_ingreso":
            cuota_ingreso,

        "tiene_pagos_tardios":
            tiene_pagos_tardios
    }

    # Ejecutar predicción
    test(input_record)