
import os

import requests
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(
    page_title="CrediRisk | Evaluación crediticia",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. ESTILOS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f4f7fb;
            color: #172b4d;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        .brand {
            font-size: 1.1rem;
            font-weight: 800;
            color: #12366b;
        }

        .hero {
            background: linear-gradient(120deg, #102e5a, #1b4d87);
            color: white;
            padding: 30px;
            border-radius: 18px;
            margin: 18px 0 24px 0;
        }

        .hero h1 {
            color: white;
            font-size: 2rem;
            margin-bottom: 8px;
        }

        .hero p {
            color: #dbeafe;
            margin-bottom: 0;
        }

        .section-title {
            color: #173b6c;
            font-size: 1.12rem;
            font-weight: 750;
            margin-bottom: 4px;
        }

        .section-subtitle {
            color: #64748b;
            font-size: .88rem;
            margin-bottom: 18px;
        }

        div[data-testid="stForm"] {
            background: white;
            padding: 22px;
            border: 1px solid #e4eaf2;
            border-radius: 16px;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: #174b88;
            color: white;
            border: 0;
            border-radius: 9px;
            min-height: 46px;
            font-weight: 700;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #103663;
            color: white;
        }

        div[data-testid="stMetric"] {
            background: white;
            padding: 18px;
            border: 1px solid #e4eaf2;
            border-radius: 14px;
        }

        .footnote {
            color: #64748b;
            font-size: .82rem;
            border-top: 1px solid #dce4ee;
            padding-top: 16px;
            margin-top: 28px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. ENCABEZADO
# ============================================================

header1, header2 = st.columns([3, 1])

with header1:
    st.markdown(
        '<div class="brand">◆ CREDIRISK | INTELIGENCIA CREDITICIA</div>',
        unsafe_allow_html=True,
    )

with header2:
    st.caption("PROTOTIPO DE EVALUACIÓN")

st.markdown(
    """
    <div class="hero">
        <h1>Evaluación de riesgo crediticio</h1>
        <p>
            Analiza el perfil financiero del solicitante mediante
            un modelo predictivo de Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Completa la información conocida del solicitante. "
    "Los resultados son estimaciones del modelo y no representan "
    "una decisión definitiva de crédito."
)


# ============================================================
# 4. CATEGORÍAS ESPERADAS POR EL MODELO
# ============================================================

ESTADOS = {
    "Soltero/a": "soltero",
    "Casado/a": "casado",
    "Divorciado/a": "divorciado",
    "Viudo/a": "viudo",
}

EDUCACION = {
    "Primaria": "primaria",
    "Secundaria": "secundaria",
    "Técnico": "tecnico",
    "Universitario": "universitario",
    "Posgrado": "posgrado",
}

EMPLEOS = {
    "Trabajador dependiente": "dependiente",
    "Trabajador independiente": "independiente",
    "Desempleado/a": "desempleado",
    "Jubilado/a": "jubilado",
}


# ============================================================
# 5. FORMULARIO DE EVALUACIÓN
# ============================================================

with st.form("credit_form"):

    # --------------------------------------------------------
    # Perfil personal
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">01 · Perfil del solicitante</div>'
        '<div class="section-subtitle">'
        "Información personal y familiar"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.number_input(
            "Edad (años)",
            min_value=18,
            max_value=100,
            value=33,
            step=1,
        )

        estado_label = st.selectbox(
            "Estado civil",
            options=list(ESTADOS.keys()),
        )

    with col2:
        educacion_label = st.selectbox(
            "Nivel educativo",
            options=list(EDUCACION.keys()),
            index=3,
        )

        dependientes = st.number_input(
            "Dependientes económicos",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

    with col3:
        empleo_label = st.selectbox(
            "Situación laboral",
            options=list(EMPLEOS.keys()),
        )

        antiguedad = st.number_input(
            "Antigüedad laboral (meses)",
            min_value=0,
            max_value=600,
            value=36,
            step=1,
        )

    st.divider()

    # --------------------------------------------------------
    # Capacidad financiera
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">02 · Capacidad financiera</div>'
        '<div class="section-subtitle">'
        "Ingresos y obligaciones económicas"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        ingresos = st.number_input(
            "Ingreso mensual (S/)",
            min_value=0.01,
            value=3599.0,
            step=100.0,
        )

    with col2:
        deudas = st.number_input(
            "Deudas actuales (S/)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )

    with col3:
        ratio = st.number_input(
            "Ratio deuda/ingreso",
            min_value=0.0,
            value=0.30,
            step=0.05,
            help=(
                "Introduce el ratio en la escala utilizada "
                "durante el entrenamiento."
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Historial crediticio
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">03 · Historial crediticio</div>'
        '<div class="section-subtitle">'
        "Indicadores registrados en el dataset"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        score = st.number_input(
            "Score de buró",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help=(
                "Verifica la escala de este indicador "
                "antes de usar datos reales."
            ),
        )

    with col2:
        pagos_tardios = st.number_input(
            "Pagos tardíos",
            min_value=0,
            value=0,
            step=1,
        )

    with col3:
        creditos = st.number_input(
            "Créditos activos",
            min_value=0,
            value=1,
            step=1,
        )

    st.divider()

    # --------------------------------------------------------
    # Condiciones del crédito
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">04 · Condiciones del crédito</div>'
        '<div class="section-subtitle">'
        "Características de la solicitud"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        monto = st.number_input(
            "Monto solicitado (S/)",
            min_value=0.01,
            value=5000.0,
            step=500.0,
        )

    with col2:
        cuota = st.number_input(
            "Cuota mensual estimada (S/)",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )

    with col3:
        plazo = st.number_input(
            "Plazo (meses)",
            min_value=1,
            value=24,
            step=1,
        )

    submitted = st.form_submit_button(
        "Evaluar perfil crediticio",
        use_container_width=True,
    )


# ============================================================
# 6. CONSTRUIR EL PAYLOAD PARA FASTAPI
# ============================================================

if submitted:

    if ingresos <= 0:
        st.error("El ingreso mensual debe ser mayor que cero.")

    elif cuota > ingresos:
        st.warning(
            "La cuota mensual supera los ingresos declarados. "
            "Verifica los datos antes de continuar."
        )

    else:
        payload = {
            "edad": int(edad),
            "estado_civil": ESTADOS[estado_label],
            "nivel_educativo": EDUCACION[educacion_label],
            "num_dependientes": int(dependientes),
            "tipo_empleo": EMPLEOS[empleo_label],
            "antiguedad_laboral_meses": int(antiguedad),
            "ingresos_mensuales": float(ingresos),
            "score_buro": float(score),
            "num_creditos_activos": int(creditos),
            "historial_pagos_tardios": int(pagos_tardios),
            "deudas_actuales": float(deudas),
            "monto_solicitado": float(monto),
            "plazo_meses": int(plazo),
            "cuota_mensual_estimada": float(cuota),
            "ratio_deuda_ingreso": float(ratio),
        }

        # ====================================================
        # 7. COMUNICACIÓN CON EL BACKEND
        # ====================================================

        try:
            with st.spinner(
                "Conectando con FastAPI y evaluando el perfil..."
            ):

                st.caption(f"API configurada: {PREDICT_URL}")

                response = requests.post(
                    PREDICT_URL,
                    json=payload,
                    timeout=30,
                )

                response.raise_for_status()
                result = response.json()

            # ================================================
            # 8. MOSTRAR RESULTADO
            # ================================================

            st.divider()
            st.markdown("## Resultado de la evaluación")

            risk_labels = {
                "VERY_LOW": "Muy bajo",
                "LOW": "Bajo",
                "MEDIUM": "Medio",
                "HIGH": "Alto",
            }

            risk_code = result.get("risk", "UNKNOWN")
            risk_label = risk_labels.get(risk_code, risk_code)

            p_default = float(result["p_default"])
            p_no_default = float(result["p_no_default"])
            prediction = int(result["prediction"])
            threshold = float(result["threshold"])

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Nivel de riesgo",
                risk_label,
            )

            metric2.metric(
                "Probabilidad de incumplimiento",
                f"{p_default:.2%}",
            )

            metric3.metric(
                "Probabilidad de no incumplimiento",
                f"{p_no_default:.2%}",
            )

            st.progress(
                min(max(p_default, 0.0), 1.0),
                text="Probabilidad estimada de incumplimiento",
            )

            if prediction == 1:
                st.error(
                    "La predicción alcanza o supera el umbral "
                    "de clasificación del modelo."
                )
            else:
                st.success(
                    "La predicción está por debajo del umbral "
                    "de clasificación del modelo."
                )

            st.caption(
                f"Umbral de clasificación: {threshold:.0%}"
            )

            with st.expander("Ver resumen de los datos enviados"):
                st.json(payload)

            with st.expander("Ver respuesta JSON de FastAPI"):
                st.json(result)

            st.warning(
                "Este resultado es una estimación estadística. "
                "No constituye una aprobación ni una denegación "
                "definitiva de crédito."
            )

        # ====================================================
        # 9. MANEJO DE ERRORES
        # ====================================================

        except requests.exceptions.ConnectionError:
            st.error(
                "No se pudo conectar con FastAPI. "
                "Comprueba que el contenedor del backend esté activo "
                "y que API_URL tenga el valor correcto."
            )

        except requests.exceptions.Timeout:
            st.error(
                "FastAPI no respondió dentro del tiempo establecido."
            )

        except requests.exceptions.HTTPError as exc:
            st.error(f"FastAPI devolvió un error HTTP: {exc}")

            if response is not None:
                st.code(response.text)

        except (KeyError, ValueError, TypeError) as exc:
            st.error(
                f"No se pudo interpretar la respuesta de FastAPI: {exc}"
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Error durante la comunicación con FastAPI: {exc}")


# ============================================================
# 10. PIE DE PÁGINA
# ============================================================

st.markdown(
    """
    <div class="footnote">
        CrediRisk · Prototipo académico de Machine Learning.
        Utiliza datos autorizados y verifica las escalas de las
        variables antes de utilizar el sistema en escenarios reales.
    </div>
    """,
    unsafe_allow_html=True,
)
