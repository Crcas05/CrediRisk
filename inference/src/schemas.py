"""
Esquemas Pydantic: definen y validan el contrato de entrada/salida de la API.
Son todos los campos nentamente que tiene el modelo
CreditFeatures recibe los 15 atributos "crudos" desde el frontend y calcula
las 4 features derivadas que el modelo espera (ver to_model_input).
"""

from pydantic import BaseModel, Field

class CreditFeatures(BaseModel):
    """Datos que ingresa el usuario en el formulario del frontend."""

    edad:                    int   = Field(..., example=33)
    ingresos_mensuales:      float = Field(..., example=3599)
    score_buro:              float = Field(..., example=1)
    estado_civil:            str   = Field(..., example="s")
    nivel_educativo:         str   = Field(..., example="s")
    tipo_empleo:             str   = Field(..., example="d")
    antiguedad_laboral_meses: int  = Field(..., example=36)
    deudas_actuales:         float = Field(..., example=1000)
    monto_solicitado:        float = Field(..., example=5000)
    cuota_mensual_estimada:  float = Field(..., example=500)
    historial_pagos_tardios: int   = Field(..., example=0)
    ratio_deuda_ingreso:     float = Field(..., example=0.30)
    num_creditos_activos:    int   = Field(..., example=1)
    num_dependientes:        int   = Field(..., example=1)
    plazo_meses:             int   = Field(..., example=24)

    """" Api espera estos campos y controla que tenga el tipo de dato correcto"""

    def to_model_input(self) -> dict:                       #to_model_input --> preparar lso textos para XGBoost
        """Calcula las features derivadas y devuelve el registro completo."""

        alm = self.antiguedad_laboral_meses
        ingresos = self.ingresos_mensuales
        monto = self.monto_solicitado
        cuota = self.cuota_mensual_estimada

        return {
            "edad":                         self.edad,
            "ingresos_mensuales":           ingresos,
            "score_buro":                   self.score_buro,
            "estado_civil":                 self.estado_civil,
            "nivel_educativo":              self.nivel_educativo,
            "tipo_empleo":                  self.tipo_empleo,
            "antiguedad_laboral_meses":     alm,
            "deudas_actuales":              self.deudas_actuales,
            "monto_solicitado":             monto,
            "cuota_mensual_estimada":       cuota,
            "historial_pagos_tardios":      self.historial_pagos_tardios,
            "ratio_deuda_ingreso":          self.ratio_deuda_ingreso,
            "num_creditos_activos":         self.num_creditos_activos,
            "num_dependientes":             self.num_dependientes,
            "plazo_meses":                  self.plazo_meses,
            # Variables derivadas que se creo en feature
            "antiguedad_laboral_anios":     round(alm / 12, 2),     # convertir 36 meses a 3 años
            "monto_solicitado_ingreso":     round(monto / ingresos, 3),
            "cuota_ingreso":                round(cuota / ingresos, 3),
            "tiene_pagos_tardios":          1 if self.historial_pagos_tardios > 0 else 0,
        }


class PredictionResponse(BaseModel):
    """Respuesta que devuelve la API de inferencia."""

    prediction:         int   = Field(
        ..., example=0,
        description="0=No Default, 1=Default"
    )
    p_default:          float = Field(..., example=0.1537)
    p_no_default:       float = Field(..., example=0.8463)
    risk:               str   = Field(
        ..., example="VERY_LOW",
        description="VERY_LOW / LOW / MEDIUM / HIGH"
    )
    threshold:          float = Field(..., example=0.50)