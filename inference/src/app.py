"""
Api de inferencia (FastApi) que sirve el modelo de riesgo crediticio
"""

from fastapi import FastAPI                                     # traemos FastAPI para construir el backend/API
from src.schemas import CreditFeatures, PredictionResponse       # Se conecta el api con schemas
from src.predictor import Predictor                               # se coencta el api con predictor

# CREAMOS NUESTRA APLICACION
app = FastAPI(
    title="Credit Risk Predicticion Service", 
    version="1.0.0"
)
predictor = Predictor() # Cargamos el modelo una sola vez

# Permite ver si la APP levanto y sigue viva
@app.get("/health")
def health():
    return {"status": "ok"}

# Analizar si la API esta lista para recibir solicitudes 
@app.get("/ready")
def ready():
    return{
        "status":"ready",
        "model_loaded": predictor.is_loaded()
    }

# Cumple la funcionalidad de enviar datos al servidor para que realice algo
@app.post("/predict", response_model=PredictionResponse)
def predict(features: CreditFeatures):
    """Recibe los datos del cliente y devuelve la predicción."""

    result = predictor.predict(
        features.to_model_input()
    )

    return result
