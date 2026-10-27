

# Etapa 1: instalar dependencias
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Etapa 2: imagen final
FROM python:3.12-slim

WORKDIR /app

# Copiar las dependencias instaladas
COPY --from=builder /install /usr/local

# Copiar el código de FastAPI
COPY src/ ./src/

# Copiar el modelo ajustado
COPY artifacts/tuned_model.pkl ./artifacts/tuned_model.pkl

# Puerto interno de la API
EXPOSE 8080

# Arrancar FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
