
# Imagen base de Python
FROM python:3.12-slim
# Directorio de trabajo
WORKDIR /app
# Copiar e instalar las dependencias de Streamlit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copiar el código de la interfaz
COPY . .
# Puerto de Streamlit
EXPOSE 8501
# Iniciar la aplicación web
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
