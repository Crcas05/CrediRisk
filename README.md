
# REPASO GENERAL — AI ENGINEER, MLOps Y AGENTIC AI

## 1. MLOps: pipeline de entrenamiento

### 1.1. Las 6 fases del pipeline

1. **Ingestión de datos:** `01_ingestion.py`
2. **Validación de datos:** `02_validation.py`
3. **Análisis exploratorio (EDA):** `03_eda.py`
4. **Limpieza de datos:** `04_cleaning.py`
5. **Feature Engineering:** `05_feature_engineering.py`
6. **Preprocesamiento:** `06_preprocessing.py`

**Estado:** fases definidas dentro del flujo de trabajo. Pendiente verificar su integración completa en un pipeline automatizado.

### 1.2. Entrenamiento y evaluación

- División de datos: 80 % para entrenamiento y 20 % para prueba.
- Entrenamiento y evaluación del modelo.
- Validación cruzada y ajuste de hiperparámetros.
- Inferencia y explicabilidad del modelo.
- **Estado actual:** modelo XGBoost entrenado y artefacto `tuned_model.pkl` utilizado para realizar predicciones.
- **Siguiente objetivo:** automatizar el entrenamiento, la evaluación y la validación del modelo.

### 1.3. Herramientas implementadas y practicadas

- **XGBoost + Scikit-learn:** modelo predictivo con preprocesamiento y clasificación del riesgo crediticio.
- **FastAPI + Pydantic:** API de inferencia con endpoint `/predict` y validación de datos.
- **Postman:** pruebas del endpoint y comprobación de respuestas de la API.
- **Streamlit:** interfaz web para ingresar datos del solicitante y visualizar predicciones.
- **Docker:** construcción y ejecución de contenedores para FastAPI y Streamlit.
- **Docker Compose:** orquestación de ambos servicios mediante `docker-compose.yaml` y una red compartida.
- **DVC + MinIO:** configuración del versionado de datos y almacenamiento remoto; pendiente verificar la transferencia completa.
- **Docker Hub:** espacio de repositorios de CRCAS: https://hub.docker.com/repositories/crcas.
- **Git:** versión instalada y repositorio local disponible. Identidad local configurada parcialmente; pendiente completar el correo y verificar los primeros commits.

### 1.4. Proyecto práctico: CrediRisk

Aplicación experimental de evaluación del riesgo crediticio que integra Machine Learning, API e interfaz web.

**Arquitectura del proyecto:**

- **Backend:** FastAPI, Pydantic y modelo XGBoost.
- **Frontend:** Streamlit.
- **Artefacto del modelo:** `inference/artifacts/tuned_model.pkl`.
- **Endpoint de inferencia:** `/predict`.
- **Contenerización:** Dockerfiles independientes para FastAPI y Streamlit.
- **Orquestación:** `streamlit-app/docker-compose.yaml`.
- **Comunicación interna:** `API_URL=http://fastapi:8080`.

**Imágenes construidas localmente:**

- `crcas/credit-risk-api:dev`
- `crcas/credit-risk-streamlit:dev`

**Puertos configurados en Docker Compose:**

- FastAPI: puerto `8001` de la máquina anfitriona → puerto `8080` del contenedor.
- Streamlit: puerto `8501` de la máquina anfitriona → puerto `8501` del contenedor.

**Logros verificados:**

- Construcción de las imágenes Docker de ambos servicios.
- Ejecución de FastAPI y Streamlit mediante contenedores.
- Configuración de la red interna de Docker Compose.
- Comunicación entre Streamlit y FastAPI mediante el nombre del servicio `fastapi`.
- Visualización de predicciones del modelo desde la interfaz web.
- Uso práctico de comandos como `docker image build`, `docker run`, `docker ps`, `docker logs`, `docker stop` y `docker rm`.
- Ejecución coordinada de los servicios mediante `docker compose up -d`.

**Estado:** aplicación ejecutada localmente con ambos servicios orquestados. La publicación de las versiones actuales de las imágenes en Docker Hub está pendiente de verificar.

### 1.5. Próximas integraciones y despliegue

- **MLflow:** seguimiento de experimentos, parámetros y métricas, y gestión de versiones del modelo.
- **Automatización del entrenamiento:** integrar las fases de preparación de datos, entrenamiento y evaluación en un pipeline reproducible.
- **Docker Hub:** publicar y verificar las imágenes actualizadas de FastAPI y Streamlit.
- **GitHub:** completar la configuración del repositorio remoto y publicar el código fuente.
- **GitHub Actions:** automatizar pruebas, construcción de imágenes y publicación en Docker Hub mediante CI/CD.
- **Despliegue automatizado:** configurar la actualización de la aplicación en un entorno de ejecución.
- **Podman:** comparación práctica con Docker.
- **Kubernetes:** orquestación de contenedores a mayor escala.
- **Grafana:** visualización de métricas y dashboards de monitorización.

## 2. Prompt Engineering

- **Componentes del prompt:** System, User y Assistant.
- **Tipos de prompt:** Zero-shot, Few-shot, Templates y Prompt Chaining.
- **Técnicas de razonamiento de LLMs:** Chain of Thought (CoT), Self-Consistency y Tree of Thought.

## 3. Patrones y tipos de agentes

- **Patrones agénticos:** ReAct, Plan-and-Execute y Multiagente.
- **Comportamiento y colaboración:** agentes reactivos, autónomos, sistemas multiagente y copilotos.
- **Organización y uso empresarial:** workflows, componentes reutilizables, sistemas multiagente y copilotos.

## 4. Automatización y observabilidad

- **Salida JSON:** estructuración de respuestas.
- **JSON mode:** generación de respuestas en formato JSON.
- **Function Calling / Tool Calling:** solicitud de ejecución de herramientas.
- **Validación:** comprobación de estructuras, tipos y valores.
- **Langfuse:** observabilidad de aplicaciones LLM mediante trazas, latencia, errores y evaluación.

## 5. Frameworks de agentes de IA

- **LangGraph:** biblioteca de Python que organiza flujos mediante nodos, aristas y estado.
- **CrewAI:** framework para construir sistemas multiagente mediante agentes, roles, tareas y colaboración.

## 6. Desarrollo de software asistido por IA

- **Especificación:** contrato que define requisitos y criterios de aceptación para construir software.
- **OpenCode:** agente de programación de código abierto compatible con distintos proveedores de modelos.
- **OpenSpec:** enfoque de desarrollo guiado por especificaciones (*Spec-Driven Development*).

## 7. Objetivo final

Construir aplicaciones de IA que integren Machine Learning, MLOps, LLMs y agentes inteligentes, desde la preparación de datos y el entrenamiento hasta la inferencia, automatización, despliegue y observabilidad.

### Proyecto práctico actual: CrediRisk

Aplicación experimental de evaluación del riesgo crediticio desarrollada con XGBoost, FastAPI, Pydantic, Streamlit, Docker y Docker Compose.

**Estado actual:**

- Modelo de Machine Learning entrenado y utilizado para realizar predicciones.
- API de inferencia implementada con FastAPI.
- Interfaz web implementada con Streamlit.
- Imágenes Docker de ambos servicios construidas.
- Servicios orquestados mediante Docker Compose.
- Comunicación entre contenedores verificada mediante la red interna.
- Predicciones visualizadas correctamente desde la interfaz web.

**Próximo hito: publicación y CI/CD**

1. Completar la configuración de Git y publicar el código fuente en GitHub.
2. Publicar y verificar las imágenes de FastAPI y Streamlit en Docker Hub.
3. Configurar GitHub Actions para ejecutar pruebas y construir las imágenes automáticamente.
4. Automatizar la publicación de imágenes mediante CI/CD.
5. Configurar el despliegue automatizado de la aplicación en un entorno adecuado.
6. Integrar MLflow para registrar experimentos, métricas y versiones del modelo.
7. Implementar un flujo de reentrenamiento automático que prepare nuevos datos, entrene y evalúe el modelo, valide sus resultados y permita desplegar únicamente las versiones aprobadas.

**Flujo objetivo de CI/CD:**

GitHub → GitHub Actions → Pruebas y construcción de imágenes → Docker Hub → Despliegue de la aplicación.

**Flujo objetivo de reentrenamiento de Machine Learning:**

Nuevos datos → Preparación y validación → Entrenamiento → Evaluación → Registro de la versión → Despliegue del modelo aprobado.

> **Nota:** el reentrenamiento automático no forma parte de CI/CD por defecto. Será necesario configurarlo explícitamente, definir cuándo se ejecuta y establecer criterios de evaluación antes de desplegar una nueva versión del modelo.
