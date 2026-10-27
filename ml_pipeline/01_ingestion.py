import pandas as pd

from config.path import RAW_DATA_PATH, INGESTION_PATH

def ingest_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df.to_csv(INGESTION_PATH, index=False)

    return df

#Ejecutar
if __name__ == "__main__":
    df = ingest_data()

print("✅ Ingesta completada correctamente.")
print(f"📥 Origen: {RAW_DATA_PATH}")
print(f"📤 Salida: {INGESTION_PATH}")
print(f"📊 Filas: {df.shape[0]}")
print(f"📊 Columnas: {df.shape[1]}")
