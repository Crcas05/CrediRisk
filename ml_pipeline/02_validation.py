import pandera.pandas as pa
from pandera import Column, Check
import pandas as pd
from config.path import INGESTION_PATH, VALIDATION_PATH

# Definimos las reglas
credit_schema = pa.DataFrameSchema(
    {
        "cliente_id": Column(str, unique=True, nullable=False),
        "estado_civil": Column(str, Check.isin(["soltero", "casado","divorciado","viudo"])),
        "historial_pagos_tardios":Column(int, Check.ge(0), nullable=False),  
    },
    strict=False,
    coerce=True
)

# Validamos el dataframe
def validaate_data(df):
    try:
        validate_df = credit_schema(df, lazy=True)
        print("Data validation successful. ")
        return validate_df
    
    except pa.errors.SchemaErrors as e:
        print("Data validation errors found:")
        print(e.failure_cases)
        return None

# Ejecutamos
if __name__ == "__main__":
    df = pd.read_csv(INGESTION_PATH)

    print("=== Estructura del DataFrame ===")
    df.info()

    validated_df = validaate_data(df)

    if validated_df is not None:
        validated_df.to_csv(VALIDATION_PATH, index=False)
        print("Validated dataset saved successfully.")

   

