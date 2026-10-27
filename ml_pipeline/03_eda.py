"""Entendemos el comportamiento de los datos
- Estadistica descriptiva: media, moda, mediana y desviacion standar
- Distribucion de variables
- Correlacion
- Visualizaciones
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.path import VALIDATION_PATH, EDA_PATH, EDA_REPORT_DIR

def eda_data(df):
    print("Basic Statitstics:")
    print(df.describe(include='all'))

    # Grafico de Edad
    plt.figure()
    sns.histplot(df["edad"], kde=True)
    plt.title("Distribución de Edad")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.savefig(EDA_REPORT_DIR / "distribucion_edad.png")
    plt.show()


    # Grafco de Score de buro
    plt.figure()
    sns.histplot(df["score_buro"], kde=True)
    plt.title("Distribución del Score de Buró")
    plt.xlabel("Score de Buró")
    plt.ylabel("Frecuencia")
    plt.savefig(EDA_REPORT_DIR / "distribucion_score_buro.png")
    plt.show()


    # Distribución del ratio deuda/ingreso
    plt.figure()
    sns.histplot(df["ratio_deuda_ingreso"], kde=True)
    plt.title("Distribución del Ratio Deuda/Ingreso")
    plt.xlabel("Ratio Deuda/Ingreso")
    plt.ylabel("Frecuencia")
    plt.savefig(EDA_REPORT_DIR / "distribucion_ratio_deuda_ingreso.png")
    plt.show()

    df.to_csv(EDA_PATH, index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv(VALIDATION_PATH)
    eda_data(df)

print("Asimetría:")
print(df["ratio_deuda_ingreso"].skew())

print("Curtosis:")
print(df["ratio_deuda_ingreso"].kurt())