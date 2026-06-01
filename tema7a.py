import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generar_datos_operativos() -> pd.DataFrame:
    """
    Genera una serie temporal simulada de latencia de una API.

    La serie incluye:
    - patrón base estable,
    - variación natural,
    - estacionalidad ligera,
    - anomalías puntuales,
    - cambio de nivel.
    """
    np.random.seed(42)

    n = 240
    timestamps = pd.date_range(
        start="2026-05-25 00:00:00",
        periods=n,
        freq="5min"
    )

    base_latency = 180
    ruido = np.random.normal(loc=0, scale=12, size=n)

    # Patrón horario simple: algo más de latencia en las horas centrales
    hora = timestamps.hour
    estacionalidad = np.where((hora >= 9) & (hora <= 18), 25, 0)

    latency_ms = base_latency + ruido + estacionalidad

    # Anomalías puntuales
    latency_ms[60] = 420
    latency_ms[130] = 510
    latency_ms[190] = 470

    # Cambio de nivel: a partir de cierto momento la latencia empeora
    latency_ms[160:] += 55

    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": "checkout-api",
        "latency_ms": latency_ms
    })

    return df


def detectar_anomalias(
    df: pd.DataFrame,
    ventana: int = 24,
    factor_desviacion: float = 2.5
) -> pd.DataFrame:
    """
    Detecta anomalías comparando cada punto con una ventana móvil previa.

    ventana=24 equivale a 24 periodos de 5 minutos, es decir, 2 horas.
    factor_desviacion controla la sensibilidad.
    """
    df = df.copy()

    df["rolling_mean"] = (
        df["latency_ms"]
        .rolling(window=ventana, min_periods=ventana)
        .mean()
    )

    df["rolling_std"] = (
        df["latency_ms"]
        .rolling(window=ventana, min_periods=ventana)
        .std()
    )

    df["upper_limit"] = df["rolling_mean"] + factor_desviacion * df["rolling_std"]
    df["lower_limit"] = df["rolling_mean"] - factor_desviacion * df["rolling_std"]

    df["is_anomaly"] = (
        (df["latency_ms"] > df["upper_limit"]) |
        (df["latency_ms"] < df["lower_limit"])
    )

    # Evitamos marcar anomalías antes de tener suficiente histórico
    df.loc[df["rolling_mean"].isna(), "is_anomaly"] = False

    return df


def generar_resumen(df: pd.DataFrame) -> None:
    total = len(df)
    total_anomalias = int(df["is_anomaly"].sum())
    porcentaje = total_anomalias / total * 100

    print("Resumen de detección")
    print("--------------------")
    print(f"Servicio analizado: {df['service'].iloc[0]}")
    print(f"Puntos analizados: {total}")
    print(f"Anomalías detectadas: {total_anomalias}")
    print(f"Porcentaje anómalo: {porcentaje:.2f}%")

    if total_anomalias > 0:
        print("\nPrimeras anomalías detectadas:")
        columnas = ["timestamp", "latency_ms", "rolling_mean", "upper_limit"]
        print(df.loc[df["is_anomaly"], columnas].head(10).to_string(index=False))


def visualizar(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(df["timestamp"], df["latency_ms"], label="Latencia real")
    plt.plot(df["timestamp"], df["rolling_mean"], label="Media móvil")
    plt.plot(df["timestamp"], df["upper_limit"], label="Límite superior dinámico")

    anomalías = df[df["is_anomaly"]]
    plt.scatter(
        anomalías["timestamp"],
        anomalías["latency_ms"],
        label="Anomalía detectada",
        marker="x",
        s=80
    )

    plt.title("Detección simple de anomalías operativas")
    plt.xlabel("Tiempo")
    plt.ylabel("Latencia ms")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = generar_datos_operativos()
    df = detectar_anomalias(df, ventana=24, factor_desviacion=2.5)

    generar_resumen(df)
    visualizar(df)

    df.to_csv("anomalias_operativas.csv", index=False)
    print("\nArchivo generado: anomalias_operativas.csv")


if __name__ == "__main__":
    main()