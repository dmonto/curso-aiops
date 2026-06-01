import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def generar_dataset_operativo() -> pd.DataFrame:
    """
    Genera un dataset simulado de métricas operativas.

    Incluye comportamiento normal y varios tipos de anomalías:
    - latencia alta con errores
    - caída de tráfico
    - saturación de CPU y memoria
    """
    np.random.seed(42)

    n = 600
    timestamps = pd.date_range(
        start="2026-05-25 00:00:00",
        periods=n,
        freq="5min"
    )

    hora = timestamps.hour

    # Tráfico más alto en horario laboral
    request_count = np.where(
        (hora >= 9) & (hora <= 18),
        np.random.normal(1200, 180, n),
        np.random.normal(350, 80, n)
    )

    request_count = np.maximum(request_count, 20)

    latency_ms = np.random.normal(180, 25, n) + (request_count / 1200) * 40
    error_rate = np.random.normal(0.004, 0.002, n)
    cpu_percent = 25 + (request_count / 1200) * 35 + np.random.normal(0, 5, n)
    memory_percent = np.random.normal(55, 6, n)

    # Asegurar límites razonables
    error_rate = np.clip(error_rate, 0, 1)
    cpu_percent = np.clip(cpu_percent, 0, 100)
    memory_percent = np.clip(memory_percent, 0, 100)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": "checkout-api",
        "environment": "prod",
        "latency_ms": latency_ms,
        "error_rate": error_rate,
        "request_count": request_count,
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
    })

    # Anomalía 1: degradación de aplicación
    idx = range(180, 195)
    df.loc[idx, "latency_ms"] += 280
    df.loc[idx, "error_rate"] += 0.06

    # Anomalía 2: caída brusca de tráfico
    idx = range(320, 335)
    df.loc[idx, "request_count"] *= 0.12
    df.loc[idx, "latency_ms"] += 80

    # Anomalía 3: saturación de infraestructura
    idx = range(470, 490)
    df.loc[idx, "cpu_percent"] = np.random.normal(94, 2, len(idx))
    df.loc[idx, "memory_percent"] = np.random.normal(91, 3, len(idx))
    df.loc[idx, "latency_ms"] += 150

    return df


def entrenar_modelo_no_supervisado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entrena Isolation Forest sobre variables operativas.

    El modelo no recibe etiquetas. Solo aprende el patrón general de los datos.
    """
    features = [
        "latency_ms",
        "error_rate",
        "request_count",
        "cpu_percent",
        "memory_percent"
    ]

    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.06,
        random_state=42
    )

    model.fit(X_scaled)

    # predict devuelve:
    #  1 para normal
    # -1 para anomalía
    prediction = model.predict(X_scaled)

    # decision_function: valores más bajos suelen indicar mayor rareza.
    raw_score = model.decision_function(X_scaled)

    result = df.copy()
    result["model_prediction"] = prediction
    result["is_anomaly"] = prediction == -1

    # Convertimos el score para que sea más intuitivo:
    # mayor valor = más anomalía
    result["anomaly_score"] = -raw_score

    return result


def explicar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade una explicación operativa sencilla basada en qué variables están altas o bajas.
    No es explicación interna del modelo, pero ayuda a interpretar la detección.
    """
    result = df.copy()

    p95_latency = result["latency_ms"].quantile(0.95)
    p95_error = result["error_rate"].quantile(0.95)
    p95_cpu = result["cpu_percent"].quantile(0.95)
    p95_memory = result["memory_percent"].quantile(0.95)
    p05_traffic = result["request_count"].quantile(0.05)

    def reason(row: pd.Series) -> str:
        motivos = []

        if row["latency_ms"] >= p95_latency:
            motivos.append("latencia alta")
        if row["error_rate"] >= p95_error:
            motivos.append("tasa de error alta")
        if row["request_count"] <= p05_traffic:
            motivos.append("caída de tráfico")
        if row["cpu_percent"] >= p95_cpu:
            motivos.append("CPU alta")
        if row["memory_percent"] >= p95_memory:
            motivos.append("memoria alta")

        if not motivos:
            return "patrón multivariable raro"

        return ", ".join(motivos)

    result["anomaly_reason"] = np.where(
        result["is_anomaly"],
        result.apply(reason, axis=1),
        ""
    )

    def action(row: pd.Series) -> str:
        if not row["is_anomaly"]:
            return ""

        reason_text = row["anomaly_reason"]

        if "tasa de error alta" in reason_text:
            return "Revisar logs de aplicación y últimos despliegues"
        if "caída de tráfico" in reason_text:
            return "Revisar balanceador, frontend, DNS o entrada de tráfico"
        if "CPU alta" in reason_text or "memoria alta" in reason_text:
            return "Revisar saturación de infraestructura y escalado"
        if "latencia alta" in reason_text:
            return "Revisar dependencias lentas y trazas distribuidas"

        return "Revisar patrón operativo y correlacionar con eventos"

    result["recommended_action"] = result.apply(action, axis=1)

    return result


def mostrar_resumen(df: pd.DataFrame) -> None:
    total = len(df)
    anomalies = df[df["is_anomaly"]].copy()

    print("Resumen del modelo no supervisado")
    print("----------------------------------")
    print(f"Registros analizados: {total}")
    print(f"Anomalías detectadas: {len(anomalies)}")
    print(f"Porcentaje detectado: {len(anomalies) / total * 100:.2f}%")

    print("\nPrimeras anomalías:")
    columns = [
        "timestamp",
        "latency_ms",
        "error_rate",
        "request_count",
        "cpu_percent",
        "memory_percent",
        "anomaly_score",
        "anomaly_reason",
        "recommended_action",
    ]

    print(
        anomalies[columns]
        .head(15)
        .to_string(index=False)
    )


def visualizar(df: pd.DataFrame) -> None:
    normal = df[~df["is_anomaly"]]
    anomalies = df[df["is_anomaly"]]

    plt.figure(figsize=(12, 6))

    plt.scatter(
        normal["timestamp"],
        normal["latency_ms"],
        s=12,
        label="Normal"
    )

    plt.scatter(
        anomalies["timestamp"],
        anomalies["latency_ms"],
        s=45,
        marker="x",
        label="Anomalía"
    )

    plt.title("Detección no supervisada de anomalías operativas")
    plt.xlabel("Tiempo")
    plt.ylabel("Latencia ms")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = generar_dataset_operativo()
    df = entrenar_modelo_no_supervisado(df)
    df = explicar_anomalias(df)

    mostrar_resumen(df)
    visualizar(df)

    output_path = "anomalias_no_supervisadas.csv"
    df.to_csv(output_path, index=False)

    print(f"\nArchivo generado: {output_path}")


if __name__ == "__main__":
    main()