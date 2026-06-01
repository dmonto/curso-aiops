import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generar_datos() -> pd.DataFrame:
    """
    Genera datos operativos simulados para varios servicios.

    Incluye:
    - estacionalidad por hora,
    - comportamiento diferente por servicio,
    - anomalías de latencia,
    - caída de tráfico,
    - subida de errores.
    """
    np.random.seed(42)

    services = ["checkout-api", "catalog-api", "auth-api"]
    timestamps = pd.date_range(
        start="2026-05-25 00:00:00",
        periods=7 * 24 * 12,
        freq="5min"
    )

    rows = []

    for service in services:
        for ts in timestamps:
            hour = ts.hour
            business_hours = 9 <= hour <= 18

            if service == "checkout-api":
                base_latency = 180
                base_traffic = 1100 if business_hours else 350
            elif service == "catalog-api":
                base_latency = 120
                base_traffic = 900 if business_hours else 250
            else:
                base_latency = 90
                base_traffic = 1500 if business_hours else 500

            request_count = max(10, np.random.normal(base_traffic, base_traffic * 0.12))
            latency_ms = np.random.normal(base_latency, 18)

            # Relación natural: más tráfico suele elevar algo la latencia
            latency_ms += request_count / 80

            error_rate = max(0, np.random.normal(0.004, 0.002))

            rows.append({
                "timestamp": ts,
                "service": service,
                "environment": "prod",
                "hour": hour,
                "day_of_week": ts.dayofweek,
                "latency_ms": latency_ms,
                "request_count": request_count,
                "error_rate": error_rate,
            })

    df = pd.DataFrame(rows)

    # Anomalía 1: checkout-api con latencia alta
    mask = (
        (df["service"] == "checkout-api") &
        (df["timestamp"] >= "2026-05-27 10:00:00") &
        (df["timestamp"] <= "2026-05-27 11:00:00")
    )
    df.loc[mask, "latency_ms"] += 280
    df.loc[mask, "error_rate"] += 0.04

    # Anomalía 2: catalog-api con caída de tráfico
    mask = (
        (df["service"] == "catalog-api") &
        (df["timestamp"] >= "2026-05-28 15:00:00") &
        (df["timestamp"] <= "2026-05-28 16:00:00")
    )
    df.loc[mask, "request_count"] *= 0.15

    # Anomalía 3: auth-api con errores altos
    mask = (
        (df["service"] == "auth-api") &
        (df["timestamp"] >= "2026-05-30 09:30:00") &
        (df["timestamp"] <= "2026-05-30 10:30:00")
    )
    df.loc[mask, "error_rate"] += 0.08
    df.loc[mask, "latency_ms"] += 100

    return df


def calcular_thresholds_por_servicio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula thresholds dinámicos por servicio.

    Para latencia:
    - rolling_mean
    - rolling_std
    - upper_threshold_std
    - rolling_p95

    Para tráfico:
    - rolling_p05 como umbral inferior dinámico

    Para error_rate:
    - rolling_p95
    """
    result = []

    for service, group in df.groupby("service"):
        g = group.sort_values("timestamp").copy()

        # 24 puntos de 5 minutos = 2 horas
        short_window = 24

        # 288 puntos de 5 minutos = 24 horas
        daily_window = 288

        g["latency_mean_2h"] = (
            g["latency_ms"]
            .rolling(window=short_window, min_periods=short_window)
            .mean()
        )

        g["latency_std_2h"] = (
            g["latency_ms"]
            .rolling(window=short_window, min_periods=short_window)
            .std()
        )

        g["latency_upper_std"] = (
            g["latency_mean_2h"] + 3 * g["latency_std_2h"]
        )

        g["latency_p95_24h"] = (
            g["latency_ms"]
            .rolling(window=daily_window, min_periods=daily_window)
            .quantile(0.95)
        )

        g["traffic_p05_24h"] = (
            g["request_count"]
            .rolling(window=daily_window, min_periods=daily_window)
            .quantile(0.05)
        )

        g["error_rate_p95_24h"] = (
            g["error_rate"]
            .rolling(window=daily_window, min_periods=daily_window)
            .quantile(0.95)
        )

        result.append(g)

    return pd.concat(result).sort_values(["service", "timestamp"])


def detectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas dinámicas sobre thresholds calculados.
    """
    result = df.copy()

    result["latency_anomaly_std"] = (
        result["latency_ms"] > result["latency_upper_std"]
    )

    result["latency_anomaly_p95"] = (
        result["latency_ms"] > result["latency_p95_24h"] * 1.20
    )

    result["traffic_drop_anomaly"] = (
        result["request_count"] < result["traffic_p05_24h"] * 0.70
    )

    result["error_rate_anomaly"] = (
        result["error_rate"] > result["error_rate_p95_24h"] * 2.0
    )

    anomaly_columns = [
        "latency_anomaly_std",
        "latency_anomaly_p95",
        "traffic_drop_anomaly",
        "error_rate_anomaly",
    ]

    # Evitamos marcar anomalías cuando todavía no hay histórico suficiente
    for col in anomaly_columns:
        result[col] = result[col].fillna(False)

    result["anomaly_score"] = result[anomaly_columns].sum(axis=1)

    result["is_anomaly"] = result["anomaly_score"] > 0

    return result


def recomendar_accion(row: pd.Series) -> str:
    if not row["is_anomaly"]:
        return ""

    if row["error_rate_anomaly"] and (
        row["latency_anomaly_std"] or row["latency_anomaly_p95"]
    ):
        return "Escalar: errores altos con degradación de latencia"

    if row["traffic_drop_anomaly"]:
        return "Revisar balanceador, DNS, frontend o entrada de tráfico"

    if row["error_rate_anomaly"]:
        return "Revisar logs de aplicación y últimos despliegues"

    if row["latency_anomaly_std"] or row["latency_anomaly_p95"]:
        return "Revisar trazas y dependencias lentas"

    return "Revisar comportamiento operativo"


def clasificar_severidad(row: pd.Series) -> str:
    if not row["is_anomaly"]:
        return "NORMAL"

    if row["anomaly_score"] >= 3:
        return "HIGH"

    if row["error_rate_anomaly"] and row["environment"] == "prod":
        return "HIGH"

    if row["anomaly_score"] == 2:
        return "MEDIUM"

    return "LOW"


def enriquecer_resultado(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["severity"] = result.apply(clasificar_severidad, axis=1)
    result["recommended_action"] = result.apply(recomendar_accion, axis=1)
    return result


def mostrar_resumen(df: pd.DataFrame) -> None:
    print("Resumen de thresholds dinámicos")
    print("--------------------------------")
    print(f"Registros analizados: {len(df)}")
    print(f"Anomalías detectadas: {int(df['is_anomaly'].sum())}")
    print()

    print("Anomalías por servicio:")
    print(
        df[df["is_anomaly"]]
        .groupby("service")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nAnomalías por severidad:")
    print(
        df[df["is_anomaly"]]
        .groupby("severity")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nPrimeras anomalías:")
    columns = [
        "timestamp",
        "service",
        "latency_ms",
        "latency_upper_std",
        "latency_p95_24h",
        "request_count",
        "traffic_p05_24h",
        "error_rate",
        "error_rate_p95_24h",
        "anomaly_score",
        "severity",
        "recommended_action",
    ]

    print(
        df[df["is_anomaly"]][columns]
        .head(15)
        .to_string(index=False)
    )


def visualizar_servicio(df: pd.DataFrame, service: str) -> None:
    g = df[df["service"] == service].sort_values("timestamp")
    anomalies = g[g["is_anomaly"]]

    plt.figure(figsize=(12, 6))

    plt.plot(
        g["timestamp"],
        g["latency_ms"],
        label="Latencia real"
    )

    plt.plot(
        g["timestamp"],
        g["latency_upper_std"],
        label="Threshold dinámico std"
    )

    plt.plot(
        g["timestamp"],
        g["latency_p95_24h"],
        label="Threshold dinámico p95 24h"
    )

    plt.scatter(
        anomalies["timestamp"],
        anomalies["latency_ms"],
        marker="x",
        s=70,
        label="Anomalía"
    )

    plt.title(f"Thresholds dinámicos - {service}")
    plt.xlabel("Tiempo")
    plt.ylabel("Latencia ms")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    output_file = f"thresholds_{service}.png"
    plt.savefig(output_file)
    plt.close()

    print(f"Gráfico generado: {output_file}")


def main() -> None:
    df = generar_datos()
    df = calcular_thresholds_por_servicio(df)
    df = detectar_anomalias(df)
    df = enriquecer_resultado(df)

    mostrar_resumen(df)

    output_csv = "thresholds_dinamicos_aiops.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nCSV generado: {output_csv}")

    for service in df["service"].unique():
        visualizar_servicio(df, service)


if __name__ == "__main__":
    main()