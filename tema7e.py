import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from google.cloud import monitoring_v3
from google.api import label_pb2
from google.api import metric_pb2
from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp


load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "europe-west1")

METRIC_TYPE = "custom.googleapis.com/aiops/anomaly_score"


@dataclass
class AnomalyPoint:
    service: str
    environment: str
    region: str
    anomaly_score: float
    latency_ms: float
    latency_threshold: float
    error_rate: float
    timestamp: pd.Timestamp


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def generar_datos_operativos() -> pd.DataFrame:
    """
    Genera una serie temporal simulada para tres servicios.
    Después introduce anomalías controladas para probar la integración.
    """
    np.random.seed(42)

    services = ["checkout-api", "catalog-api", "auth-api"]
    timestamps = pd.date_range(
        start=pd.Timestamp.utcnow().floor("5min") - pd.Timedelta(hours=8),
        periods=96,
        freq="5min"
    )

    rows = []

    for service in services:
        for ts in timestamps:
            hour = ts.hour
            business_hours = 9 <= hour <= 18

            if service == "checkout-api":
                base_latency = 180
                base_traffic = 1200 if business_hours else 350
            elif service == "catalog-api":
                base_latency = 120
                base_traffic = 900 if business_hours else 250
            else:
                base_latency = 90
                base_traffic = 1500 if business_hours else 500

            request_count = max(10, np.random.normal(base_traffic, base_traffic * 0.15))
            latency_ms = np.random.normal(base_latency, 18) + request_count / 90
            error_rate = max(0, np.random.normal(0.004, 0.002))

            rows.append({
                "timestamp": ts,
                "service": service,
                "environment": "prod",
                "region": LOCATION,
                "request_count": request_count,
                "latency_ms": latency_ms,
                "error_rate": error_rate,
            })

    df = pd.DataFrame(rows)

    # Anomalía reciente: checkout-api con latencia y errores altos
    cutoff_start = timestamps[-10]
    cutoff_end = timestamps[-4]

    mask = (
        (df["service"] == "checkout-api") &
        (df["timestamp"] >= cutoff_start) &
        (df["timestamp"] <= cutoff_end)
    )

    df.loc[mask, "latency_ms"] += 320
    df.loc[mask, "error_rate"] += 0.06

    return df


def calcular_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un threshold dinámico sencillo por servicio.
    Después genera anomaly_score de 0 a 100.
    """
    result = []

    for service, group in df.groupby("service"):
        g = group.sort_values("timestamp").copy()

        window = 24  # 2 horas con frecuencia de 5 minutos

        g["latency_mean"] = (
            g["latency_ms"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

        g["latency_std"] = (
            g["latency_ms"]
            .rolling(window=window, min_periods=window)
            .std()
        )

        g["latency_threshold"] = g["latency_mean"] + 3 * g["latency_std"]

        # Score por latencia
        g["latency_score"] = np.where(
            g["latency_ms"] > g["latency_threshold"],
            ((g["latency_ms"] - g["latency_threshold"]) / g["latency_threshold"]) * 100,
            0
        )

        # Score por error rate
        error_baseline = (
            g["error_rate"]
            .rolling(window=window, min_periods=window)
            .quantile(0.95)
        )

        g["error_rate_threshold"] = error_baseline * 2

        g["error_score"] = np.where(
            g["error_rate"] > g["error_rate_threshold"],
            60,
            0
        )

        g["anomaly_score"] = (
            g["latency_score"] + g["error_score"]
        ).clip(0, 100)

        g["is_anomaly"] = g["anomaly_score"] >= 80

        result.append(g)

    final = pd.concat(result).sort_values(["service", "timestamp"])

    # Cuando no hay histórico suficiente, dejamos score a 0
    final["anomaly_score"] = final["anomaly_score"].fillna(0)
    final["latency_threshold"] = final["latency_threshold"].fillna(final["latency_ms"])

    return final


def crear_metric_descriptor_si_no_existe(project_id: str) -> None:
    """
    Crea el descriptor de la métrica personalizada si todavía no existe.
    """
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    descriptor_name = f"{project_name}/metricDescriptors/{METRIC_TYPE}"

    try:
        client.get_metric_descriptor(name=descriptor_name)
        print(f"Metric descriptor ya existe: {METRIC_TYPE}")
        return
    except Exception:
        pass

    descriptor = metric_pb2.MetricDescriptor()
    descriptor.type = METRIC_TYPE
    descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
    descriptor.display_name = "AIOps anomaly score"
    descriptor.description = (
        "Puntuación de anomalía operativa generada por el pipeline AIOps. "
        "Valores altos indican mayor probabilidad de comportamiento anómalo."
    )

    descriptor.labels.append(
        label_pb2.LabelDescriptor(
            key="service",
            value_type=label_pb2.LabelDescriptor.ValueType.STRING,
            description="Servicio analizado",
        )
    )

    descriptor.labels.append(
        label_pb2.LabelDescriptor(
            key="environment",
            value_type=label_pb2.LabelDescriptor.ValueType.STRING,
            description="Entorno operativo",
        )
    )

    descriptor.labels.append(
        label_pb2.LabelDescriptor(
            key="region",
            value_type=label_pb2.LabelDescriptor.ValueType.STRING,
            description="Región o localización lógica",
        )
    )

    client.create_metric_descriptor(
        name=project_name,
        metric_descriptor=descriptor,
    )

    print(f"Metric descriptor creado: {METRIC_TYPE}")
    

def publicar_punto_cloud_monitoring(
    project_id: str,
    point: AnomalyPoint
) -> None:
    """
    Publica un único punto de anomaly_score en Cloud Monitoring.
    """
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = METRIC_TYPE
    series.metric.labels["service"] = point.service
    series.metric.labels["environment"] = point.environment
    series.metric.labels["region"] = point.region

    series.resource.type = "global"
    series.resource.labels["project_id"] = project_id

    timestamp = Timestamp()
    timestamp.seconds = int(point.timestamp.timestamp())

    interval = monitoring_v3.TimeInterval()
    interval.end_time = timestamp

    value = monitoring_v3.TypedValue()
    value.double_value = float(point.anomaly_score)

    monitoring_point = monitoring_v3.Point()
    monitoring_point.interval = interval
    monitoring_point.value = value

    series.points = [monitoring_point]

    client.create_time_series(
        name=project_name,
        time_series=[series]
    )


def publicar_ultimos_puntos(df: pd.DataFrame, project_id: str) -> None:
    """
    Publica solo el último punto de cada servicio para evitar enviar demasiados datos
    en el laboratorio.
    """
    latest = (
        df.sort_values("timestamp")
        .groupby("service")
        .tail(1)
    )

    for _, row in latest.iterrows():
        point = AnomalyPoint(
            service=row["service"],
            environment=row["environment"],
            region=row["region"],
            anomaly_score=float(row["anomaly_score"]),
            latency_ms=float(row["latency_ms"]),
            latency_threshold=float(row["latency_threshold"]),
            error_rate=float(row["error_rate"]),
            timestamp=row["timestamp"],
        )

        publicar_punto_cloud_monitoring(project_id, point)

        print(
            f"Publicado anomaly_score={point.anomaly_score:.2f} "
            f"service={point.service} "
            f"timestamp={point.timestamp}"
        )


def crear_alert_policy(project_id: str) -> None:
    """
    Crea una política de alerta básica sobre anomaly_score.

    No configura canales de notificación.
    El incidente aparecerá en Cloud Monitoring si se cumple la condición.
    """
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    display_name = "AIOps - Anomaly score alto"

    existing_policies = client.list_alert_policies(name=project_name)
    for policy in existing_policies:
        if policy.display_name == display_name:
            print(f"Alert policy ya existe: {display_name}")
            return

    condition = monitoring_v3.AlertPolicy.Condition()
    condition.display_name = "anomaly_score > 80"

    condition.condition_threshold.filter = (
        f'metric.type = "{METRIC_TYPE}" AND resource.type = "global"'
    )
    condition.condition_threshold.comparison = (
        monitoring_v3.ComparisonType.COMPARISON_GT
    )
    condition.condition_threshold.threshold_value = 80.0
    condition.condition_threshold.duration = Duration(seconds=60)

    aggregation = monitoring_v3.Aggregation()
    aggregation.alignment_period = Duration(seconds=60)
    aggregation.per_series_aligner = (
        monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
    )

    condition.condition_threshold.aggregations = [aggregation]

    policy = monitoring_v3.AlertPolicy()
    policy.display_name = display_name
    policy.combiner = monitoring_v3.AlertPolicy.ConditionCombinerType.OR
    policy.conditions = [condition]
    policy.enabled = True
    policy.documentation.content = (
        "Anomaly score alto detectado por el pipeline AIOps. "
        "Revisar servicio, logs, latencia, error_rate y últimos despliegues."
    )
    policy.documentation.mime_type = "text/markdown"

    created = client.create_alert_policy(
        name=project_name,
        alert_policy=policy
    )

    print(f"Alert policy creada: {created.name}")


def mostrar_resumen(df: pd.DataFrame) -> None:
    print("Resumen de detección")
    print("--------------------")

    columns = [
        "timestamp",
        "service",
        "latency_ms",
        "latency_threshold",
        "error_rate",
        "anomaly_score",
        "is_anomaly",
    ]

    print(
        df.sort_values("timestamp")
        .groupby("service")
        .tail(3)[columns]
        .to_string(index=False)
    )


def main() -> None:
    project_id = require_env("PROJECT_ID")

    print(f"Proyecto: {project_id}")
    print(f"Región lógica: {LOCATION}")
    print()

    df = generar_datos_operativos()
    df = calcular_anomaly_score(df)

    mostrar_resumen(df)

    output_file = "cloud_monitoring_aiops_anomaly_scores.csv"
    df.to_csv(output_file, index=False)
    print(f"\nCSV generado: {output_file}")

    crear_metric_descriptor_si_no_existe(project_id)
    publicar_ultimos_puntos(df, project_id)

    # Activar solo si se quiere crear la política de alerta.
    # Requiere permisos de Monitoring Editor.
    crear_alert_policy(project_id)

    print("\nProceso completado.")
    print("Revisa la métrica en Cloud Monitoring > Metrics Explorer.")
    print(f"Métrica: {METRIC_TYPE}")


if __name__ == "__main__":
    main()