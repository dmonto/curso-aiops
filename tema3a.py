import argparse
import os
import random
import time
from typing import Dict, List

from dotenv import load_dotenv
from google.api import label_pb2, metric_pb2
from google.api_core.exceptions import AlreadyExists, GoogleAPICallError, PermissionDenied
from google.cloud import monitoring_v3


METRIC_TYPE = "custom.googleapis.com/aiops/incident_risk_score"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def get_project_name(project_id: str) -> str:
    return f"projects/{project_id}"


def ensure_metric_descriptor(client: monitoring_v3.MetricServiceClient, project_name: str) -> None:
    descriptor = metric_pb2.MetricDescriptor()
    descriptor.type = METRIC_TYPE
    descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
    descriptor.display_name = "AIOps Incident Risk Score"
    descriptor.description = (
        "Score operativo entre 0 y 1 que representa riesgo estimado de incidente."
    )
    descriptor.unit = "1"

    descriptor.labels.append(
        label_pb2.LabelDescriptor(
            key="service",
            value_type=label_pb2.LabelDescriptor.ValueType.STRING,
            description="Servicio lógico observado",
        )
    )

    descriptor.labels.append(
        label_pb2.LabelDescriptor(
            key="environment",
            value_type=label_pb2.LabelDescriptor.ValueType.STRING,
            description="Entorno: lab, dev, test o prod",
        )
    )

    try:
        client.create_metric_descriptor(
            name=project_name,
            metric_descriptor=descriptor,
        )
        print(f"Descriptor creado: {METRIC_TYPE}")
    except AlreadyExists:
        print(f"Descriptor ya existente: {METRIC_TYPE}")


def build_time_series(
    project_id: str,
    service: str,
    environment: str,
    value: float,
) -> monitoring_v3.TimeSeries:
    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10**9)

    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {
                "seconds": seconds,
                "nanos": nanos,
            }
        }
    )

    point = monitoring_v3.Point(
        {
            "interval": interval,
            "value": {
                "double_value": value,
            },
        }
    )

    series = monitoring_v3.TimeSeries()
    series.metric.type = METRIC_TYPE
    series.metric.labels["service"] = service
    series.metric.labels["environment"] = environment

    series.resource.type = "global"
    series.resource.labels["project_id"] = project_id

    series.points = [point]
    return series


def write_sample_scores(
    client: monitoring_v3.MetricServiceClient,
    project_id: str,
    project_name: str,
    environment: str,
) -> None:
    services: Dict[str, float] = {
        "checkout": round(random.uniform(0.60, 0.95), 2),
        "payments": round(random.uniform(0.20, 0.75), 2),
        "inventory": round(random.uniform(0.10, 0.50), 2),
    }

    time_series: List[monitoring_v3.TimeSeries] = []

    for service, value in services.items():
        print(f"Preparando métrica: service={service}, environment={environment}, value={value}")
        time_series.append(
            build_time_series(
                project_id=project_id,
                service=service,
                environment=environment,
                value=value,
            )
        )

    client.create_time_series(
        name=project_name,
        time_series=time_series,
    )

    print("Puntos enviados a Cloud Monitoring.")


def read_recent_scores(
    client: monitoring_v3.MetricServiceClient,
    project_name: str,
    minutes: int,
) -> None:
    now = time.time()

    interval = monitoring_v3.TimeInterval(
        {
            "start_time": {
                "seconds": int(now - minutes * 60),
            },
            "end_time": {
                "seconds": int(now),
            },
        }
    )

    request = monitoring_v3.ListTimeSeriesRequest(
        name=project_name,
        filter=f'metric.type = "{METRIC_TYPE}"',
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )

    print(f"\nSeries encontradas en los últimos {minutes} minutos:\n")

    found = False

    for series in client.list_time_series(request=request):
        found = True
        labels = dict(series.metric.labels)
        service = labels.get("service", "unknown")
        environment = labels.get("environment", "unknown")

        points = list(series.points)
        if not points:
            continue

        latest = points[0]
        value = latest.value.double_value
        ts = latest.interval.end_time

        print(
            f"- service={service}, environment={environment}, "
            f"value={value:.2f}, timestamp={ts}"
        )

    if not found:
        print("No se han encontrado puntos todavía. Espera 1-2 minutos y vuelve a leer.")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Demo de arquitectura Cloud Monitoring para AIOps"
    )
    parser.add_argument("--write", action="store_true", help="Escribe métricas de ejemplo")
    parser.add_argument("--read", action="store_true", help="Lee métricas recientes")
    parser.add_argument("--minutes", type=int, default=15, help="Ventana de lectura en minutos")

    args = parser.parse_args()

    project_id = require_env("PROJECT_ID")
    environment = os.getenv("AIOPS_ENVIRONMENT", "lab")

    client = monitoring_v3.MetricServiceClient()
    project_name = get_project_name(project_id)

    try:
        if args.write:
            ensure_metric_descriptor(client, project_name)
            write_sample_scores(client, project_id, project_name, environment)

        if args.read:
            read_recent_scores(client, project_name, args.minutes)

        if not args.write and not args.read:
            print("Usa --write, --read o ambos.")

    except PermissionDenied as exc:
        print("Error de permisos.")
        print("Revisa que tienes roles de Monitoring suficientes sobre el proyecto.")
        print(f"Detalle técnico: {exc}")

    except GoogleAPICallError as exc:
        print("Error llamando a Cloud Monitoring.")
        print("Revisa que monitoring.googleapis.com esté habilitada y que ADC esté configurado.")
        print(f"Detalle técnico: {exc}")


if __name__ == "__main__":
    main()