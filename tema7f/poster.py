import json
import os
import uuid
from datetime import datetime, timezone

from google.cloud import pubsub_v1
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC_ID = os.getenv("AIOPS_ANOMALY_TOPIC", "aiops-anomalies")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def build_event(
    service: str,
    anomaly_type: str,
    severity: str,
    anomaly_score: float,
    latency_ms: float,
    error_rate: float,
) -> dict:
    return {
        "incident_id": f"inc-{uuid.uuid4().hex[:10]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": service,
        "environment": "prod",
        "region": os.getenv("LOCATION", "us-central1"),
        "anomaly_type": anomaly_type,
        "severity": severity,
        "anomaly_score": anomaly_score,
        "latency_ms": latency_ms,
        "error_rate": error_rate,
        "source": "tema7_publicar_anomalia.py",
        "detector_version": "thresholds-dinamicos-v1",
    }


def publish_event(event: dict) -> str:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

    data = json.dumps(event, ensure_ascii=False).encode("utf-8")

    future = publisher.publish(
        topic_path,
        data,
        service=event["service"],
        severity=event["severity"],
        anomaly_type=event["anomaly_type"],
    )

    return future.result(timeout=30)


def main() -> None:
    global PROJECT_ID
    PROJECT_ID = require_env("PROJECT_ID")

    examples = [
        build_event(
            service="checkout-api",
            anomaly_type="ERROR_SPIKE",
            severity="HIGH",
            anomaly_score=94,
            latency_ms=620,
            error_rate=0.08,
        ),
        build_event(
            service="catalog-api",
            anomaly_type="TRAFFIC_DROP",
            severity="MEDIUM",
            anomaly_score=82,
            latency_ms=180,
            error_rate=0.01,
        ),
        build_event(
            service="auth-api",
            anomaly_type="RESOURCE_SATURATION",
            severity="HIGH",
            anomaly_score=91,
            latency_ms=340,
            error_rate=0.035,
        ),
        build_event(
            service="billing-api",
            anomaly_type="LATENCY_DEGRADATION",
            severity="LOW",
            anomaly_score=61,
            latency_ms=410,
            error_rate=0.006,
        ),
    ]

    for event in examples:
        message_id = publish_event(event)
        print(
            f"Publicado message_id={message_id} "
            f"service={event['service']} "
            f"type={event['anomaly_type']} "
            f"severity={event['severity']}"
        )


if __name__ == "__main__":
    main()