import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from google.cloud import pubsub_v1


load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


PROJECT_ID = require_env("PROJECT_ID")
TOPIC_ID = os.getenv("AIOPS_TOPIC", "aiops-events")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)


SERVICES = [
    "checkout-api",
    "payments-api",
    "inventory-worker",
    "frontend-web",
]

ENVIRONMENTS = ["lab", "dev", "prod"]

METRICS = [
    "latency_ms",
    "error_rate",
    "cpu_utilization",
    "memory_utilization",
]


def build_metric_value(metric: str) -> tuple[float, float]:
    if metric == "latency_ms":
        return round(random.uniform(100, 2200), 2), 800.0

    if metric == "error_rate":
        return round(random.uniform(0.0, 0.25), 4), 0.05

    return round(random.uniform(20, 99), 2), 75.0


def classify_severity(metric: str, value: float) -> str:
    if metric == "latency_ms":
        if value >= 1500:
            return "critical"
        if value >= 800:
            return "warning"
        return "info"

    if metric == "error_rate":
        if value >= 0.15:
            return "critical"
        if value >= 0.05:
            return "warning"
        return "info"

    if value >= 90:
        return "critical"
    if value >= 75:
        return "warning"
    return "info"


def create_event() -> dict[str, Any]:
    metric = random.choice(METRICS)
    value, threshold = build_metric_value(metric)
    severity = classify_severity(metric, value)
    service = random.choice(SERVICES)
    environment = random.choice(ENVIRONMENTS)

    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "metric_observation",
        "source": "synthetic_aiops_lab",
        "project_id": PROJECT_ID,
        "environment": environment,
        "service": service,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "severity": severity,
        "correlation_id": f"corr-{service}-{uuid.uuid4()}",
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def publish_event(event: dict[str, Any]) -> str:
    data = json.dumps(event, ensure_ascii=False).encode("utf-8")

    future = publisher.publish(
        topic_path,
        data,
        event_type=event["event_type"],
        severity=event["severity"],
        environment=event["environment"],
        service=event["service"],
        metric=event["metric"],
    )

    return future.result(timeout=30)


def main() -> None:
    print(f"Publicando eventos en {topic_path}")

    for _ in range(20):
        event = create_event()
        message_id = publish_event(event)

        print(
            f"message_id={message_id} | "
            f"service={event['service']} | "
            f"env={event['environment']} | "
            f"metric={event['metric']} | "
            f"value={event['value']} | "
            f"severity={event['severity']}"
        )

        time.sleep(0.5)


if __name__ == "__main__":
    main()