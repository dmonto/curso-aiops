import os
import random
import time
from datetime import datetime, timezone

import google.cloud.logging
from google.cloud.logging_v2.handlers import StructuredLogHandler
import logging


PROJECT_ID = os.getenv("PROJECT_ID")
LOG_NAME = os.getenv("LOG_NAME", "aiops-lab")

if not PROJECT_ID:
    raise RuntimeError("Falta PROJECT_ID en variables de entorno.")


def build_operational_event(i: int) -> dict:
    service_name = random.choice(["checkout-api", "orders-worker", "inventory-api"])
    environment = random.choice(["dev", "test", "prod"])
    latency_ms = random.randint(80, 5000)
    error_rate = round(random.uniform(0.0, 0.25), 3)

    if latency_ms > 3000 or error_rate > 0.15:
        severity = "ERROR"
        event_type = "incident_candidate"
    elif latency_ms > 1500 or error_rate > 0.08:
        severity = "WARNING"
        event_type = "degradation_candidate"
    else:
        severity = "INFO"
        event_type = "normal_operation"

    return {
        "event_id": f"evt-{i:04d}",
        "event_type": event_type,
        "service_name": service_name,
        "environment": environment,
        "latency_ms": latency_ms,
        "error_rate": error_rate,
        "region": os.getenv("REGION", "europe-west1"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aiops_candidate": severity in ["WARNING", "ERROR"],
    }, severity


def main() -> None:
    client = google.cloud.logging.Client(project=PROJECT_ID)
    client.setup_logging(log_level=logging.INFO)

    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.INFO)

    print(f"Generando logs estructurados en project={PROJECT_ID}, log={LOG_NAME}")

    for i in range(1, 21):
        event, severity = build_operational_event(i)

        if severity == "ERROR":
            logger.error("Evento operativo candidato a incidente", extra={"json_fields": event})
        elif severity == "WARNING":
            logger.warning("Evento operativo candidato a degradación", extra={"json_fields": event})
        else:
            logger.info("Evento operativo normal", extra={"json_fields": event})

        print(f"{severity}: {event}")
        time.sleep(1)

    print("Finalizado.")


if __name__ == "__main__":
    main()