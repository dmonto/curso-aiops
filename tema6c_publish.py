import json
import os
import random
import time
from datetime import datetime, timezone

from google.cloud import pubsub_v1
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "aiops-enriched-events")

if not PROJECT_ID:
    raise RuntimeError("Falta PROJECT_ID.")


SERVICES = [
    {
        "service_name": "checkout-api",
        "service_tier": "critical",
        "customer_facing": True,
        "expected_latency_ms": 1200,
        "max_error_rate": 0.05,
    },
    {
        "service_name": "payments-api",
        "service_tier": "critical",
        "customer_facing": True,
        "expected_latency_ms": 900,
        "max_error_rate": 0.03,
    },
    {
        "service_name": "orders-worker",
        "service_tier": "standard",
        "customer_facing": False,
        "expected_latency_ms": 2500,
        "max_error_rate": 0.08,
    },
    {
        "service_name": "notifications-worker",
        "service_tier": "internal",
        "customer_facing": False,
        "expected_latency_ms": 3000,
        "max_error_rate": 0.10,
    },
]


def build_event(i: int) -> dict:
    svc = random.choice(SERVICES)

    latency_ms = random.randint(100, 7000)
    error_rate = round(random.uniform(0.0, 0.35), 3)
    environment = random.choice(["dev", "test", "prod"])

    if latency_ms > 5000 or error_rate > 0.22:
        severity = "ERROR"
    elif latency_ms > 2500 or error_rate > 0.10:
        severity = "WARN"
    else:
        severity = "INFO"

    return {
        "event_id": f"evt-{i:05d}",
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "service_name": svc["service_name"],
        "environment": environment,
        "severity": severity,
        "service_tier": svc["service_tier"],
        "customer_facing": svc["customer_facing"],
        "latency_ms": latency_ms,
        "error_rate": error_rate,
        "expected_latency_ms": svc["expected_latency_ms"],
        "max_error_rate": svc["max_error_rate"],
        "correlation_id": f"{svc['service_name']}-{i:05d}",
        "maintenance_window": random.choice([False, False, False, True]),
        "message_clean": "Synthetic event for realtime inference lab",
    }


def main() -> None:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, INPUT_TOPIC)

    total = int(os.getenv("TOTAL_EVENTS", "30"))

    print(f"Publicando {total} eventos en {topic_path}")

    for i in range(1, total + 1):
        event = build_event(i)
        data = json.dumps(event).encode("utf-8")
        future = publisher.publish(topic_path, data)
        message_id = future.result()

        print(f"Publicado {message_id}: {event}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()