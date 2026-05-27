import json
import os
from datetime import datetime, timezone
from pathlib import Path
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
SUBSCRIPTION_ID = os.getenv("AIOPS_CRITICAL_SUB", "aiops-critical-sub")
INCIDENTS_FILE = Path("aiops_critical_incidents.jsonl")

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)


def decide_incident_action(event: dict[str, Any]) -> dict[str, Any]:
    service = event.get("service", "unknown")
    metric = event.get("metric", "unknown")
    value = event.get("value", "unknown")

    return {
        "action": "create_incident_candidate",
        "requires_human_review": True,
        "reason": f"Evento crítico detectado en {service}: {metric}={value}",
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def write_incident_candidate(event: dict[str, Any], decision: dict[str, Any]) -> None:
    record = {
        "event_id": event.get("event_id"),
        "correlation_id": event.get("correlation_id"),
        "service": event.get("service"),
        "metric": event.get("metric"),
        "value": event.get("value"),
        "severity": event.get("severity"),
        "decision": decision,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    with INCIDENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        event = json.loads(message.data.decode("utf-8"))
        decision = decide_incident_action(event)
        write_incident_candidate(event, decision)

        print("=" * 80)
        print("CRITICAL EVENT")
        print(f"message_id: {message.message_id}")
        print(f"service: {event.get('service')}")
        print(f"metric: {event.get('metric')}")
        print(f"value: {event.get('value')}")
        print(f"severity: {event.get('severity')}")
        print(f"action: {decision['action']}")
        print(f"human_review: {decision['requires_human_review']}")
        print(f"reason: {decision['reason']}")

        message.ack()

    except Exception as exc:
        print(f"Error en consumidor crítico: {exc}")
        message.nack()


def main() -> None:
    print(f"Escuchando eventos críticos en {subscription_path}")
    print("Pulsa Ctrl+C para detener.")

    future = subscriber.subscribe(subscription_path, callback=callback)

    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()
        print("Consumidor crítico detenido.")


if __name__ == "__main__":
    main()