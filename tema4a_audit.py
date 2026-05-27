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
SUBSCRIPTION_ID = os.getenv("AIOPS_AUDIT_SUB", "aiops-audit-sub")
AUDIT_FILE = Path("aiops_pubsub_audit.jsonl")

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)


def write_audit(event: dict[str, Any], message_id: str, attributes: dict[str, str]) -> None:
    record = {
        "message_id": message_id,
        "attributes": attributes,
        "event": event,
        "consumer": "audit",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        event = json.loads(message.data.decode("utf-8"))
        attributes = dict(message.attributes)

        write_audit(
            event=event,
            message_id=message.message_id,
            attributes=attributes,
        )

        print(
            f"AUDIT | message_id={message.message_id} | "
            f"service={event.get('service')} | "
            f"metric={event.get('metric')} | "
            f"severity={event.get('severity')}"
        )

        message.ack()

    except Exception as exc:
        print(f"Error en consumidor de auditoría: {exc}")
        message.nack()


def main() -> None:
    print(f"Escuchando auditoría en {subscription_path}")
    print("Pulsa Ctrl+C para detener.")

    future = subscriber.subscribe(subscription_path, callback=callback)

    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()
        print("Consumidor de auditoría detenido.")


if __name__ == "__main__":
    main()