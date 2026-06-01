import base64
import json
import os
from types import SimpleNamespace

# Importante: definir variables antes de importar main.py
os.environ["PROJECT_ID"] = "asteci-capacitacion-ia"
os.environ["MITIGATION_TOPIC"] = "aiops-mitigation-actions"
os.environ["DRY_RUN"] = "true"

import main

def fake_publish(decision):
    """
    Sustituye la publicación real en Pub/Sub.
    Así podemos probar la función completa sin permisos cloud.
    """
    decision["mitigation_message_id"] = "local-test-message-id"

    print("\nDECISIÓN DE MITIGACIÓN GENERADA")
    print("--------------------------------")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


# Monkey patch: evitamos llamada real a Pub/Sub
main._publish_mitigation_decision = fake_publish


def build_fake_pubsub_cloudevent(payload: dict):
    encoded = base64.b64encode(
        json.dumps(payload).encode("utf-8")
    ).decode("utf-8")

    return SimpleNamespace(
        data={
            "message": {
                "data": encoded,
                "messageId": "local-message-001",
                "publishTime": "2026-06-01T10:00:00Z",
            },
            "subscription": "local-test-subscription",
        }
    )


event = {
    "incident_id": "INC-LOCAL-001",
    "service": "checkout-api",
    "environment": "prod",
    "severity": "HIGH",
    "anomaly_type": "LATENCY_DEGRADATION",
    "anomaly_score": 87,
    "latency_ms": 1850,
    "error_rate": 0.012,
}

cloud_event = build_fake_pubsub_cloudevent(event)

main.aiops_mitigator(cloud_event)