import os
import json
from google.cloud import aiplatform
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "asteci-capacitacion-ia")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

ENDPOINT_ID = "5605407585181106176"


def main() -> None:
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
    )

    endpoint_name = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
    )

    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

    instance = {
        "timestamp": "2026-06-02T10:30:00",
        "service": "checkout-api",
        "environment": "prod",

        # AutoML las está esperando como categóricas/string
        "hour": "10",
        "day_of_week": "1",

        # Numéricas reales
        "request_count": 6200.0,
        "latency_ms": 730.0,
        "error_rate": 0.085,
        "cpu_percent": 88.0,
        "memory_percent": 82.0,
        "minutes_since_deployment": "45.0",

        # Solo si entrenaste el modelo dejando incident_type como feature
        "incident_type": "UNKNOWN",
    }

    response = endpoint.predict(instances=[instance])
    print(response.predictions)

    response = endpoint.predict(instances=[instance])

    print("\nRespuesta completa:")
    print(response)

    print("\nPredicciones:")
    print(json.dumps(response.predictions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()