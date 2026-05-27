import os
import json
from dotenv import load_dotenv
from google.cloud import aiplatform


load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


PROJECT_ID = require_env("PROJECT_ID")

# Usa VERTEX_REGION si ya lo tienes en el .env del curso.
# También acepto VERTEX_LOCATION por compatibilidad con otros scripts.
VERTEX_LOCATION = (
    os.getenv("VERTEX_REGION")
    or os.getenv("VERTEX_LOCATION")
    or "us-south1"
)

ENDPOINT_ID = "3838909664007815168"

ENDPOINT_RESOURCE_NAME = (
    f"projects/{PROJECT_ID}/locations/{VERTEX_LOCATION}/endpoints/{ENDPOINT_ID}"
)


def main() -> None:
    aiplatform.init(
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
    )

    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_RESOURCE_NAME)

    print(f"Proyecto: {PROJECT_ID}")
    print(f"Región: {VERTEX_LOCATION}")
    print(f"Endpoint: {ENDPOINT_RESOURCE_NAME}")

    instances = [
        {
            "cpu_utilization": 92.0,
            "memory_utilization": 81.0,
            "latency_ms": 1850.0,
            "error_rate": 0.12,
            "recent_alerts": 4,
            "deploy_in_last_hour": 1,
        }
    ]

    prediction = endpoint.predict(instances=instances)

    print("\nRespuesta completa:")
    print(prediction)

    print("\nPredicciones:")
    print(json.dumps(prediction.predictions, indent=2, ensure_ascii=False))

    if prediction.deployed_model_id:
        print(f"\nDeployed model id: {prediction.deployed_model_id}")


if __name__ == "__main__":
    main()