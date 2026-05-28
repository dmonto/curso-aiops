import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from google.cloud import pubsub_v1
from google.cloud import aiplatform_v1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from dotenv import load_dotenv
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("VERTEX_LOCATION", "europe-west1")
INPUT_SUBSCRIPTION = os.getenv("INPUT_SUBSCRIPTION", "aiops-enriched-events-sub")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "aiops-inference-results")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID", "").strip()

EVIDENCE_FILE = os.getenv("EVIDENCE_FILE", "realtime_inference_results.jsonl")


SEVERITY_SCORE = {
    "DEBUG": 0,
    "INFO": 0,
    "WARN": 1,
    "WARNING": 1,
    "ERROR": 3,
    "CRITICAL": 5,
}

SERVICE_TIER_SCORE = {
    "internal": 1,
    "standard": 2,
    "critical": 3,
}


if not PROJECT_ID:
    raise RuntimeError("Falta PROJECT_ID.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_severity(value: Any) -> str:
    severity = str(value or "INFO").upper()
    if severity == "WARNING":
        return "WARN"
    if severity in ["ALERT", "EMERGENCY", "FATAL"]:
        return "CRITICAL"
    return severity


def build_features(event: Dict[str, Any]) -> Dict[str, Any]:
    severity = normalize_severity(event.get("severity"))
    service_tier = str(event.get("service_tier", "internal")).lower()
    environment = str(event.get("environment", "unknown")).lower()

    latency_ms = to_int(event.get("latency_ms"))
    error_rate = to_float(event.get("error_rate"))

    expected_latency_ms = to_int(event.get("expected_latency_ms"))
    max_error_rate = to_float(event.get("max_error_rate"))

    latency_ratio = round(latency_ms / expected_latency_ms, 4) if expected_latency_ms else 0.0
    error_ratio = round(error_rate / max_error_rate, 4) if max_error_rate else 0.0

    return {
        "severity_score": SEVERITY_SCORE.get(severity, 0),
        "is_prod": 1 if environment == "prod" else 0,
        "service_tier_score": SERVICE_TIER_SCORE.get(service_tier, 1),
        "customer_facing": 1 if bool(event.get("customer_facing")) else 0,
        "maintenance_window": 1 if bool(event.get("maintenance_window")) else 0,
        "latency_ms": latency_ms,
        "error_rate": error_rate,
        "latency_over_expected_ratio": latency_ratio,
        "error_rate_over_expected_ratio": error_ratio,
    }


def should_run_inference(event: Dict[str, Any], features: Dict[str, Any]) -> bool:
    severity = normalize_severity(event.get("severity"))

    return (
        severity in ["WARN", "ERROR", "CRITICAL"]
        or features["is_prod"] == 1
        or features["service_tier_score"] >= 3
        or features["latency_over_expected_ratio"] >= 2
        or features["error_rate_over_expected_ratio"] >= 2
    )


def local_predict(features: Dict[str, Any]) -> Dict[str, Any]:
    score = 0

    score += features["severity_score"] * 12
    score += features["is_prod"] * 15
    score += features["service_tier_score"] * 8
    score += features["customer_facing"] * 10

    if features["latency_over_expected_ratio"] >= 2:
        score += 15
    if features["latency_over_expected_ratio"] >= 4:
        score += 20

    if features["error_rate_over_expected_ratio"] >= 2:
        score += 15
    if features["error_rate_over_expected_ratio"] >= 4:
        score += 20

    probability = min(score / 100, 0.99)

    if probability >= 0.90:
        priority = "P1"
        incident_type = "major_incident"
    elif probability >= 0.75:
        priority = "P2"
        incident_type = "incident_candidate"
    elif probability >= 0.55:
        priority = "P3"
        incident_type = "degradation_candidate"
    else:
        priority = "P4"
        incident_type = "operational_event"

    adjusted_reason = "none"

    if features["maintenance_window"] == 1:
        if priority == "P1":
            priority = "P2"
            adjusted_reason = "maintenance_window_active"
        elif priority == "P2":
            priority = "P3"
            adjusted_reason = "maintenance_window_active"

    action = {
        "P1": "page_owner_immediately",
        "P2": "create_incident_and_notify_owner",
        "P3": "monitor_and_group",
        "P4": "store_only",
    }[priority]

    return {
        "incident_probability": round(probability, 4),
        "predicted_priority": priority,
        "predicted_incident_type": incident_type,
        "recommended_action": action,
        "priority_adjusted_reason": adjusted_reason,
        "prediction_source": "local_dry_run",
        "prediction_status": "ok",
    }


class VertexPredictor:
    def __init__(self, project_id: str, region: str, endpoint_id: str):
        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id

        if endpoint_id:
            api_endpoint = f"{region}-aiplatform.googleapis.com"
            self.client = aiplatform_v1.PredictionServiceClient(
                client_options={"api_endpoint": api_endpoint}
            )
            self.endpoint_path = (
                f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"
            )
        else:
            self.client = None
            self.endpoint_path = None

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not self.endpoint_id:
            return local_predict(features)

        if self.client is None or self.endpoint_path is None:
            raise RuntimeError("Cliente de Vertex AI no inicializado.")

        instance = json_format.ParseDict(features, Value())

        response = self.client.predict(
            endpoint=self.endpoint_path,
            instances=[instance],
        )

        if not response.predictions:
            raise RuntimeError("Vertex AI no devolvió predicción.")

        prediction = json_format.MessageToDict(response.predictions[0])

        return {
            "incident_probability": to_float(prediction.get("incident_probability")),
            "predicted_priority": str(prediction.get("predicted_priority", "UNKNOWN")),
            "predicted_incident_type": str(
                prediction.get("predicted_incident_type", "unknown")
            ),
            "recommended_action": str(prediction.get("recommended_action", "store_only")),
            "priority_adjusted_reason": str(
                prediction.get("priority_adjusted_reason", "none")
            ),
            "prediction_source": "vertex_ai_endpoint",
            "prediction_status": "ok",
        }


def apply_operational_policy(
    event: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Capa de política posterior a la predicción.
    Aquí podríamos añadir controles de seguridad, umbrales de confianza,
    ventanas de mantenimiento, allowlist de servicios, etc.
    """
    priority = prediction.get("predicted_priority", "UNKNOWN")
    action = prediction.get("recommended_action", "store_only")

    if event.get("environment") != "prod" and priority == "P1":
        priority = "P2"
        action = "create_incident_and_notify_owner"

    return {
        **prediction,
        "final_priority": priority,
        "final_action": action,
    }


def publish_result(
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
    result: Dict[str, Any],
) -> str:
    data = json.dumps(result, ensure_ascii=False).encode("utf-8")
    future = publisher.publish(topic_path, data)
    return future.result()


def append_evidence(result: Dict[str, Any]) -> None:
    with open(EVIDENCE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def process_event(
    event: Dict[str, Any],
    predictor: VertexPredictor,
) -> Dict[str, Any]:
    start = time.perf_counter()

    features = build_features(event)

    if not should_run_inference(event, features):
        prediction = {
            "incident_probability": 0.0,
            "predicted_priority": "P4",
            "predicted_incident_type": "bypass",
            "recommended_action": "store_only",
            "priority_adjusted_reason": "bypass_gating",
            "prediction_source": "gating",
            "prediction_status": "bypassed",
        }
    else:
        try:
            prediction = predictor.predict(features)
        except Exception as exc:
            prediction = {
                "incident_probability": 0.5,
                "predicted_priority": "P3",
                "predicted_incident_type": "prediction_unavailable",
                "recommended_action": "monitor_and_group",
                "priority_adjusted_reason": "vertex_error_fallback",
                "prediction_source": "fallback",
                "prediction_status": "fallback",
                "prediction_error": str(exc),
            }

    policy_result = apply_operational_policy(event, prediction)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "inference_at": utc_now(),
        "event_id": event.get("event_id", ""),
        "event_timestamp": event.get("event_timestamp", ""),
        "service_name": event.get("service_name", "unknown"),
        "environment": event.get("environment", "unknown"),
        "severity": normalize_severity(event.get("severity")),
        "service_tier": event.get("service_tier", "unknown"),
        "customer_facing": bool(event.get("customer_facing", False)),
        "correlation_id": event.get("correlation_id", ""),
        "features": features,
        **policy_result,
        "inference_latency_ms": elapsed_ms,
    }


def main() -> None:
    subscriber = pubsub_v1.SubscriberClient()
    publisher = pubsub_v1.PublisherClient()

    subscription_path = subscriber.subscription_path(PROJECT_ID, INPUT_SUBSCRIPTION)
    output_topic_path = publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC)

    predictor = VertexPredictor(
        project_id=PROJECT_ID,
        region=REGION,
        endpoint_id=VERTEX_ENDPOINT_ID,
    )

    mode = "vertex" if VERTEX_ENDPOINT_ID else "local_dry_run"
    print(f"Escuchando: {subscription_path}")
    print(f"Publicando resultados en: {output_topic_path}")
    print(f"Modo de inferencia: {mode}")
    print("Pulsa Ctrl+C para detener.")

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        try:
            event = json.loads(message.data.decode("utf-8"))
            result = process_event(event, predictor)

            message_id = publish_result(publisher, output_topic_path, result)
            append_evidence(result)

            print(
                f"[{result['final_priority']}] "
                f"{result['service_name']} "
                f"prob={result['incident_probability']} "
                f"action={result['final_action']} "
                f"latency={result['inference_latency_ms']}ms "
                f"published={message_id}"
            )

            message.ack()

        except Exception as exc:
            print(f"Error procesando mensaje: {exc}")
            message.nack()

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        print("Worker detenido.")


if __name__ == "__main__":
    main()