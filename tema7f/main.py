import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from google.cloud import pubsub_v1


PROJECT_ID = os.getenv("PROJECT_ID")
MITIGATION_TOPIC = os.getenv("MITIGATION_TOPIC", "aiops-mitigation-actions")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

publisher = pubsub_v1.PublisherClient()


def _json_log(payload: Dict[str, Any]) -> None:
    """
    Escribe un log estructurado.
    Cloud Logging lo interpretará como JSON si el runtime lo soporta.
    """
    logging.info(json.dumps(payload, ensure_ascii=False))


def _decode_pubsub_message(cloud_event) -> Dict[str, Any]:
    """
    Decodifica el mensaje Pub/Sub recibido por Cloud Functions Gen2.
    """
    message = cloud_event.data.get("message", {})
    data = message.get("data")

    if not data:
        raise ValueError("El mensaje Pub/Sub no contiene data")

    decoded = base64.b64decode(data).decode("utf-8")
    payload = json.loads(decoded)

    return payload


def _severity_to_number(severity: str) -> int:
    values = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }
    return values.get(severity.upper(), 0)


def _choose_mitigation(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide una acción de mitigación basada en una política simple.

    Esta función no ejecuta cambios directos sobre recursos.
    Genera una decisión trazable y publicable.
    """
    service = event.get("service", "unknown")
    environment = event.get("environment", "unknown")
    severity = event.get("severity", "LOW").upper()
    anomaly_type = event.get("anomaly_type", "UNKNOWN").upper()
    anomaly_score = float(event.get("anomaly_score", 0))
    error_rate = float(event.get("error_rate", 0))
    latency_ms = float(event.get("latency_ms", 0))

    decision = {
        "decision_timestamp": datetime.now(timezone.utc).isoformat(),
        "incident_id": event.get("incident_id"),
        "service": service,
        "environment": environment,
        "severity": severity,
        "anomaly_type": anomaly_type,
        "anomaly_score": anomaly_score,
        "dry_run": DRY_RUN,
        "allowed_to_execute": False,
        "action": "LOG_ONLY",
        "reason": "Anomalía registrada sin acción automática",
        "recommended_next_step": "Revisar dashboard y logs del servicio",
    }

    # No ejecutar mitigaciones reales fuera de producción.
    if environment != "prod":
        decision["reason"] = "Entorno no productivo: solo se registra la anomalía"
        return decision

    severity_number = _severity_to_number(severity)

    # Caso 1: errores altos tras degradación de aplicación.
    if anomaly_type == "ERROR_SPIKE" and severity_number >= 3:
        decision.update({
            "action": "CREATE_INCIDENT_AND_RECOMMEND_ROLLBACK",
            "allowed_to_execute": False,
            "reason": "Error rate alto en producción; rollback requiere aprobación",
            "recommended_next_step": (
                "Revisar último despliegue, logs de aplicación y preparar rollback supervisado"
            ),
        })
        return decision

    # Caso 2: saturación de recursos.
    if anomaly_type == "RESOURCE_SATURATION" and severity_number >= 3:
        decision.update({
            "action": "RECOMMEND_SCALE_OUT",
            "allowed_to_execute": False,
            "reason": "Saturación detectada; escalado automático no habilitado en laboratorio",
            "recommended_next_step": (
                "Validar límites de coste y aumentar capacidad si el servicio lo permite"
            ),
        })
        return decision

    # Caso 3: caída de tráfico.
    if anomaly_type == "TRAFFIC_DROP" and anomaly_score >= 75:
        decision.update({
            "action": "CHECK_TRAFFIC_ENTRYPOINTS",
            "allowed_to_execute": False,
            "reason": "Caída de tráfico detectada; no se debe reiniciar el backend automáticamente",
            "recommended_next_step": (
                "Revisar balanceador, DNS, frontend, rutas y errores de entrada"
            ),
        })
        return decision

    # Caso 4: latencia alta sin muchos errores.
    if anomaly_type == "LATENCY_DEGRADATION" and latency_ms > 0:
        decision.update({
            "action": "INVESTIGATE_LATENCY",
            "allowed_to_execute": False,
            "reason": "Latencia elevada; se requiere correlación con trazas y dependencias",
            "recommended_next_step": (
                "Revisar trazas, dependencias externas, base de datos y colas"
            ),
        })
        return decision

    # Caso 5: score muy alto aunque el tipo no sea conocido.
    if anomaly_score >= 90 and error_rate >= 0.03:
        decision.update({
            "action": "CREATE_ALERT",
            "allowed_to_execute": True,
            "reason": "Score alto y error_rate elevado; se genera alerta operativa",
            "recommended_next_step": "Escalar a revisión SRE",
        })
        return decision

    return decision


def _publish_mitigation_decision(decision: Dict[str, Any]) -> None:
    """
    Publica la decisión en un topic de mitigaciones.
    """
    if not PROJECT_ID:
        raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT en variables de entorno")

    topic_path = publisher.topic_path(PROJECT_ID, MITIGATION_TOPIC)
    data = json.dumps(decision, ensure_ascii=False).encode("utf-8")

    future = publisher.publish(
        topic_path,
        data,
        service=decision.get("service", "unknown"),
        environment=decision.get("environment", "unknown"),
        severity=decision.get("severity", "unknown"),
        action=decision.get("action", "unknown"),
    )

    message_id = future.result(timeout=30)
    decision["mitigation_message_id"] = message_id


def aiops_mitigator(cloud_event) -> None:
    """
    Entry point de Cloud Function Gen2.

    Trigger:
    Pub/Sub topic aiops-anomalies
    """
    try:
        event = _decode_pubsub_message(cloud_event)
        decision = _choose_mitigation(event)
        _publish_mitigation_decision(decision)

        _json_log({
            "event_type": "aiops_mitigation_decision",
            "status": "OK",
            "input_event": event,
            "decision": decision,
        })

    except Exception as ex:
        _json_log({
            "event_type": "aiops_mitigation_decision",
            "status": "ERROR",
            "error": str(ex),
        })
        raise