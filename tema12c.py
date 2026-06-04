import argparse
import datetime as dt
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai.types import HttpOptions
from google.cloud import bigquery


PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
BQ_DATASET = os.getenv("BQ_DATASET", "aiops_lab")

PROMPT_VERSION = "support-assistant-v1"

if not PROJECT_ID:
    raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT")

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


CLASSIFICATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING"},
        "service_guess": {"type": "STRING"},
        "severity_guess": {"type": "STRING"},
        "category": {"type": "STRING"},
        "missing_information": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "should_escalate": {"type": "BOOLEAN"},
        "escalation_target": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": [
        "intent",
        "service_guess",
        "severity_guess",
        "category",
        "missing_information",
        "should_escalate",
        "escalation_target",
        "confidence",
    ],
}


ASSISTANT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "user_response": {"type": "STRING"},
        "technical_diagnosis": {"type": "STRING"},
        "recommended_checks": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "ticket_title": {"type": "STRING"},
        "ticket_description": {"type": "STRING"},
        "severity": {"type": "STRING"},
        "affected_service": {"type": "STRING"},
        "needs_human_review": {"type": "BOOLEAN"},
        "escalation_target": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "missing_information": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "user_response",
        "technical_diagnosis",
        "recommended_checks",
        "ticket_title",
        "ticket_description",
        "severity",
        "affected_service",
        "needs_human_review",
        "escalation_target",
        "confidence",
        "missing_information",
    ],
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED_TOKEN]", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key=)[A-Za-z0-9._\-]+", r"\1[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"(?i)(password=)[^&\s]+", r"\1[REDACTED_PASSWORD]", redacted)
    redacted = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]", redacted)
    return redacted


def client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )    


def ensure_lab_tables() -> None:
    bq = bigquery.Client(project=PROJECT_ID)

    dataset_id = f"{PROJECT_ID}.{BQ_DATASET}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "EU"
    bq.create_dataset(dataset, exists_ok=True)

    runbooks_table = bigquery.Table(
        f"{dataset_id}.support_runbooks",
        schema=[
            bigquery.SchemaField("runbook_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("service_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("symptoms", "STRING"),
            bigquery.SchemaField("checks", "STRING"),
            bigquery.SchemaField("safe_actions", "STRING"),
            bigquery.SchemaField("escalation_team", "STRING"),
        ],
    )
    bq.create_table(runbooks_table, exists_ok=True)

    interactions_table = bigquery.Table(
        f"{dataset_id}.support_assistant_interactions",
        schema=[
            bigquery.SchemaField("interaction_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING"),
            bigquery.SchemaField("user_message", "STRING"),
            bigquery.SchemaField("intent", "STRING"),
            bigquery.SchemaField("service_guess", "STRING"),
            bigquery.SchemaField("severity", "STRING"),
            bigquery.SchemaField("needs_human_review", "BOOLEAN"),
            bigquery.SchemaField("escalation_target", "STRING"),
            bigquery.SchemaField("confidence", "FLOAT"),
            bigquery.SchemaField("response_json", "STRING"),
            bigquery.SchemaField("prompt_version", "STRING"),
            bigquery.SchemaField("model_id", "STRING"),
        ],
    )
    bq.create_table(interactions_table, exists_ok=True)


def seed_runbooks_if_empty() -> None:
    bq = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.support_runbooks"

    query = f"SELECT COUNT(*) AS n FROM `{table_id}`"
    count = list(bq.query(query).result())[0]["n"]

    if count > 0:
        return

    rows = [
        {
            "runbook_id": "rb-checkout-500",
            "service_name": "checkout-api",
            "title": "Errores 500 en checkout-api",
            "symptoms": "Aumento de HTTP 500, latencia alta, fallos en POST /checkout.",
            "checks": json.dumps(
                [
                    "Revisar logs ERROR de checkout-api.",
                    "Comprobar métricas 5xx en Cloud Monitoring.",
                    "Verificar conexiones activas en Cloud SQL.",
                    "Comprobar si hubo despliegue reciente.",
                    "Revisar backlog en orders-events.",
                ],
                ensure_ascii=False,
            ),
            "safe_actions": json.dumps(
                [
                    "Preparar resumen para SRE.",
                    "Solicitar hora de inicio e impacto aproximado.",
                    "No ejecutar rollback sin aprobación.",
                ],
                ensure_ascii=False,
            ),
            "escalation_team": "sre-platform",
        },
        {
            "runbook_id": "rb-pubsub-backlog",
            "service_name": "orders-consumer",
            "title": "Backlog elevado en consumidor Pub/Sub",
            "symptoms": "Aumento de mensajes pendientes, consumidores lentos o fallando.",
            "checks": json.dumps(
                [
                    "Revisar backlog de la suscripción.",
                    "Comprobar errores del consumidor.",
                    "Validar latencia de procesamiento.",
                    "Revisar cuota y escalado del servicio consumidor.",
                ],
                ensure_ascii=False,
            ),
            "safe_actions": json.dumps(
                [
                    "Escalar si el backlog crece durante más de 15 minutos.",
                    "No modificar concurrencia en producción sin revisión.",
                ],
                ensure_ascii=False,
            ),
            "escalation_team": "sre-messaging",
        },
        {
            "runbook_id": "rb-billing-latency",
            "service_name": "billing-worker",
            "title": "Latencia alta en billing-worker",
            "symptoms": "Procesos de facturación lentos, timeouts, colas acumuladas.",
            "checks": json.dumps(
                [
                    "Revisar latencia p95/p99.",
                    "Comprobar errores de dependencia externa.",
                    "Verificar CPU y memoria.",
                    "Revisar cambios recientes en configuración.",
                ],
                ensure_ascii=False,
            ),
            "safe_actions": json.dumps(
                [
                    "Generar informe técnico si afecta a cierre contable.",
                    "Escalar a equipo billing si hay impacto funcional.",
                ],
                ensure_ascii=False,
            ),
            "escalation_team": "billing-ops",
        },
    ]

    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"Error insertando runbooks de ejemplo: {errors}")


def classify_message(user_message: str) -> Dict[str, Any]:
    prompt = f"""
Actúa como sistema de triage para un asistente interno de soporte AIOps.

Clasifica la solicitud del usuario.

Intenciones posibles:
- report_incident
- ask_diagnosis
- ask_runbook
- ask_status
- ask_escalation
- ask_user_communication
- other

Categorías posibles:
- application
- database
- messaging
- infrastructure
- security
- cost
- unknown

Reglas:
- Usa P1 solo si hay caída crítica o impacto amplio.
- Usa P2 si hay degradación importante en servicio productivo.
- Usa P3 si parece incidencia limitada.
- Usa P4 si es consulta o baja urgencia.
- Si faltan datos, inclúyelos en missing_information.
- No inventes datos no presentes.
- Devuelve solo JSON válido.

Mensaje del usuario:
{redact_sensitive_text(user_message)}
""".strip()

    cli = client()
    response = cli.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": CLASSIFICATION_SCHEMA,
        },
    )

    if not response.text:
        raise RuntimeError("El modelo no devolvió clasificación.")

    return json.loads(response.text)


def get_runbooks(service_guess: str, category: str) -> List[Dict[str, Any]]:
    bq = bigquery.Client(project=PROJECT_ID)
    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.support_runbooks`"

    query = f"""
    SELECT
      runbook_id,
      service_name,
      title,
      symptoms,
      checks,
      safe_actions,
      escalation_team
    FROM {table_id}
    WHERE LOWER(service_name) = LOWER(@service_name)
       OR LOWER(symptoms) LIKE CONCAT('%', LOWER(@category), '%')
    LIMIT 5
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("service_name", "STRING", service_guess or ""),
            bigquery.ScalarQueryParameter("category", "STRING", category or ""),
        ]
    )

    rows = [dict(row) for row in bq.query(query, job_config=job_config).result()]
    return rows


def get_recent_log_summaries(service_guess: str) -> List[Dict[str, Any]]:
    bq = bigquery.Client(project=PROJECT_ID)
    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.log_summaries`"

    query = f"""
    SELECT
      created_at,
      service_name,
      severity,
      confidence,
      needs_human_review,
      summary,
      affected_services_json,
      observed_patterns_json
    FROM {table_id}
    WHERE LOWER(service_name) = LOWER(@service_name)
    ORDER BY created_at DESC
    LIMIT 3
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("service_name", "STRING", service_guess or "")
        ]
    )

    try:
        rows = [dict(row) for row in bq.query(query, job_config=job_config).result()]
    except Exception:
        rows = []

    return rows


def get_recent_reports(service_guess: str) -> List[Dict[str, Any]]:
    bq = bigquery.Client(project=PROJECT_ID)
    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.technical_reports`"

    query = f"""
    SELECT
      created_at,
      incident_id,
      report_id,
      service_name,
      severity,
      confidence,
      human_review_required,
      title,
      gcs_uri
    FROM {table_id}
    WHERE LOWER(service_name) = LOWER(@service_name)
    ORDER BY created_at DESC
    LIMIT 3
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("service_name", "STRING", service_guess or "")
        ]
    )

    try:
        rows = [dict(row) for row in bq.query(query, job_config=job_config).result()]
    except Exception:
        rows = []

    return rows


def build_context(user_message: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    service_guess = classification.get("service_guess", "")
    category = classification.get("category", "")

    return {
        "user_message": redact_sensitive_text(user_message),
        "classification": classification,
        "runbooks": get_runbooks(service_guess, category),
        "recent_log_summaries": get_recent_log_summaries(service_guess),
        "recent_technical_reports": get_recent_reports(service_guess),
        "assistant_policy": {
            "do_not_execute_changes": True,
            "do_not_request_passwords_or_tokens": True,
            "mark_human_review_for_p1_p2": True,
            "separate_evidence_from_hypothesis": True,
            "ask_for_missing_information_when_needed": True,
        },
    }


def generate_support_response(context: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Actúa como asistente interno de soporte para un entorno AIOps en Google Cloud.

Debes ayudar al usuario con una respuesta clara, segura y accionable.

Reglas:
- Usa solo el contexto proporcionado.
- No inventes causa raíz.
- No ejecutes cambios ni digas que los has ejecutado.
- No pidas contraseñas, tokens ni claves.
- Si faltan datos, pídelos de forma concreta.
- Si la severidad es P1 o P2, marca needs_human_review=true.
- Si propones rollback, escalado o cambios de configuración, exige revisión humana.
- Genera un borrador de ticket útil.
- Devuelve únicamente JSON válido.

Contexto:
{json.dumps(context, indent=2, ensure_ascii=False, default=str)}
""".strip()

    cli = client()
    response = cli.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": ASSISTANT_RESPONSE_SCHEMA,
        },
    )

    if not response.text:
        raise RuntimeError("El modelo no devolvió respuesta.")

    result = json.loads(response.text)

    severity = result.get("severity", "")
    confidence = float(result.get("confidence", 0))

    if severity in ["P1", "P2"] or confidence < 0.65:
        result["needs_human_review"] = True

    return result


def save_interaction(
    user_id: str,
    user_message: str,
    classification: Dict[str, Any],
    response: Dict[str, Any],
) -> None:
    bq = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.support_assistant_interactions"

    raw = f"{user_id}-{user_message}-{utc_now().isoformat()}"
    interaction_id = hashlib.sha256(raw.encode()).hexdigest()[:20]

    row = {
        "interaction_id": interaction_id,
        "created_at": utc_now().isoformat(),
        "user_id": user_id,
        "user_message": redact_sensitive_text(user_message),
        "intent": classification.get("intent"),
        "service_guess": classification.get("service_guess"),
        "severity": response.get("severity"),
        "needs_human_review": bool(response.get("needs_human_review", True)),
        "escalation_target": response.get("escalation_target"),
        "confidence": float(response.get("confidence", 0)),
        "response_json": json.dumps(response, ensure_ascii=False),
        "prompt_version": PROMPT_VERSION,
        "model_id": MODEL_ID,
    }

    errors = bq.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Error guardando interacción: {errors}")


def print_response(response: Dict[str, Any]) -> None:
    print("\n=== Respuesta al usuario ===")
    print(response.get("user_response", ""))

    print("\n=== Diagnóstico técnico ===")
    print(response.get("technical_diagnosis", ""))

    print("\n=== Comprobaciones recomendadas ===")
    for item in response.get("recommended_checks", []):
        print(f"- {item}")

    print("\n=== Borrador de ticket ===")
    print(f"Título: {response.get('ticket_title', '')}")
    print(response.get("ticket_description", ""))

    print("\n=== Control ===")
    print(f"Severidad: {response.get('severity')}")
    print(f"Servicio afectado: {response.get('affected_service')}")
    print(f"Revisión humana: {response.get('needs_human_review')}")
    print(f"Escalado: {response.get('escalation_target')}")
    print(f"Confianza: {response.get('confidence')}")

    missing = response.get("missing_information", [])
    if missing:
        print("\n=== Información faltante ===")
        for item in missing:
            print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Asistente interno de soporte AIOps")
    parser.add_argument("--user-id", default="student-user", help="Identificador del usuario")
    parser.add_argument("--message", default=None, help="Mensaje del usuario")
    args = parser.parse_args()

    ensure_lab_tables()
    seed_runbooks_if_empty()

    if args.message:
        user_message = args.message
    else:
        user_message = input("Describe la solicitud de soporte: ").strip()

    if not user_message:
        print("No se recibió mensaje.")
        return

    classification = classify_message(user_message)
    context = build_context(user_message, classification)
    response = generate_support_response(context)
    save_interaction(args.user_id, user_message, classification, response)

    print_response(response)


if __name__ == "__main__":
    main()