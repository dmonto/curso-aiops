import datetime as dt
import hashlib
import json
import os
import re
from typing import Any, Dict, List

from google import genai
from google.genai.types import HttpOptions
from google.cloud import bigquery


PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("GENAI_REGION", "global")
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

BQ_DATASET = os.getenv("DATASET_ID", "aiops_lab")
BQ_TABLE = os.getenv("LLM_TABLE", "llm_incident_summaries")


if not PROJECT_ID:
    raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT")


os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "incident_title": {"type": "STRING"},
        "severity": {"type": "STRING"},
        "affected_services": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "probable_cause": {"type": "STRING"},
        "supporting_evidence": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "recommended_actions": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "needs_human_approval": {"type": "BOOLEAN"},
        "confidence": {"type": "NUMBER"},
        "missing_information": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "incident_title",
        "severity",
        "affected_services",
        "probable_cause",
        "supporting_evidence",
        "recommended_actions",
        "needs_human_approval",
        "confidence",
        "missing_information",
    ],
}


def redact_sensitive_text(text: str) -> str:
    """
    Redacta patrones simples antes de enviar contexto al modelo.
    En producción se ampliaría con DLP, clasificación y reglas corporativas.
    """
    patterns = [
        (r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED_TOKEN]"),
        (r"(?i)(api[_-]?key=)[A-Za-z0-9._\-]+", r"\1[REDACTED_API_KEY]"),
        (r"(?i)(password=)[^&\s]+", r"\1[REDACTED_PASSWORD]"),
        (r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]"),
    ]

    redacted = text
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)

    return redacted


def sample_operational_events() -> List[Dict[str, Any]]:
    now = dt.datetime.now(dt.timezone.utc)

    return [
        {
            "timestamp": (now - dt.timedelta(minutes=18)).isoformat(),
            "source": "cloud-monitoring",
            "service": "checkout-api",
            "severity": "WARNING",
            "message": "Alert triggered: HTTP 5xx rate above 8% for service checkout-api.",
        },
        {
            "timestamp": (now - dt.timedelta(minutes=16)).isoformat(),
            "source": "cloud-logging",
            "service": "checkout-api",
            "severity": "ERROR",
            "message": "POST /checkout failed with status=500 latency_ms=2310 trace_id=trc-8812.",
        },
        {
            "timestamp": (now - dt.timedelta(minutes=15)).isoformat(),
            "source": "cloud-logging",
            "service": "checkout-api",
            "severity": "ERROR",
            "message": "Database connection error: too many connections for cloud-sql-orders.",
        },
        {
            "timestamp": (now - dt.timedelta(minutes=13)).isoformat(),
            "source": "deployment",
            "service": "checkout-api",
            "severity": "INFO",
            "message": "New revision deployed: checkout-api-00042. Previous revision: checkout-api-00041.",
        },
        {
            "timestamp": (now - dt.timedelta(minutes=10)).isoformat(),
            "source": "cloud-logging",
            "service": "checkout-api",
            "severity": "ERROR",
            "message": "Retry storm detected. authorization: Bearer abc.def.secret-token",
        },
        {
            "timestamp": (now - dt.timedelta(minutes=7)).isoformat(),
            "source": "cloud-monitoring",
            "service": "cloud-sql-orders",
            "severity": "WARNING",
            "message": "Cloud SQL active connections at 96% of configured limit.",
        },
        {
            "timestamp": (now - dt.timedelta(minutes=5)).isoformat(),
            "source": "pubsub",
            "service": "orders-events",
            "severity": "WARNING",
            "message": "Subscription backlog increased from 1.2k to 7.9k messages in 10 minutes.",
        },
    ]


def build_prompt(events: List[Dict[str, Any]]) -> str:
    event_lines = []

    for idx, event in enumerate(events, start=1):
        line = (
            f"{idx}. [{event['timestamp']}] "
            f"source={event['source']} "
            f"service={event['service']} "
            f"severity={event['severity']} "
            f"message={event['message']}"
        )
        event_lines.append(redact_sensitive_text(line))

    runbook = """
Runbook resumido:
- Si aumentan los 5xx justo después de un despliegue, comparar métricas antes/después de la nueva revisión.
- Si aparecen errores de conexión a Cloud SQL, revisar conexiones activas, pool de conexiones y límites.
- Si hay retry storm, revisar timeouts, backoff exponencial y circuit breaker.
- Rollback, escalado o cambios de configuración requieren aprobación humana salvo política explícita.
"""

    return f"""
Actúa como asistente SRE para un entorno AIOps en Google Cloud.

Tarea:
Analiza las evidencias operativas y devuelve un diagnóstico inicial en JSON.

Reglas:
- Usa solo las evidencias incluidas.
- Distingue evidencia de hipótesis.
- No inventes recursos, equipos ni métricas.
- No propongas acciones destructivas.
- Las acciones de rollback, escalado o cambio de configuración requieren aprobación humana.
- Si falta información, indícalo en missing_information.
- confidence debe estar entre 0 y 1.

{runbook}

Evidencias:
{chr(10).join(event_lines)}
""".strip()


def analyze_with_llm(prompt: str) -> Dict[str, Any]:
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
        },
    )

    if not response.text:
        raise RuntimeError("El modelo no devolvió texto.")

    return json.loads(response.text)


def ensure_bigquery_table() -> str:
    bq = bigquery.Client(project=PROJECT_ID)

    dataset_id = f"{PROJECT_ID}.{BQ_DATASET}"
    table_id = f"{dataset_id}.{BQ_TABLE}"

    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "EU"

    try:
        bq.create_dataset(dataset, exists_ok=True)
    except Exception as exc:
        raise RuntimeError(f"No se pudo crear/verificar el dataset {dataset_id}: {exc}") from exc

    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("model_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("incident_title", "STRING"),
        bigquery.SchemaField("severity", "STRING"),
        bigquery.SchemaField("confidence", "FLOAT"),
        bigquery.SchemaField("needs_human_approval", "BOOLEAN"),
        bigquery.SchemaField("affected_services_json", "STRING"),
        bigquery.SchemaField("recommended_actions_json", "STRING"),
        bigquery.SchemaField("full_result_json", "STRING"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    bq.create_table(table, exists_ok=True)

    return table_id


def write_result_to_bigquery(result: Dict[str, Any], table_id: str) -> None:
    bq = bigquery.Client(project=PROJECT_ID)

    run_id = hashlib.sha256(
        f"{dt.datetime.now(dt.timezone.utc).isoformat()}-{result['incident_title']}".encode()
    ).hexdigest()[:16]

    row = {
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "incident_title": result.get("incident_title"),
        "severity": result.get("severity"),
        "confidence": float(result.get("confidence", 0)),
        "needs_human_approval": bool(result.get("needs_human_approval", True)),
        "affected_services_json": json.dumps(result.get("affected_services", []), ensure_ascii=False),
        "recommended_actions_json": json.dumps(result.get("recommended_actions", []), ensure_ascii=False),
        "full_result_json": json.dumps(result, ensure_ascii=False),
    }

    errors = bq.insert_rows_json(table_id, [row])

    if errors:
        raise RuntimeError(f"Errores insertando en BigQuery: {errors}")


def main() -> None:
    events = sample_operational_events()
    prompt = build_prompt(events)

    result = analyze_with_llm(prompt)

    print("\n=== Diagnóstico generado ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    table_id = ensure_bigquery_table()
    write_result_to_bigquery(result, table_id)

    print(f"\nResultado guardado en BigQuery: {table_id}")


if __name__ == "__main__":
    main()