import argparse
import datetime as dt
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

from google import genai
from google.genai.types import HttpOptions
from google.cloud import bigquery
from google.cloud import logging_v2


PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("GGENAI_LOCATION", "global")
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

BQ_DATASET = os.getenv("BQ_DATASET", "aiops_lab")
BQ_TABLE = os.getenv("BQ_LOG_SUMMARY_TABLE", "log_summaries")

if not PROJECT_ID:
    raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT")

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


SUMMARY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "severity": {"type": "STRING"},
        "affected_services": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "observed_patterns": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "message_pattern": {"type": "STRING"},
                    "count": {"type": "INTEGER"},
                    "first_seen": {"type": "STRING"},
                    "last_seen": {"type": "STRING"},
                    "evidence": {"type": "STRING"},
                },
                "required": [
                    "message_pattern",
                    "count",
                    "first_seen",
                    "last_seen",
                    "evidence",
                ],
            },
        },
        "probable_cause": {"type": "STRING"},
        "recommended_next_steps": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "needs_human_review": {"type": "BOOLEAN"},
        "confidence": {"type": "NUMBER"},
        "missing_information": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "summary",
        "severity",
        "affected_services",
        "observed_patterns",
        "probable_cause",
        "recommended_next_steps",
        "needs_human_review",
        "confidence",
        "missing_information",
    ],
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def normalize_message(message: str) -> str:
    """
    Reduce variabilidad para agrupar errores similares.
    Sustituye IDs, trazas, números largos y timestamps por placeholders.
    """
    text = message.strip()

    text = re.sub(r"\btrace[_-]?id[=:][A-Za-z0-9._\-]+", "trace_id=<TRACE_ID>", text, flags=re.I)
    text = re.sub(r"\brequest[_-]?id[=:][A-Za-z0-9._\-]+", "request_id=<REQUEST_ID>", text, flags=re.I)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}T[^\s]+", "<TIMESTAMP>", text)
    text = re.sub(r"\b\d{6,}\b", "<NUMBER>", text)
    text = re.sub(r"\blatency_ms=\d+\b", "latency_ms=<NUMBER>", text)
    text = re.sub(r"\bstatus=\d{3}\b", "status=<STATUS>", text)

    return text[:500]


def redact_sensitive_text(text: str) -> str:
    """
    Redacción mínima antes de enviar contenido al modelo.
    En producción, esta lógica debería complementarse con controles de DLP y clasificación.
    """
    redacted = text
    redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED_TOKEN]", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key=)[A-Za-z0-9._\-]+", r"\1[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"(?i)(password=)[^&\s]+", r"\1[REDACTED_PASSWORD]", redacted)
    redacted = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]", redacted)
    return redacted


def payload_to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        if "message" in payload:
            return str(payload["message"])
        return json.dumps(payload, ensure_ascii=False)

    return str(payload)


def fetch_logs(minutes: int, limit: int) -> List[Dict[str, Any]]:
    client = logging_v2.Client(project=PROJECT_ID)

    start_time = utc_now() - dt.timedelta(minutes=minutes)
    start_iso = start_time.isoformat().replace("+00:00", "Z")

    logging_filter = f'''
timestamp >= "{start_iso}"
severity >= ERROR
'''.strip()

    entries = client.list_entries(
        filter_=logging_filter,
        order_by=logging_v2.DESCENDING,
        max_results=limit,
    )

    rows: List[Dict[str, Any]] = []

    for entry in entries:
        message = payload_to_text(entry.payload)
        timestamp = entry.timestamp.isoformat() if entry.timestamp else None

        rows.append(
            {
                "timestamp": timestamp,
                "severity": entry.severity,
                "log_name": entry.log_name,
                "resource_type": entry.resource.type if entry.resource else None,
                "message": redact_sensitive_text(message),
            }
        )

    return rows


def fallback_sample_logs() -> List[Dict[str, Any]]:
    """
    Permite ejecutar el laboratorio aunque el proyecto todavía no tenga logs reales.
    """
    now = utc_now()

    sample_messages = [
        "POST /checkout failed status=500 latency_ms=2380 trace_id=abc123",
        "POST /checkout failed status=500 latency_ms=2510 trace_id=abc124",
        "Database connection error: too many connections for cloud-sql-orders",
        "Database connection error: too many connections for cloud-sql-orders",
        "Retry storm detected for dependency cloud-sql-orders",
        "Pub/Sub publish failed after retries topic=orders-events",
    ]

    rows = []
    for idx, message in enumerate(sample_messages):
        rows.append(
            {
                "timestamp": (now - dt.timedelta(minutes=idx * 2)).isoformat(),
                "severity": "ERROR",
                "log_name": "sample-log",
                "resource_type": "sample",
                "message": message,
            }
        )

    return rows


def deduplicate_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for row in logs:
        pattern = normalize_message(row["message"])

        if pattern not in grouped:
            grouped[pattern] = {
                "message_pattern": pattern,
                "count": 0,
                "first_seen": row["timestamp"],
                "last_seen": row["timestamp"],
                "max_severity": row["severity"],
                "examples": [],
            }

        item = grouped[pattern]
        item["count"] += 1

        timestamps = [ts for ts in [item["first_seen"], item["last_seen"], row["timestamp"]] if ts]
        item["first_seen"] = min(timestamps)
        item["last_seen"] = max(timestamps)

        if len(item["examples"]) < 3:
            item["examples"].append(row["message"])

    return sorted(grouped.values(), key=lambda x: x["count"], reverse=True)


def build_summary_prompt( minutes: int, patterns: List[Dict[str, Any]]) -> str:
    compact_patterns = patterns[:20]

    return f"""
Actúa como asistente SRE especializado en análisis de logs.

Ventana temporal: últimos {minutes} minutos

Has recibido patrones de logs ya filtrados, redactados y deduplicados.
Genera un resumen técnico operativo.

Reglas:
- Usa solo los patrones proporcionados.
- No inventes métricas, recursos ni causas.
- Distingue evidencias observadas de hipótesis.
- Si falta información para confirmar causa raíz, indícalo.
- No propongas acciones destructivas.
- Las acciones de rollback, escalado o cambio de configuración requieren revisión humana.
- confidence debe estar entre 0 y 1.
- Devuelve únicamente JSON válido.

Patrones de logs:
{json.dumps(compact_patterns, indent=2, ensure_ascii=False)}
""".strip()


def summarize_with_gemini(prompt: str) -> Dict[str, Any]:
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
            "response_schema": SUMMARY_SCHEMA,
        },
    )

    if not response.text:
        raise RuntimeError("Gemini no devolvió contenido.")

    return json.loads(response.text)


def ensure_bigquery_table() -> str:
    bq = bigquery.Client(project=PROJECT_ID)

    dataset_id = f"{PROJECT_ID}.{BQ_DATASET}"
    table_id = f"{dataset_id}.{BQ_TABLE}"

    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "EU"
    bq.create_dataset(dataset, exists_ok=True)

    schema = [
        bigquery.SchemaField("summary_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("window_minutes", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("model_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("severity", "STRING"),
        bigquery.SchemaField("confidence", "FLOAT"),
        bigquery.SchemaField("needs_human_review", "BOOLEAN"),
        bigquery.SchemaField("summary", "STRING"),
        bigquery.SchemaField("affected_services_json", "STRING"),
        bigquery.SchemaField("observed_patterns_json", "STRING"),
        bigquery.SchemaField("full_result_json", "STRING"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    bq.create_table(table, exists_ok=True)

    return table_id


def write_summary_to_bigquery(
    minutes: int,
    result: Dict[str, Any],
    table_id: str,
) -> None:
    bq = bigquery.Client(project=PROJECT_ID)

    raw_id = f"genai-{minutes}-{utc_now().isoformat()}-{result.get('summary', '')}"
    summary_id = hashlib.sha256(raw_id.encode()).hexdigest()[:20]

    row = {
        "summary_id": summary_id,
        "created_at": utc_now().isoformat(),
        "window_minutes": minutes,
        "model_id": MODEL_ID,
        "severity": result.get("severity"),
        "confidence": float(result.get("confidence", 0)),
        "needs_human_review": bool(result.get("needs_human_review", True)),
        "summary": result.get("summary"),
        "affected_services_json": json.dumps(result.get("affected_services", []), ensure_ascii=False),
        "observed_patterns_json": json.dumps(result.get("observed_patterns", []), ensure_ascii=False),
        "full_result_json": json.dumps(result, ensure_ascii=False),
    }

    errors = bq.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Errores insertando en BigQuery: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumen automático de logs con Gemini y BigQuery")
    parser.add_argument("--minutes", type=int, default=60, help="Ventana temporal en minutos")
    parser.add_argument("--limit", type=int, default=100, help="Número máximo de logs a recuperar")
    parser.add_argument("--use-sample-if-empty", action="store_true", help="Usar logs simulados si no hay logs reales")
    args = parser.parse_args()

    logs = fetch_logs( args.minutes, args.limit)

    if not logs and args.use_sample_if_empty:
        print("No se encontraron logs reales. Usando logs simulados para el laboratorio.")
        logs = fallback_sample_logs()

    if not logs:
        print("No se encontraron logs para los filtros indicados.")
        return

    patterns = deduplicate_logs(logs)
    prompt = build_summary_prompt(args.minutes, patterns)
    result = summarize_with_gemini(prompt)

    print("\n=== Resumen automático ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    table_id = ensure_bigquery_table()
    write_summary_to_bigquery(args.minutes, result, table_id)

    print(f"\nResumen guardado en BigQuery: {table_id}")


if __name__ == "__main__":
    main()