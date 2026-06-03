import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from google.api_core.exceptions import NotFound
from google.cloud import bigquery


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno obligatoria: {name}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def ensure_dataset(client: bigquery.Client, project_id: str, dataset_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")

    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset_ref.location = os.getenv("AIOPS_BQ_LOCATION", "EU")
        client.create_dataset(dataset_ref)
        print(f"Dataset creado: {project_id}.{dataset_id}")


def ensure_incidents_table(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> bigquery.Table:
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    schema = [
        bigquery.SchemaField("incident_id", "STRING"),
        bigquery.SchemaField("created_ts", "TIMESTAMP"),
        bigquery.SchemaField("updated_ts", "TIMESTAMP"),
        bigquery.SchemaField("closed_ts", "TIMESTAMP"),
        bigquery.SchemaField("service_name", "STRING"),
        bigquery.SchemaField("alert_type", "STRING"),
        bigquery.SchemaField("severity", "STRING"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("owner_team", "STRING"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("summary", "STRING"),
        bigquery.SchemaField("runbook", "STRING"),
        bigquery.SchemaField("recommended_action", "STRING"),
        bigquery.SchemaField("evidence_json", "STRING"),
        bigquery.SchemaField("source_alert_count", "INTEGER"),
    ]

    try:
        return client.get_table(table_ref)
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        print(f"Tabla de incidentes creada: {table_ref}")
        return table


def ensure_chatops_table(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> bigquery.Table:
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    schema = [
        bigquery.SchemaField("notification_id", "STRING"),
        bigquery.SchemaField("created_ts", "TIMESTAMP"),
        bigquery.SchemaField("incident_id", "STRING"),
        bigquery.SchemaField("service_name", "STRING"),
        bigquery.SchemaField("channel", "STRING"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("dry_run", "BOOLEAN"),
        bigquery.SchemaField("response_code", "INTEGER"),
        bigquery.SchemaField("error_message", "STRING"),
        bigquery.SchemaField("message_text", "STRING"),
    ]

    try:
        return client.get_table(table_ref)
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        print(f"Tabla ChatOps creada: {table_ref}")
        return table


def seed_demo_incident_if_empty(
    client: bigquery.Client,
    table: bigquery.Table,
    project_id: str,
    dataset_id: str,
    incidents_table: str,
) -> None:
    query = f"""
    SELECT COUNT(*) AS total
    FROM `{project_id}.{dataset_id}.{incidents_table}`
    WHERE status IN ('OPEN', 'INVESTIGATING', 'MITIGATING')
    """

    total = list(client.query(query).result())[0]["total"]

    if total > 0:
        return

    demo_incident = {
        "incident_id": "INC-DEMO-CHATOPS",
        "created_ts": utc_now(),
        "updated_ts": utc_now(),
        "closed_ts": None,
        "service_name": "checkout-api",
        "alert_type": "LATENCY_RISK",
        "severity": "HIGH",
        "status": "INVESTIGATING",
        "owner_team": "sre-payments",
        "title": "HIGH: LATENCY_RISK en checkout-api",
        "summary": (
            "La latencia p95 muestra una tendencia de degradación y podría "
            "superar el umbral operativo en los próximos 30 minutos."
        ),
        "runbook": "runbook-latency-degradation",
        "recommended_action": (
            "Revisar cambios recientes, dependencias externas y preparar mitigación."
        ),
        "evidence_json": json.dumps(
            {
                "metric_name": "latency_p95_ms",
                "current_value": 720,
                "predicted_value": 940,
                "threshold": 900,
                "confidence": 0.88,
            },
            ensure_ascii=False,
        ),
        "source_alert_count": 1,
    }

    errors = client.insert_rows_json(table, [demo_incident])
    if errors:
        raise RuntimeError(f"No se pudo insertar incidente demo: {errors}")

    print("No había incidentes abiertos. Se ha creado un incidente demo.")


def fetch_open_incidents(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    incidents_table: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    query = f"""
    SELECT
      incident_id,
      created_ts,
      updated_ts,
      service_name,
      alert_type,
      severity,
      status,
      owner_team,
      title,
      summary,
      runbook,
      recommended_action,
      evidence_json,
      source_alert_count
    FROM `{project_id}.{dataset_id}.{incidents_table}`
    WHERE status IN ('OPEN', 'INVESTIGATING', 'MITIGATING')
    ORDER BY
      CASE severity
        WHEN 'CRITICAL' THEN 4
        WHEN 'HIGH' THEN 3
        WHEN 'WARNING' THEN 2
        ELSE 1
      END DESC,
      updated_ts DESC
    LIMIT @max_results
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("max_results", "INT64", max_results)
        ]
    )

    rows = client.query(query, job_config=job_config).result()
    return [dict(row) for row in rows]


def already_notified_recently(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    chatops_table: str,
    incident_id: str,
    minutes: int = 60,
) -> bool:
    query = f"""
    SELECT COUNT(*) AS total
    FROM `{project_id}.{dataset_id}.{chatops_table}`
    WHERE incident_id = @incident_id
      AND created_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @minutes MINUTE)
      AND status IN ('SENT', 'DRY_RUN')
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("incident_id", "STRING", incident_id),
            bigquery.ScalarQueryParameter("minutes", "INT64", minutes),
        ]
    )

    total = list(client.query(query, job_config=job_config).result())[0]["total"]
    return total > 0


def severity_icon(severity: str) -> str:
    severity = str(severity).upper()
    if severity == "CRITICAL":
        return "🚨"
    if severity == "HIGH":
        return "⚠️"
    if severity == "WARNING":
        return "🟠"
    return "ℹ️"


def build_message(incident: Dict[str, Any]) -> str:
    icon = severity_icon(str(incident.get("severity", "")))

    evidence_text = ""
    raw_evidence = incident.get("evidence_json")

    if raw_evidence:
        try:
            evidence = json.loads(raw_evidence)
            compact = {
                key: evidence.get(key)
                for key in [
                    "metric_name",
                    "current_value",
                    "predicted_value",
                    "threshold",
                    "confidence",
                ]
                if key in evidence
            }
            if compact:
                evidence_text = "\n\nEvidencia:\n" + json.dumps(
                    compact,
                    ensure_ascii=False,
                    indent=2,
                )
        except json.JSONDecodeError:
            evidence_text = "\n\nEvidencia: no se pudo interpretar evidence_json."

    message = f"""
{icon} Incidente {incident.get("severity")}: {incident.get("alert_type")} en {incident.get("service_name")}

Incidente: {incident.get("incident_id")}
Estado: {incident.get("status")}
Owner: {incident.get("owner_team")}
Runbook: {incident.get("runbook")}

Resumen:
{incident.get("summary")}

Acción recomendada:
{incident.get("recommended_action")}

Comandos sugeridos:
ack {incident.get("incident_id")}
escalate {incident.get("incident_id")}
block-deploys {incident.get("service_name")}
{evidence_text}
""".strip()

    return message


def send_to_google_chat(webhook_url: str, message_text: str) -> Tuple[int, str]:
    response = requests.post(
        webhook_url,
        json={"text": message_text},
        timeout=10,
    )

    if response.status_code >= 300:
        return response.status_code, response.text

    return response.status_code, ""


def record_notification(
    client: bigquery.Client,
    table: bigquery.Table,
    incident_id: str,
    service_name: str,
    channel: str,
    status: str,
    dry_run: bool,
    message_text: str,
    response_code: Optional[int] = None,
    error_message: str = "",
) -> None:
    row = {
        "notification_id": f"NOTIF-{uuid.uuid4().hex[:12].upper()}",
        "created_ts": utc_now(),
        "incident_id": incident_id,
        "service_name": service_name,
        "channel": channel,
        "status": status,
        "dry_run": dry_run,
        "response_code": response_code,
        "error_message": error_message,
        "message_text": message_text,
    }

    errors = client.insert_rows_json(table, [row])
    if errors:
        raise RuntimeError(f"Error registrando notificación ChatOps: {errors}")


def main() -> int:
    try:
        project_id = require_env("PROJECT_ID")
        dataset_id = os.getenv("AIOPS_DATASET", "aiops_lab")
        incidents_table_name = os.getenv("AIOPS_INCIDENTS_TABLE", "incidents")
        chatops_table_name = os.getenv("AIOPS_CHATOPS_TABLE", "chatops_notifications")

        webhook_url = os.getenv("CHATOP+S_WEBHOOK_URL", "").strip()
        dry_run = env_bool("CHATOPS_DRY_RUN", default=True)
        channel = os.getenv("CHATOPS_CHANNEL", "google-chat-aiops")

        client = bigquery.Client(project=project_id)

        ensure_dataset(client, project_id, dataset_id)

        incidents_table = ensure_incidents_table(
            client,
            project_id,
            dataset_id,
            incidents_table_name,
        )

        chatops_table = ensure_chatops_table(
            client,
            project_id,
            dataset_id,
            chatops_table_name,
        )

        seed_demo_incident_if_empty(
            client,
            incidents_table,
            project_id,
            dataset_id,
            incidents_table_name,
        )

        incidents = fetch_open_incidents(
            client,
            project_id,
            dataset_id,
            incidents_table_name,
            max_results=10,
        )

        if not incidents:
            print("No hay incidentes abiertos.")
            return 0

        sent = 0
        skipped = 0
        failed = 0

        for incident in incidents:
            incident_id = str(incident["incident_id"])
            service_name = str(incident["service_name"])

            if already_notified_recently(
                client,
                project_id,
                dataset_id,
                chatops_table_name,
                incident_id,
                minutes=60,
            ):
                print(f"Saltado por deduplicación temporal: {incident_id}")
                skipped += 1
                continue

            message = build_message(incident)

            print("\n--- Mensaje ChatOps ---")
            print(message)

            if dry_run or not webhook_url:
                record_notification(
                    client=client,
                    table=chatops_table,
                    incident_id=incident_id,
                    service_name=service_name,
                    channel=channel,
                    status="DRY_RUN",
                    dry_run=True,
                    message_text=message,
                    response_code=None,
                    error_message="No se envió por dry-run o falta CHATOPS_WEBHOOK_URL.",
                )
                sent += 1
                continue

            response_code, error_message = send_to_google_chat(webhook_url, message)

            if response_code >= 300:
                status = "FAILED"
                failed += 1
            else:
                status = "SENT"
                sent += 1

            record_notification(
                client=client,
                table=chatops_table,
                incident_id=incident_id,
                service_name=service_name,
                channel=channel,
                status=status,
                dry_run=False,
                message_text=message,
                response_code=response_code,
                error_message=error_message,
            )

        print(
            f"\nResultado ChatOps: enviados_o_dry_run={sent}, "
            f"saltados={skipped}, fallidos={failed}"
        )

        return 1 if failed else 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())