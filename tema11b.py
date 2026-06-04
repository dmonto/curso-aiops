import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

try:
    from google.cloud import bigquery
    from google.api_core.exceptions import GoogleAPIError
except ImportError:
    bigquery = None
    GoogleAPIError = Exception


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]+")
PASSWORD_RE = re.compile(r"(?i)\b(password|passwd|pwd)\s*=\s*[^&\s,;]+")
API_KEY_RE = re.compile(r"(?i)\b(api[_-]?key|x-api-key|client_secret|secret)\s*[:=]\s*[^&\s,;]+")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")


@dataclass
class ProtectedEvent:
    event_id: str
    event_ts: str
    service_name: str
    environment: str
    severity: str
    event_type: str
    user_hash: str
    source_ip_masked: str
    original_length: int
    protected_message: str
    findings: str
    risk_score: int
    protection_action: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: str, salt: str, length: int = 16) -> str:
    if not value:
        return ""
    digest = hashlib.sha256((salt + "|" + value.lower().strip()).encode("utf-8")).hexdigest()
    return digest[:length]


def mask_email(email: str) -> str:
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "***"
    else:
        masked_local = local[0] + "***" + local[-1]
    return f"{masked_local}@{domain}"


def mask_ip(ip: str) -> str:
    parts = ip.split(".")
    if len(parts) != 4:
        return "[IP_MASKED]"
    return ".".join(parts[:2] + ["x", "x"])


def detect_findings(text: str) -> Dict[str, List[str]]:
    findings = {
        "email": EMAIL_RE.findall(text or ""),
        "ip": IP_RE.findall(text or ""),
        "bearer_token": BEARER_RE.findall(text or ""),
        "password": PASSWORD_RE.findall(text or ""),
        "api_key_or_secret": API_KEY_RE.findall(text or ""),
        "jwt": JWT_RE.findall(text or ""),
    }
    return {k: v for k, v in findings.items() if v}


def protect_text(text: str) -> Tuple[str, Dict[str, List[str]], int, str]:
    findings = detect_findings(text)
    protected = text or ""

    # Redactar secretos primero
    protected = BEARER_RE.sub("Bearer [REDACTED]", protected)
    protected = JWT_RE.sub("[JWT_REDACTED]", protected)
    protected = PASSWORD_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", protected)
    protected = API_KEY_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", protected)

    # Enmascarar emails e IPs
    protected = EMAIL_RE.sub(lambda m: mask_email(m.group(0)), protected)
    protected = IP_RE.sub(lambda m: mask_ip(m.group(0)), protected)

    risk_score = 0
    if "email" in findings:
        risk_score += 2
    if "ip" in findings:
        risk_score += 1
    if "bearer_token" in findings:
        risk_score += 5
    if "password" in findings:
        risk_score += 5
    if "api_key_or_secret" in findings:
        risk_score += 5
    if "jwt" in findings:
        risk_score += 5

    if risk_score >= 8:
        action = "REDACT_AND_ALERT"
    elif risk_score >= 3:
        action = "MASK_AND_REVIEW"
    elif risk_score > 0:
        action = "MASK"
    else:
        action = "NONE"

    return protected, findings, risk_score, action


def extract_first_email(text: str) -> str:
    match = EMAIL_RE.search(text or "")
    return match.group(0) if match else ""


def extract_first_ip(text: str) -> str:
    match = IP_RE.search(text or "")
    return match.group(0) if match else ""


def sample_events() -> List[Dict[str, str]]:
    return [
        {
            "event_id": "evt-001",
            "service_name": "checkout",
            "environment": "prod",
            "severity": "ERROR",
            "event_type": "http_error",
            "message": "Payment timeout for user ana.garcia@empresa.com from 10.20.30.40 endpoint=/pay",
        },
        {
            "event_id": "evt-002",
            "service_name": "identity",
            "environment": "prod",
            "severity": "WARNING",
            "event_type": "login_failed",
            "message": "login failed user=diego@empresa.com ip=81.44.10.22 password=Prueba123!",
        },
        {
            "event_id": "evt-003",
            "service_name": "orders",
            "environment": "dev",
            "severity": "INFO",
            "event_type": "debug_trace",
            "message": "Calling downstream API with Authorization: Bearer eyJhbGciOiJIUzI1Ni.fake.token",
        },
        {
            "event_id": "evt-004",
            "service_name": "inventory",
            "environment": "prod",
            "severity": "INFO",
            "event_type": "business_event",
            "message": "stock refresh completed sku_count=458 duration_ms=820",
        },
        {
            "event_id": "evt-005",
            "service_name": "support",
            "environment": "test",
            "severity": "ERROR",
            "event_type": "ticket_sync",
            "message": "ticket sync failed for customer admin@cliente.com x-api-key: ABCD-1234-SECRET",
        },
    ]


def protect_event(raw: Dict[str, str], salt: str) -> ProtectedEvent:
    message = raw["message"]
    protected_message, findings, risk_score, action = protect_text(message)

    first_email = extract_first_email(message)
    first_ip = extract_first_ip(message)

    return ProtectedEvent(
        event_id=raw["event_id"],
        event_ts=now_iso(),
        service_name=raw["service_name"],
        environment=raw["environment"],
        severity=raw["severity"],
        event_type=raw["event_type"],
        user_hash=stable_hash(first_email, salt),
        source_ip_masked=mask_ip(first_ip) if first_ip else "",
        original_length=len(message),
        protected_message=protected_message,
        findings=json.dumps(findings, ensure_ascii=False),
        risk_score=risk_score,
        protection_action=action,
    )


def write_csv(events: List[ProtectedEvent], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def ensure_bigquery_table(project_id: str, dataset_id: str, table_id: str) -> None:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery no está instalado.")

    client = bigquery.Client(project=project_id)

    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset_ref.location = "EU"

    try:
        client.get_dataset(dataset_ref)
    except Exception:
        client.create_dataset(dataset_ref, exists_ok=True)

    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("event_ts", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("service_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("environment", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("severity", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_hash", "STRING"),
        bigquery.SchemaField("source_ip_masked", "STRING"),
        bigquery.SchemaField("original_length", "INTEGER"),
        bigquery.SchemaField("protected_message", "STRING"),
        bigquery.SchemaField("findings", "STRING"),
        bigquery.SchemaField("risk_score", "INTEGER"),
        bigquery.SchemaField("protection_action", "STRING"),
    ]

    table_ref = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
    client.create_table(table_ref, exists_ok=True)


def write_bigquery(project_id: str, dataset_id: str, table_id: str, events: List[ProtectedEvent]) -> None:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery no está instalado.")

    ensure_bigquery_table(project_id, dataset_id, table_id)

    client = bigquery.Client(project=project_id)
    table_full_name = f"{project_id}.{dataset_id}.{table_id}"
    rows = [asdict(e) for e in events]

    errors = client.insert_rows_json(table_full_name, rows)
    if errors:
        raise RuntimeError(f"Errores insertando en BigQuery: {errors}")


def print_summary(events: List[ProtectedEvent]) -> None:
    print("\nResumen de protección de datos operativos\n")

    total = len(events)
    risky = [e for e in events if e.risk_score >= 3]
    alerts = [e for e in events if e.protection_action == "REDACT_AND_ALERT"]

    print(f"Eventos procesados: {total}")
    print(f"Eventos con hallazgos relevantes: {len(risky)}")
    print(f"Eventos que requieren alerta: {len(alerts)}")

    print("\nDetalle:")
    for e in events:
        print(
            f"- {e.event_id} | {e.service_name} | {e.severity} | "
            f"risk_score={e.risk_score} | action={e.protection_action}"
        )

    print("\nEjemplo de mensaje protegido:")
    print(events[0].protected_message)


def main() -> None:
    load_dotenv()

    project_id = os.getenv("PROJECT_ID", "")
    dataset_id = os.getenv("BQ_DATASET", "aiops_security")
    table_id = os.getenv("BQ_TABLE", "protected_events")
    salt = os.getenv("OPERATIONAL_DATA_SALT", "")

    if not salt or salt == "cambia-este-valor-en-tu-entorno":
        print(
            "ERROR: Define OPERATIONAL_DATA_SALT con un valor propio. "
            "No uses el valor de ejemplo.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_events = sample_events()
    protected_events = [protect_event(e, salt) for e in raw_events]

    output_path = Path("reports") / "protected_operational_events.csv"
    write_csv(protected_events, output_path)

    print_summary(protected_events)
    print(f"\nCSV generado: {output_path}")

    write_to_bq = os.getenv("WRITE_TO_BIGQUERY", "false").lower() == "true"

    if write_to_bq:
        if not project_id:
            print("ERROR: PROJECT_ID es obligatorio para escribir en BigQuery.", file=sys.stderr)
            sys.exit(1)

        try:
            write_bigquery(project_id, dataset_id, table_id, protected_events)
            print(f"Eventos insertados en BigQuery: {project_id}.{dataset_id}.{table_id}")
        except GoogleAPIError as exc:
            print(f"ERROR BigQuery: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()