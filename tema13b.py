import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from google.api_core.exceptions import Forbidden, GoogleAPIError, NotFound
from google.cloud import bigquery


load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET", "aiops_governance")
BQ_TABLE = os.getenv("BQ_TABLE", "change_control_log")
LOCATION = os.getenv("LOCATION", "europe-west1")

if not PROJECT_ID:
    raise RuntimeError("Falta PROJECT_ID en el archivo .env")


REQUIRED_FIELDS = [
    "change_id",
    "title",
    "environment",
    "change_type",
    "requested_by",
    "owner_functional",
    "owner_technical",
    "reason",
    "affected_resources",
    "before",
    "after",
    "test_plan",
    "rollback_plan",
]


@dataclass
class ChangeControlRecord:
    event_ts: str
    change_id: str
    title: str
    environment: str
    change_type: str
    requested_by: str
    owner_functional: str
    owner_technical: str
    calculated_risk: str
    decision: str
    blockers: List[str]
    warnings: List[str]
    diff_summary: List[str]
    approvals: List[str]
    affected_resources_json: str
    evidence: List[str]
    policy_version: str = "change-control-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_change_request(path: str = "change_request.yaml") -> Dict[str, Any]:
    input_file = Path(path)

    if not input_file.exists():
        raise FileNotFoundError(f"No existe {path}")

    with input_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_required_fields(change: Dict[str, Any]) -> List[str]:
    blockers = []

    for field in REQUIRED_FIELDS:
        value = change.get(field)
        if value is None or value == "" or value == [] or value == {}:
            blockers.append(f"Falta campo obligatorio: {field}")

    return blockers


def calculate_risk(change: Dict[str, Any]) -> str:
    score = 0

    environment = change.get("environment", "").lower()
    change_type = change.get("change_type", "").lower()
    before = change.get("before", {}) or {}
    after = change.get("after", {}) or {}

    if environment == "prod":
        score += 3

    if change_type in ["model_deployment", "automation", "iam", "workflow"]:
        score += 2

    if before.get("model_version") != after.get("model_version"):
        score += 2

    if before.get("threshold") != after.get("threshold"):
        score += 2

    if before.get("action") != after.get("action"):
        score += 3

    if before.get("service_account_role") != after.get("service_account_role"):
        score += 3

    if after.get("action") in ["create_ticket", "automatic_remediation"]:
        score += 3

    if after.get("action") == "automatic_remediation":
        score += 4

    if score <= 2:
        return "bajo"
    if score <= 5:
        return "medio"
    if score <= 9:
        return "alto"
    return "critico"


def build_diff_summary(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    diff = []

    all_keys = sorted(set(before.keys()) | set(after.keys()))

    for key in all_keys:
        old_value = before.get(key, "<no definido>")
        new_value = after.get(key, "<no definido>")

        if old_value != new_value:
            diff.append(f"{key}: {old_value} -> {new_value}")

    return diff


def evaluate_change(change: Dict[str, Any], risk: str, diff: List[str]) -> Dict[str, List[str]]:
    blockers = []
    warnings = []

    environment = change.get("environment", "").lower()
    after = change.get("after", {}) or {}
    approvals = change.get("approvals", []) or []
    rollback_plan = change.get("rollback_plan", []) or []
    evidence = change.get("evidence", []) or []

    if risk in ["alto", "critico"] and len(rollback_plan) == 0:
        blockers.append("Cambios de riesgo alto/crítico requieren rollback_plan.")

    if environment == "prod" and len(evidence) == 0:
        blockers.append("Cambios en producción requieren evidencias.")

    if environment == "prod" and "owner_funcional" not in approvals:
        blockers.append("Cambios en producción requieren aprobación de owner_funcional.")

    if after.get("action") in ["create_ticket", "automatic_remediation"]:
        if "sre_operaciones" not in approvals:
            blockers.append("Cambios que crean tickets o remediación requieren aprobación de SRE.")

    if after.get("action") == "automatic_remediation":
        if "seguridad" not in approvals:
            blockers.append("Remediación automática requiere aprobación de seguridad.")
        if risk != "critico":
            warnings.append("La remediación automática normalmente debería clasificarse como crítica.")

    for item in diff:
        if "threshold" in item:
            warnings.append("Cambio de umbral: revisar impacto en volumen de alertas y tickets.")
        if "service_account_role" in item:
            blockers.append("Cambio de rol IAM detectado: requiere revisión explícita de seguridad/cloud admin.")
        if "model_version" in item:
            warnings.append("Cambio de versión de modelo: revisar métricas, baseline y rollback.")

    return {
        "blockers": blockers,
        "warnings": warnings,
    }


def table_id() -> str:
    return f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"


def ensure_table() -> None:
    client = bigquery.Client(project=PROJECT_ID)
    dataset_id = f"{PROJECT_ID}.{BQ_DATASET}"
    full_table_id = table_id()

    try:
        client.get_dataset(dataset_id)
    except NotFound:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = LOCATION
        client.create_dataset(dataset)
        print(f"Dataset creado: {dataset_id}")

    schema = [
        bigquery.SchemaField("event_ts", "TIMESTAMP"),
        bigquery.SchemaField("change_id", "STRING"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("environment", "STRING"),
        bigquery.SchemaField("change_type", "STRING"),
        bigquery.SchemaField("requested_by", "STRING"),
        bigquery.SchemaField("owner_functional", "STRING"),
        bigquery.SchemaField("owner_technical", "STRING"),
        bigquery.SchemaField("calculated_risk", "STRING"),
        bigquery.SchemaField("decision", "STRING"),
        bigquery.SchemaField("blockers", "STRING", mode="REPEATED"),
        bigquery.SchemaField("warnings", "STRING", mode="REPEATED"),
        bigquery.SchemaField("diff_summary", "STRING", mode="REPEATED"),
        bigquery.SchemaField("approvals", "STRING", mode="REPEATED"),
        bigquery.SchemaField("affected_resources_json", "STRING"),
        bigquery.SchemaField("evidence", "STRING", mode="REPEATED"),
        bigquery.SchemaField("policy_version", "STRING"),
    ]

    try:
        client.get_table(full_table_id)
    except NotFound:
        table = bigquery.Table(full_table_id, schema=schema)
        client.create_table(table)
        print(f"Tabla creada: {full_table_id}")


def persist_bigquery(record: ChangeControlRecord) -> None:
    ensure_table()
    client = bigquery.Client(project=PROJECT_ID)

    errors = client.insert_rows_json(table_id(), [asdict(record)])

    if errors:
        raise RuntimeError(f"Errores insertando en BigQuery: {errors}")

    print(f"Registro insertado en BigQuery: {table_id()}")


def persist_local(record: ChangeControlRecord) -> None:
    output_file = "change_control_log.jsonl"

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    print(f"Registro guardado localmente: {output_file}")


def print_result(record: ChangeControlRecord) -> None:
    print("\n=== Control de cambios AIOps ===")
    print(f"Cambio: {record.change_id} - {record.title}")
    print(f"Entorno: {record.environment}")
    print(f"Tipo: {record.change_type}")
    print(f"Riesgo calculado: {record.calculated_risk}")
    print(f"Decisión: {record.decision}")

    print("\nDiferencias detectadas:")
    if record.diff_summary:
        for item in record.diff_summary:
            print(f"- {item}")
    else:
        print("- No hay diferencias entre before y after.")

    if record.warnings:
        print("\nWarnings:")
        for warning in record.warnings:
            print(f"- {warning}")

    if record.blockers:
        print("\nBloqueos:")
        for blocker in record.blockers:
            print(f"- {blocker}")


def main() -> None:
    change = load_change_request()

    blockers = validate_required_fields(change)
    risk = calculate_risk(change)

    before = change.get("before", {}) or {}
    after = change.get("after", {}) or {}
    diff = build_diff_summary(before, after)

    evaluation = evaluate_change(change, risk, diff)
    blockers.extend(evaluation["blockers"])
    warnings = evaluation["warnings"]

    decision = "ready_for_approval" if not blockers else "blocked"

    record = ChangeControlRecord(
        event_ts=utc_now(),
        change_id=change.get("change_id", ""),
        title=change.get("title", ""),
        environment=change.get("environment", ""),
        change_type=change.get("change_type", ""),
        requested_by=change.get("requested_by", ""),
        owner_functional=change.get("owner_functional", ""),
        owner_technical=change.get("owner_technical", ""),
        calculated_risk=risk,
        decision=decision,
        blockers=blockers,
        warnings=warnings,
        diff_summary=diff,
        approvals=change.get("approvals", []) or [],
        affected_resources_json=json.dumps(change.get("affected_resources", []), ensure_ascii=False),
        evidence=change.get("evidence", []) or [],
    )

    print_result(record)

    try:
        persist_bigquery(record)
    except (Forbidden, GoogleAPIError, RuntimeError) as ex:
        print(f"No se pudo escribir en BigQuery: {ex}")
        print("Se usará registro local como fallback.")
        persist_local(record)


if __name__ == "__main__":
    main()