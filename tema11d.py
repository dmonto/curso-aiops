import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


SENSITIVE_METHOD_KEYWORDS = {
    "setiampolicy": 40,
    "createiam": 25,
    "createserviceaccountkey": 50,
    "deleteserviceaccountkey": 30,
    "generateaccesstoken": 35,
    "signblob": 30,
    "signjwt": 30,
    "updatefunction": 20,
    "createexecution": 15,
    "delete": 20,
    "patch": 10,
    "update": 10,
}

SENSITIVE_RESOURCE_TOKENS = {
    "prod": 20,
    "production": 20,
    "raw": 15,
    "secret": 25,
    "secrets": 25,
    "iam": 20,
    "security": 20,
    "confidential": 20,
}

SERVICE_DOMAINS = {
    "bigquery.googleapis.com": "BIGQUERY",
    "storage.googleapis.com": "STORAGE",
    "aiplatform.googleapis.com": "VERTEX_AI",
    "pubsub.googleapis.com": "PUBSUB",
    "cloudfunctions.googleapis.com": "CLOUD_FUNCTIONS",
    "workflows.googleapis.com": "WORKFLOWS",
    "iam.googleapis.com": "IAM",
    "cloudresourcemanager.googleapis.com": "RESOURCE_MANAGER",
    "logging.googleapis.com": "LOGGING",
}


@dataclass
class AuditEvent:
    timestamp: str
    principal: str
    principal_type: str
    service_name: str
    service_domain: str
    method_name: str
    resource_name: str
    status_code: int
    status_message: str
    caller_ip: str
    user_agent: str
    permissions: str
    denied_permissions: str
    log_name: str
    risk_score: int
    risk_level: str
    reasons: str


def find_gcloud() -> str:
    gcloud = shutil.which("gcloud")
    if not gcloud:
        raise RuntimeError("No encuentro gcloud en PATH. Abre una terminal donde Google Cloud CLI funcione.")
    return gcloud


def run_gcloud(args: List[str]) -> Tuple[int, str, str]:
    completed = subprocess.run(
        [find_gcloud()] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"Falta variable de entorno: {name}")
    return value


def classify_principal(principal: str) -> str:
    if not principal:
        return "UNKNOWN"
    if principal.startswith("service-") and "gcp-sa-" in principal:
        return "SERVICE_AGENT"
    if principal.endswith(".gserviceaccount.com"):
        return "SERVICE_ACCOUNT"
    if "@" in principal:
        return "USER"
    if principal.startswith("principal://") or principal.startswith("principalSet://"):
        return "FEDERATED"
    return "OTHER"


def load_audit_logs(project_id: str, hours: int, limit: int) -> List[Dict[str, Any]]:
    log_filter = 'protoPayload.@type="type.googleapis.com/google.cloud.audit.AuditLog"'

    code, out, err = run_gcloud(
        [
            "logging",
            "read",
            log_filter,
            f"--project={project_id}",
            f"--freshness={hours}h",
            f"--limit={limit}",
            "--format=json",
        ]
    )

    if code != 0:
        raise RuntimeError(
            "No he podido leer Cloud Audit Logs.\n"
            f"Proyecto: {project_id}\n"
            f"Error: {err or out}\n\n"
            "Comprueba que tienes permisos para leer logs del proyecto."
        )

    if not out:
        return []

    return json.loads(out)


def extract_permissions(proto: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    permissions = []
    denied = []

    for item in proto.get("authorizationInfo", []) or []:
        permission = item.get("permission", "")
        granted = item.get("granted", None)

        if permission:
            permissions.append(permission)

        if permission and granted is False:
            denied.append(permission)

    return sorted(set(permissions)), sorted(set(denied))


def calculate_risk(
    principal: str,
    service_name: str,
    method_name: str,
    resource_name: str,
    status_code: int,
    denied_permissions: List[str],
) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    lower_method = (method_name or "").lower()
    lower_resource = (resource_name or "").lower()

    if status_code == 7:
        score += 25
        reasons.append("PERMISSION_DENIED")

    for keyword, points in SENSITIVE_METHOD_KEYWORDS.items():
        if keyword in lower_method:
            score += points
            reasons.append(f"método sensible: {keyword}")

    for token, points in SENSITIVE_RESOURCE_TOKENS.items():
        if token in lower_resource:
            score += points
            reasons.append(f"recurso sensible: {token}")

    if denied_permissions:
        score += min(20, len(denied_permissions) * 5)
        reasons.append("permisos denegados en authorizationInfo")

    principal_type = classify_principal(principal)

    if principal_type == "SERVICE_ACCOUNT" and ("iam" in lower_method or "setiampolicy" in lower_method):
        score += 25
        reasons.append("service account ejecutando operación IAM")

    if service_name == "iam.googleapis.com":
        score += 10
        reasons.append("servicio IAM")

    if service_name == "aiplatform.googleapis.com" and "delete" in lower_method:
        score += 20
        reasons.append("eliminación en Vertex AI")

    return min(score, 100), reasons


def risk_level(score: int) -> str:
    if score >= 75:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    if score >= 15:
        return "LOW"
    return "INFO"


def normalize_entry(entry: Dict[str, Any]) -> AuditEvent:
    proto = entry.get("protoPayload", {}) or {}

    principal = (
        proto.get("authenticationInfo", {}) or {}
    ).get("principalEmail", "")

    request_metadata = proto.get("requestMetadata", {}) or {}
    status = proto.get("status", {}) or {}

    service_name = proto.get("serviceName", "")
    method_name = proto.get("methodName", "")
    resource_name = proto.get("resourceName", "")

    permissions, denied_permissions = extract_permissions(proto)

    status_code = int(status.get("code", 0) or 0)
    status_message = status.get("message", "")

    score, reasons = calculate_risk(
        principal=principal,
        service_name=service_name,
        method_name=method_name,
        resource_name=resource_name,
        status_code=status_code,
        denied_permissions=denied_permissions,
    )

    return AuditEvent(
        timestamp=entry.get("timestamp", ""),
        principal=principal or "UNKNOWN",
        principal_type=classify_principal(principal),
        service_name=service_name,
        service_domain=SERVICE_DOMAINS.get(service_name, "OTHER"),
        method_name=method_name,
        resource_name=resource_name,
        status_code=status_code,
        status_message=status_message,
        caller_ip=request_metadata.get("callerIp", ""),
        user_agent=request_metadata.get("callerSuppliedUserAgent", ""),
        permissions=";".join(permissions),
        denied_permissions=";".join(denied_permissions),
        log_name=entry.get("logName", ""),
        risk_score=score,
        risk_level=risk_level(score),
        reasons="; ".join(reasons),
    )


def write_reports(events: List[AuditEvent], project_id: str) -> None:
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"access-audit-{project_id}-{timestamp}.csv"
    json_path = out_dir / f"access-audit-{project_id}-{timestamp}.json"
    md_path = out_dir / f"access-audit-{project_id}-{timestamp}.md"

    rows = [asdict(e) for e in events]

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    by_principal = Counter(e.principal for e in events)
    by_service = Counter(e.service_domain for e in events)
    by_risk = Counter(e.risk_level for e in events)

    high_events = [e for e in events if e.risk_level in {"HIGH", "MEDIUM"}]
    denied_events = [e for e in events if e.status_code == 7]

    with md_path.open("w", encoding="utf-8") as f:
        f.write("### Informe de auditoría de accesos\n\n")
        f.write(f"Proyecto: `{project_id}`\n\n")
        f.write(f"Fecha de generación: `{datetime.now(timezone.utc).isoformat()}`\n\n")
        f.write(f"Eventos analizados: `{len(events)}`\n\n")

        f.write("#### Resumen por riesgo\n\n")
        for level in ["HIGH", "MEDIUM", "LOW", "INFO"]:
            f.write(f"- {level}: {by_risk.get(level, 0)}\n")

        f.write("\n#### Resumen por servicio\n\n")
        for service, count in by_service.most_common():
            f.write(f"- {service}: {count}\n")

        f.write("\n#### Principales identidades\n\n")
        for principal, count in by_principal.most_common(10):
            f.write(f"- `{principal}`: {count}\n")

        f.write("\n#### Eventos denegados\n\n")
        if not denied_events:
            f.write("No se han encontrado eventos `PERMISSION_DENIED`.\n")
        else:
            for e in denied_events[:20]:
                f.write(
                    f"- `{e.timestamp}` | `{e.principal}` | `{e.service_domain}` | "
                    f"`{e.method_name}` | `{e.resource_name}`\n"
                )

        f.write("\n#### Eventos de mayor riesgo\n\n")
        if not high_events:
            f.write("No se han encontrado eventos de riesgo medio o alto con las reglas actuales.\n")
        else:
            for e in sorted(high_events, key=lambda x: x.risk_score, reverse=True)[:25]:
                f.write(f"##### {e.risk_level} - score {e.risk_score}\n\n")
                f.write(f"- Timestamp: `{e.timestamp}`\n")
                f.write(f"- Principal: `{e.principal}`\n")
                f.write(f"- Tipo principal: `{e.principal_type}`\n")
                f.write(f"- Servicio: `{e.service_domain}`\n")
                f.write(f"- Método: `{e.method_name}`\n")
                f.write(f"- Recurso: `{e.resource_name}`\n")
                f.write(f"- IP: `{e.caller_ip}`\n")
                f.write(f"- Permisos denegados: `{e.denied_permissions}`\n")
                f.write(f"- Motivos: {e.reasons}\n\n")

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


def print_summary(events: List[AuditEvent]) -> None:
    print("\nAuditoría de accesos\n")
    print(f"Eventos analizados: {len(events)}")

    if not events:
        print("No se han encontrado eventos en la ventana indicada.")
        return

    by_risk = Counter(e.risk_level for e in events)
    print("\nRiesgo:")
    for level in ["HIGH", "MEDIUM", "LOW", "INFO"]:
        print(f"  {level}: {by_risk.get(level, 0)}")

    print("\nEventos destacados:")
    for e in sorted(events, key=lambda x: x.risk_score, reverse=True)[:10]:
        print(f"- [{e.risk_level}] score={e.risk_score} | {e.principal} | {e.service_domain}")
        print(f"  Método: {e.method_name}")
        print(f"  Recurso: {e.resource_name}")
        if e.reasons:
            print(f"  Motivos: {e.reasons}")


def main() -> None:
    try:
        project_id = get_env("PROJECT_ID")
        hours = int(os.getenv("AUDIT_LOOKBACK_HOURS", "24"))
        limit = int(os.getenv("AUDIT_LIMIT", "500"))

        raw_entries = load_audit_logs(project_id, hours, limit)
        events = [normalize_entry(entry) for entry in raw_entries]

        print_summary(events)
        write_reports(events, project_id)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()