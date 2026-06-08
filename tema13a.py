import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from google.api_core.exceptions import Forbidden, GoogleAPIError
from google.cloud import storage


load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DOCS_BUCKET = os.getenv("DOCS_BUCKET")
DOCS_PREFIX = os.getenv("DOCS_PREFIX", "aiops-docs")

REQUIRED_PATHS = [
    "system.name",
    "system.environment",
    "system.criticality",
    "system.objective",
    "owners.functional",
    "owners.technical",
    "architecture.components",
    "data.dataset_uri",
    "data.label",
    "data.features",
    "model.id",
    "model.version",
    "model.metrics.precision",
    "model.metrics.recall",
    "operation.rollback",
    "governance.approval_status",
    "governance.next_review_date",
]


def get_nested(data: Dict[str, Any], path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_documentation(data: Dict[str, Any]) -> List[str]:
    findings = []

    for path in REQUIRED_PATHS:
        value = get_nested(data, path)
        if value is None or value == "" or value == []:
            findings.append(f"Falta campo obligatorio: {path}")

    criticality = str(get_nested(data, "system.criticality") or "").lower()
    approval_status = str(get_nested(data, "governance.approval_status") or "").lower()
    rollback = str(get_nested(data, "operation.rollback") or "")

    if criticality in ["alta", "critica", "crítica"]:
        approved_by = get_nested(data, "governance.approved_by") or []
        if len(approved_by) < 2:
            findings.append("Criticidad alta/crítica requiere al menos dos aprobadores documentados.")

    if approval_status in ["approved", "approved_for_review"] and len(rollback.strip()) < 20:
        findings.append("Una solución aprobada requiere un rollback suficientemente descrito.")

    features = get_nested(data, "data.features") or []
    if len(features) < 3:
        findings.append("El dataset tiene pocas features documentadas; revisa si la descripción es suficiente.")

    evidence = get_nested(data, "governance.evidence") or []
    if approval_status in ["approved", "approved_for_review"] and not evidence:
        findings.append("Una solución aprobada requiere evidencias documentadas.")

    return findings


def render_list(items: List[Any]) -> str:
    if not items:
        return "- No documentado"
    return "\n".join(f"- {item}" for item in items)


def render_components(components: List[Dict[str, str]]) -> str:
    if not components:
        return "| Componente | Propósito |\n|---|---|\n| No documentado | No documentado |"

    lines = ["| Componente | Propósito |", "|---|---|"]
    for component in components:
        lines.append(f"| {component.get('name', '')} | {component.get('purpose', '')} |")
    return "\n".join(lines)


def generate_markdown(data: Dict[str, Any], findings: List[str]) -> str:
    system = data["system"]
    owners = data["owners"]
    architecture = data["architecture"]
    dataset = data["data"]
    model = data["model"]
    operation = data["operation"]
    governance = data["governance"]

    metrics = model.get("metrics", {})

    generated_at = datetime.now(timezone.utc).isoformat()

    return f"""### Documentación técnica AIOps: {system["name"]}

#### Resumen

| Campo | Valor |
|---|---|
| Sistema | {system["name"]} |
| Entorno | {system["environment"]} |
| Criticidad | {system["criticality"]} |
| Generado | {generated_at} |

#### Objetivo operativo

{system["objective"]}

#### Alcance

Incluye:

{render_list(system.get("scope", {}).get("includes", []))}

No incluye:

{render_list(system.get("scope", {}).get("excludes", []))}

#### Owners

| Rol | Responsable |
|---|---|
| Owner funcional | {owners.get("functional", "No documentado")} |
| Owner técnico | {owners.get("technical", "No documentado")} |
| Seguridad | {owners.get("security", "No documentado")} |
| Cloud Admin | {owners.get("cloud_admin", "No documentado")} |

#### Arquitectura

{render_components(architecture.get("components", []))}

#### Datos

| Campo | Valor |
|---|---|
| Dataset | {dataset.get("dataset_uri", "No documentado")} |
| Rango temporal | {dataset.get("time_range", "No documentado")} |
| Label | {dataset.get("label", "No documentado")} |

Features:

{render_list(dataset.get("features", []))}

Limitaciones:

{render_list(dataset.get("limitations", []))}

#### Modelo

| Campo | Valor |
|---|---|
| Modelo | {model.get("id", "No documentado")} |
| Versión | {model.get("version", "No documentado")} |
| Tipo | {model.get("type", "No documentado")} |
| Baseline | {model.get("baseline", "No documentado")} |

Métricas:

| Métrica | Valor |
|---|---:|
| Precision | {metrics.get("precision", "No documentado")} |
| Recall | {metrics.get("recall", "No documentado")} |
| False positive rate | {metrics.get("false_positive_rate", "No documentado")} |
| Latencia p95 ms | {metrics.get("latency_ms_p95", "No documentado")} |

#### Operación

| Campo | Valor |
|---|---|
| Regla de alerta | {operation.get("alert_rule", "No documentado")} |
| Acción | {operation.get("action", "No documentado")} |
| Rollback | {operation.get("rollback", "No documentado")} |

Monitorización:

{render_list(operation.get("monitoring", []))}

#### Gobierno

| Campo | Valor |
|---|---|
| Estado de aprobación | {governance.get("approval_status", "No documentado")} |
| Próxima revisión | {governance.get("next_review_date", "No documentado")} |

Aprobadores:

{render_list(governance.get("approved_by", []))}

Evidencias:

{render_list(governance.get("evidence", []))}

#### Riesgos documentales detectados

{render_list(findings) if findings else "- No se han detectado riesgos documentales."}
"""


def upload_to_gcs(local_file: Path, bucket_name: str, object_name: str) -> None:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_file))
    print(f"Documento subido a gs://{bucket_name}/{object_name}")


def main() -> None:
    input_file = Path("aiops_system.yaml")

    if not input_file.exists():
        raise FileNotFoundError("No existe aiops_system.yaml")

    with input_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    findings = validate_documentation(data)
    markdown = generate_markdown(data, findings)

    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)

    system_name = data["system"]["name"]
    environment = data["system"]["environment"]
    output_file = output_dir / f"{system_name}_{environment}_technical_doc.md"

    output_file.write_text(markdown, encoding="utf-8")

    print(f"Documento generado: {output_file}")

    if findings:
        print("\nRiesgos documentales:")
        for finding in findings:
            print(f"- {finding}")

    if DOCS_BUCKET:
        object_name = f"{DOCS_PREFIX}/{output_file.name}"
        try:
            upload_to_gcs(output_file, DOCS_BUCKET, object_name)
        except (Forbidden, GoogleAPIError) as ex:
            print(f"No se pudo subir a Cloud Storage: {ex}")
            print("El documento queda disponible localmente.")


if __name__ == "__main__":
    main()