import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request
from google.api_core.exceptions import NotFound
from google.cloud import bigquery


load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET", "aiops_cost")
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")

RIGHTSIZING_LOCATIONS = [
    x.strip()
    for x in os.getenv(
        "RIGHTSIZING_LOCATIONS",
        "us-central1,northamerica-south1,us-south1",
    ).split(",")
    if x.strip()
]

RIGHTSIZING_APPLY_CHANGES = (
    os.getenv("RIGHTSIZING_APPLY_CHANGES", "false").lower() == "true"
)

RIGHTSIZING_MIN_SAVING_EUR = float(
    os.getenv("RIGHTSIZING_MIN_SAVING_EUR", "10")
)

RIGHTSIZING_ALLOW_AUTO_DEV = (
    os.getenv("RIGHTSIZING_ALLOW_AUTO_DEV", "true").lower() == "true"
)

RIGHTSIZING_REQUIRE_APPROVAL_PROD = (
    os.getenv("RIGHTSIZING_REQUIRE_APPROVAL_PROD", "true").lower() == "true"
)

if not PROJECT_ID:
    raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT o PROJECT_ID en el .env")

client = bigquery.Client(project=PROJECT_ID)

PLAN_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.rightsizing_action_plan"
OPTIMIZATION_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.resource_optimization_candidates"


RECOMMENDERS = [
    "google.compute.instance.MachineTypeRecommender",
    "google.compute.instance.IdleResourceRecommender",
    "google.compute.disk.IdleResourceRecommender",
    "google.cloudsql.instance.OverprovisionedRecommender",
]


def get_access_token() -> str:
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


def money_to_float(cost_projection: Dict[str, Any]) -> float:
    """
    Convierte google.type.Money a float aproximado.
    Algunos recommenders pueden no incluir impacto económico.
    """
    if not cost_projection:
        return 0.0

    cost = cost_projection.get("cost", {})
    units = float(cost.get("units", 0) or 0)
    nanos = float(cost.get("nanos", 0) or 0) / 1_000_000_000

    return units + nanos


def extract_target_resource(recommendation: Dict[str, Any]) -> Optional[str]:
    groups = recommendation.get("content", {}).get("operationGroups", [])
    for group in groups:
        for operation in group.get("operations", []):
            resource = operation.get("resource")
            if resource:
                return resource
    return None


def extract_operations_summary(recommendation: Dict[str, Any]) -> str:
    groups = recommendation.get("content", {}).get("operationGroups", [])
    operations = []

    for group in groups:
        for operation in group.get("operations", []):
            action = operation.get("action", "")
            path = operation.get("path", "")
            value = operation.get("value", "")
            operations.append(
                {
                    "action": action,
                    "path": path,
                    "value": value,
                }
            )

    return json.dumps(operations, ensure_ascii=False)


def fetch_recommendations_for(
    project_id: str,
    location: str,
    recommender_id: str,
    token: str,
) -> List[Dict[str, Any]]:
    base_url = (
        "https://recommender.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/"
        f"recommenders/{recommender_id}/recommendations"
    )

    headers = {"Authorization": f"Bearer {token}"}
    params = {"pageSize": 100}
    recommendations = []

    while True:
        response = requests.get(base_url, headers=headers, params=params, timeout=30)

        if response.status_code in (403, 404):
            print(
                f"No se pudo consultar {recommender_id} en {location}: "
                f"{response.status_code}"
            )
            return []

        response.raise_for_status()
        payload = response.json()

        recommendations.extend(payload.get("recommendations", []))

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

        params["pageToken"] = next_page_token

    return recommendations


def fetch_real_recommendations() -> List[Dict[str, Any]]:
    print("\nConsultando Recommender API...")

    token = get_access_token()
    rows = []

    for location in RIGHTSIZING_LOCATIONS:
        for recommender_id in RECOMMENDERS:
            recs = fetch_recommendations_for(
                PROJECT_ID,
                location,
                recommender_id,
                token,
            )

            for rec in recs:
                primary_impact = rec.get("primaryImpact", {})
                cost_saving = abs(
                    money_to_float(primary_impact.get("costProjection", {}))
                )

                target_resource = extract_target_resource(rec)
                operations_summary = extract_operations_summary(rec)

                rows.append(
                    {
                        "source": "RECOMMENDER_API",
                        "location": location,
                        "recommender_id": recommender_id,
                        "recommendation_name": rec.get("name"),
                        "description": rec.get("description"),
                        "subtype": rec.get("recommenderSubtype"),
                        "state": rec.get("stateInfo", {}).get("state"),
                        "target_resource": target_resource,
                        "estimated_monthly_saving": cost_saving,
                        "operations_summary": operations_summary,
                        "raw_json": json.dumps(rec, ensure_ascii=False),
                    }
                )

    print(f"Recomendaciones reales encontradas: {len(rows)}")
    return rows


def load_synthetic_candidates() -> List[Dict[str, Any]]:
    """
    Fallback para laboratorio:
    si no hay permisos o recomendaciones reales, se generan candidatos
    a partir de la tabla de optimización del tema anterior.
    """
    try:
        client.get_table(OPTIMIZATION_TABLE)
    except NotFound:
        print(
            f"No existe {OPTIMIZATION_TABLE}. "
            "Se crearán candidatos sintéticos mínimos."
        )
        return [
            {
                "source": "LAB_SYNTHETIC",
                "location": "lab",
                "recommender_id": "synthetic.rightsizing",
                "recommendation_name": "synthetic-vm-rightsize-1",
                "description": "Reducir una VM de laboratorio con baja utilización estimada.",
                "subtype": "CHANGE_MACHINE_TYPE",
                "state": "ACTIVE",
                "target_resource": (
                    "//compute.googleapis.com/projects/"
                    f"{PROJECT_ID}/zones/europe-west1-b/instances/lab-vm-aiops-01"
                ),
                "estimated_monthly_saving": 35.0,
                "operations_summary": json.dumps(
                    [
                        {
                            "action": "replace",
                            "path": "/machineType",
                            "value": "n2-standard-2",
                        }
                    ],
                    ensure_ascii=False,
                ),
                "raw_json": "{}",
            }
        ]

    sql = f"""
    SELECT
      service,
      optimization_type,
      monthly_cost_estimate,
      estimated_monthly_saving,
      risk,
      effort,
      recommendation
    FROM `{OPTIMIZATION_TABLE}`
    WHERE estimated_monthly_saving >= {RIGHTSIZING_MIN_SAVING_EUR}
    ORDER BY priority_score DESC
    LIMIT 10
    """

    df = client.query(sql, location=BQ_LOCATION).to_dataframe()

    rows = []

    for idx, row in df.iterrows():
        service = str(row["service"])
        saving = float(row["estimated_monthly_saving"])

        rows.append(
            {
                "source": "LAB_FROM_OPTIMIZATION_TABLE",
                "location": "lab",
                "recommender_id": "synthetic.rightsizing",
                "recommendation_name": f"synthetic-rightsizing-{idx + 1}",
                "description": f"Revisión de capacidad para {service}: {row['recommendation']}",
                "subtype": str(row["optimization_type"]),
                "state": "ACTIVE",
                "target_resource": f"//lab.googleapis.com/projects/{PROJECT_ID}/services/{service}",
                "estimated_monthly_saving": saving,
                "operations_summary": json.dumps(
                    [
                        {
                            "action": "review",
                            "path": "/capacity",
                            "value": "rightsize-required",
                        }
                    ],
                    ensure_ascii=False,
                ),
                "raw_json": json.dumps(row.to_dict(), ensure_ascii=False),
            }
        )

    print(f"Candidatos sintéticos generados: {len(rows)}")
    return rows


def classify_environment(target_resource: Optional[str]) -> str:
    if not target_resource:
        return "unknown"

    value = target_resource.lower()

    if any(token in value for token in ["prod", "production"]):
        return "prod"

    if any(token in value for token in ["dev", "test", "uat", "lab", "sandbox"]):
        return "dev"

    return "unknown"


def classify_risk(row: Dict[str, Any], env: str) -> str:
    recommender_id = row.get("recommender_id", "")
    subtype = row.get("subtype", "") or ""

    if env == "prod":
        return "HIGH"

    if "IdleResourceRecommender" in recommender_id:
        return "MEDIUM"

    if "MachineTypeRecommender" in recommender_id:
        return "MEDIUM"

    if "OverprovisionedRecommender" in recommender_id:
        return "HIGH"

    if "ENDPOINT" in subtype.upper():
        return "HIGH"

    return "MEDIUM"


def decide_automation_mode(env: str, risk: str, saving: float) -> str:
    if env == "prod" and RIGHTSIZING_REQUIRE_APPROVAL_PROD:
        return "MANUAL_APPROVAL"

    if env == "dev" and RIGHTSIZING_ALLOW_AUTO_DEV and risk in ("LOW", "MEDIUM"):
        if saving <= 100:
            return "AUTO_ELIGIBLE_DRY_RUN"

    if risk == "HIGH":
        return "MANUAL_APPROVAL"

    return "SEMI_AUTOMATIC"


def build_action_plan_rows(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan_rows = []

    for rec in recommendations:
        saving = float(rec.get("estimated_monthly_saving", 0) or 0)

        if saving < RIGHTSIZING_MIN_SAVING_EUR:
            continue

        target_resource = rec.get("target_resource")
        env = classify_environment(target_resource)
        risk = classify_risk(rec, env)
        automation_mode = decide_automation_mode(env, risk, saving)

        approval_required = automation_mode in ("MANUAL_APPROVAL", "SEMI_AUTOMATIC")

        status = "PROPOSED"
        if automation_mode == "AUTO_ELIGIBLE_DRY_RUN":
            status = "DRY_RUN_ONLY"

        plan_rows.append(
            {
                "plan_id": str(uuid.uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": rec.get("source"),
                "location": rec.get("location"),
                "recommender_id": rec.get("recommender_id"),
                "recommendation_name": rec.get("recommendation_name"),
                "target_resource": target_resource,
                "environment": env,
                "recommendation_type": rec.get("subtype"),
                "description": rec.get("description"),
                "estimated_monthly_saving": round(saving, 4),
                "risk": risk,
                "automation_mode": automation_mode,
                "approval_required": approval_required,
                "apply_changes": False,
                "status": status,
                "operations_summary": rec.get("operations_summary"),
                "rollback_required": True,
                "validation_required": (
                    "Revisar CPU, memoria, latencia, errores, SLO, owner y ventana de cambio."
                ),
                "rollback_plan": (
                    "Restaurar configuración anterior si aumenta latencia, error rate, CPU p95 "
                    "o memoria p95 por encima del umbral acordado."
                ),
                "raw_json": rec.get("raw_json"),
            }
        )

    return plan_rows


def ensure_plan_table() -> None:
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, BQ_DATASET)
    table_ref = dataset_ref.table("rightsizing_action_plan")

    schema = [
        bigquery.SchemaField("plan_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("location", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("recommender_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("recommendation_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("target_resource", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("environment", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("recommendation_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("estimated_monthly_saving", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("risk", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("automation_mode", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("approval_required", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("apply_changes", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("operations_summary", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("rollback_required", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("validation_required", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("rollback_plan", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("raw_json", "STRING", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_ref, schema=schema)

    try:
        client.get_table(table)
    except NotFound:
        client.create_table(table)
        print(f"Tabla creada: {PLAN_TABLE}")


def persist_plan(plan_rows: List[Dict[str, Any]]) -> None:
    if not plan_rows:
        print("No hay acciones de rightsizing por encima del umbral configurado.")
        return

    errors = client.insert_rows_json(PLAN_TABLE, plan_rows)

    if errors:
        raise RuntimeError(f"Error insertando plan de rightsizing: {errors}")

    print(f"Acciones insertadas en plan: {len(plan_rows)}")


def print_plan(plan_rows: List[Dict[str, Any]]) -> None:
    if not plan_rows:
        return

    df = pd.DataFrame(plan_rows)

    columns = [
        "target_resource",
        "environment",
        "recommendation_type",
        "estimated_monthly_saving",
        "risk",
        "automation_mode",
        "status",
    ]

    print("\nPlan de rightsizing")
    print(df[columns].to_string(index=False))


def main() -> None:
    ensure_plan_table()

    real_recommendations = []

    try:
        real_recommendations = fetch_real_recommendations()
    except Exception as exc:
        print(f"No se pudieron obtener recomendaciones reales: {exc}")

    if real_recommendations:
        recommendations = real_recommendations
    else:
        recommendations = load_synthetic_candidates()

    plan_rows = build_action_plan_rows(recommendations)

    print_plan(plan_rows)
    persist_plan(plan_rows)

    if RIGHTSIZING_APPLY_CHANGES:
        print(
            "\nRIGHTSIZING_APPLY_CHANGES=true, pero este laboratorio no aplica cambios reales. "
            "La ejecución debe implementarse solo con aprobación, rollback y entorno controlado."
        )

    print("\nConsulta recomendada:")
    print(f"""
SELECT
  created_at,
  source,
  target_resource,
  environment,
  recommendation_type,
  estimated_monthly_saving,
  risk,
  automation_mode,
  approval_required,
  status,
  validation_required,
  rollback_plan
FROM `{PLAN_TABLE}`
ORDER BY created_at DESC, estimated_monthly_saving DESC
LIMIT 20;
""")


if __name__ == "__main__":
    main()