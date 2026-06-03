import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno obligatoria: {name}")
    return value


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def ensure_dataset(client: bigquery.Client, project_id: str, dataset_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")

    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset_ref.location = os.getenv("AIOPS_BQ_LOCATION", "EU")
        client.create_dataset(dataset_ref)
        print(f"Dataset creado: {project_id}.{dataset_id}")


def generate_operational_signals() -> pd.DataFrame:
    """
    Genera señales agregadas por servicio.

    En producción, estas señales vendrían de:
    - tablas SLI/SLO
    - incidentes
    - MTTR
    - feedback
    - auditoría de cambios
    - remediaciones
    - ChatOps
    """

    now = utc_now()

    services = [
        {
            "service_name": "checkout-api",
            "owner_team": "sre-payments",
            "criticality": "HIGH",
        },
        {
            "service_name": "incident-triage-api",
            "owner_team": "sre-aiops",
            "criticality": "HIGH",
        },
        {
            "service_name": "logging-pipeline",
            "owner_team": "platform-observability",
            "criticality": "MEDIUM",
        },
        {
            "service_name": "reporting-dashboard",
            "owner_team": "data-platform",
            "criticality": "LOW",
        },
    ]

    rows: List[Dict[str, Any]] = []

    for service in services:
        service_name = service["service_name"]

        # Valores base razonables
        slo_compliance_pct = random.uniform(96.0, 99.9)
        avg_mttr_minutes = random.uniform(20, 70)
        avg_mtta_minutes = random.uniform(3, 20)
        false_positive_pct = random.uniform(5, 30)
        high_risk_changes_30d = random.randint(0, 8)
        changes_without_approval_30d = random.randint(0, 3)
        remediation_success_pct = random.uniform(55, 95)
        feedback_coverage_pct = random.uniform(30, 90)
        open_improvement_items = random.randint(0, 12)

        # Degradaciones intencionadas para que el análisis sea útil.
        if service_name == "incident-triage-api":
            false_positive_pct = random.uniform(25, 42)
            feedback_coverage_pct = random.uniform(55, 85)
            remediation_success_pct = random.uniform(45, 75)
            open_improvement_items = random.randint(6, 14)

        if service_name == "checkout-api":
            slo_compliance_pct = random.uniform(94, 98)
            high_risk_changes_30d = random.randint(4, 10)
            changes_without_approval_30d = random.randint(1, 4)
            avg_mttr_minutes = random.uniform(45, 85)

        if service_name == "logging-pipeline":
            avg_mtta_minutes = random.uniform(12, 28)
            feedback_coverage_pct = random.uniform(20, 50)

        if service_name == "reporting-dashboard":
            slo_compliance_pct = random.uniform(97, 99.8)
            avg_mttr_minutes = random.uniform(30, 90)

        rows.append(
            {
                "signal_ts": now.isoformat(),
                "service_name": service_name,
                "owner_team": service["owner_team"],
                "criticality": service["criticality"],
                "slo_compliance_pct": round(slo_compliance_pct, 2),
                "avg_mttr_minutes": round(avg_mttr_minutes, 2),
                "avg_mtta_minutes": round(avg_mtta_minutes, 2),
                "false_positive_pct": round(false_positive_pct, 2),
                "high_risk_changes_30d": high_risk_changes_30d,
                "changes_without_approval_30d": changes_without_approval_30d,
                "remediation_success_pct": round(remediation_success_pct, 2),
                "feedback_coverage_pct": round(feedback_coverage_pct, 2),
                "open_improvement_items": open_improvement_items,
                "metadata_json": json.dumps(
                    {
                        "generated_by": "aiops_operational_culture_scorecard.py",
                        "window_days": 30,
                    },
                    ensure_ascii=False,
                ),
            }
        )

    return pd.DataFrame(rows)


def load_operational_signals(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
    df: pd.DataFrame,
) -> None:
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    df = df.copy()

    # BigQuery espera TIMESTAMP, no object/string.
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)

    df["service_name"] = df["service_name"].astype(str)
    df["owner_team"] = df["owner_team"].astype(str)
    df["criticality"] = df["criticality"].astype(str)
    df["slo_compliance_pct"] = df["slo_compliance_pct"].astype("float64")
    df["avg_mttr_minutes"] = df["avg_mttr_minutes"].astype("float64")
    df["avg_mtta_minutes"] = df["avg_mtta_minutes"].astype("float64")
    df["false_positive_pct"] = df["false_positive_pct"].astype("float64")
    df["high_risk_changes_30d"] = df["high_risk_changes_30d"].astype("int64")
    df["changes_without_approval_30d"] = df["changes_without_approval_30d"].astype("int64")
    df["remediation_success_pct"] = df["remediation_success_pct"].astype("float64")
    df["feedback_coverage_pct"] = df["feedback_coverage_pct"].astype("float64")
    df["open_improvement_items"] = df["open_improvement_items"].astype("int64")
    df["metadata_json"] = df["metadata_json"].astype(str)

    print("\nTipos antes de cargar a BigQuery:")
    print(df.dtypes)

    schema = [
        bigquery.SchemaField("signal_ts", "TIMESTAMP"),
        bigquery.SchemaField("service_name", "STRING"),
        bigquery.SchemaField("owner_team", "STRING"),
        bigquery.SchemaField("criticality", "STRING"),
        bigquery.SchemaField("slo_compliance_pct", "FLOAT"),
        bigquery.SchemaField("avg_mttr_minutes", "FLOAT"),
        bigquery.SchemaField("avg_mtta_minutes", "FLOAT"),
        bigquery.SchemaField("false_positive_pct", "FLOAT"),
        bigquery.SchemaField("high_risk_changes_30d", "INTEGER"),
        bigquery.SchemaField("changes_without_approval_30d", "INTEGER"),
        bigquery.SchemaField("remediation_success_pct", "FLOAT"),
        bigquery.SchemaField("feedback_coverage_pct", "FLOAT"),
        bigquery.SchemaField("open_improvement_items", "INTEGER"),
        bigquery.SchemaField("metadata_json", "STRING"),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    print(f"Señales operativas cargadas en BigQuery: {table_ref}")
    print(f"Filas: {len(df)}")

def create_scorecard(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    signals_table: str,
    scorecard_table: str,
) -> pd.DataFrame:
    source = f"`{project_id}.{dataset_id}.{signals_table}`"
    target = f"`{project_id}.{dataset_id}.{scorecard_table}`"

    query = f"""
    CREATE OR REPLACE TABLE {target} AS
    WITH normalized AS (
      SELECT
        signal_ts,
        service_name,
        owner_team,
        criticality,

        slo_compliance_pct,
        avg_mttr_minutes,
        avg_mtta_minutes,
        false_positive_pct,
        high_risk_changes_30d,
        changes_without_approval_30d,
        remediation_success_pct,
        feedback_coverage_pct,
        open_improvement_items,

        -- Subscores de 0 a 100. Cuanto mayor, mejor.
        LEAST(100, GREATEST(0, slo_compliance_pct)) AS reliability_score,

        LEAST(100, GREATEST(0, 100 - avg_mttr_minutes)) AS recovery_score,

        LEAST(100, GREATEST(0, 100 - (avg_mtta_minutes * 4))) AS response_score,

        LEAST(100, GREATEST(0, 100 - (false_positive_pct * 2))) AS alert_quality_score,

        LEAST(
          100,
          GREATEST(
            0,
            100
            - (high_risk_changes_30d * 8)
            - (changes_without_approval_30d * 15)
          )
        ) AS change_governance_score,

        LEAST(100, GREATEST(0, remediation_success_pct)) AS remediation_score,

        LEAST(100, GREATEST(0, feedback_coverage_pct)) AS learning_score,

        LEAST(100, GREATEST(0, 100 - (open_improvement_items * 5))) AS improvement_debt_score
      FROM {source}
    ),
    scored AS (
      SELECT
        *,
        ROUND(
          reliability_score * 0.20
          + recovery_score * 0.15
          + response_score * 0.10
          + alert_quality_score * 0.15
          + change_governance_score * 0.15
          + remediation_score * 0.10
          + learning_score * 0.10
          + improvement_debt_score * 0.05,
          2
        ) AS operational_health_score
      FROM normalized
    ),
    interpreted AS (
      SELECT
        *,
        CASE
          WHEN operational_health_score >= 85 THEN 'HEALTHY'
          WHEN operational_health_score >= 70 THEN 'WATCH'
          WHEN operational_health_score >= 55 THEN 'AT_RISK'
          ELSE 'CRITICAL_ATTENTION'
        END AS operational_status,

        CASE
          WHEN reliability_score < 97 THEN 'SLO_REVIEW'
          WHEN change_governance_score < 70 THEN 'CHANGE_GOVERNANCE'
          WHEN alert_quality_score < 70 THEN 'REDUCE_ALERT_NOISE'
          WHEN recovery_score < 60 THEN 'MTTR_REDUCTION'
          WHEN remediation_score < 70 THEN 'REMEDIATION_REVIEW'
          WHEN learning_score < 60 THEN 'IMPROVE_FEEDBACK_LOOP'
          WHEN improvement_debt_score < 60 THEN 'REDUCE_OPERATIONAL_DEBT'
          ELSE 'CONTINUE_MONITORING'
        END AS primary_improvement_area
      FROM scored
    )
    SELECT
      *,
      CASE primary_improvement_area
        WHEN 'SLO_REVIEW'
          THEN 'Revisar SLO, error budget y causas de incumplimiento.'
        WHEN 'CHANGE_GOVERNANCE'
          THEN 'Revisar cambios de alto riesgo, aprobaciones y gates CI/CD.'
        WHEN 'REDUCE_ALERT_NOISE'
          THEN 'Ajustar umbrales, ventanas y reglas para reducir falsos positivos.'
        WHEN 'MTTR_REDUCTION'
          THEN 'Revisar detección, acknowledge, runbooks y remediaciones.'
        WHEN 'REMEDIATION_REVIEW'
          THEN 'Validar si las remediaciones reducen impacto real.'
        WHEN 'IMPROVE_FEEDBACK_LOOP'
          THEN 'Aumentar cobertura de feedback post-incidente y ChatOps.'
        WHEN 'REDUCE_OPERATIONAL_DEBT'
          THEN 'Priorizar backlog de mejoras operativas abiertas.'
        ELSE 'Mantener seguimiento y revisar tendencia en el siguiente ciclo.'
      END AS recommended_action
    FROM interpreted
    ORDER BY operational_health_score ASC;
    """

    client.query(query).result()

    result_query = f"""
    SELECT
      service_name,
      owner_team,
      criticality,
      operational_health_score,
      operational_status,
      primary_improvement_area,
      recommended_action,
      slo_compliance_pct,
      avg_mttr_minutes,
      avg_mtta_minutes,
      false_positive_pct,
      high_risk_changes_30d,
      changes_without_approval_30d,
      remediation_success_pct,
      feedback_coverage_pct,
      open_improvement_items
    FROM {target}
    ORDER BY operational_health_score ASC;
    """

    job = client.query(result_query)
    return job.to_dataframe(create_bqstorage_client=False)


def ensure_decision_log_table(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> bigquery.Table:
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    schema = [
        bigquery.SchemaField("decision_id", "STRING"),
        bigquery.SchemaField("decision_ts", "TIMESTAMP"),
        bigquery.SchemaField("service_name", "STRING"),
        bigquery.SchemaField("owner_team", "STRING"),
        bigquery.SchemaField("decision_type", "STRING"),
        bigquery.SchemaField("decision_summary", "STRING"),
        bigquery.SchemaField("evidence_json", "STRING"),
        bigquery.SchemaField("status", "STRING"),
    ]

    try:
        return client.get_table(table_ref)
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        print(f"Tabla de decision log creada: {table_ref}")
        return table


def create_decision_log_entries(
    client: bigquery.Client,
    table: bigquery.Table,
    scorecard_df: pd.DataFrame,
) -> None:
    rows: List[Dict[str, Any]] = []

    for _, row in scorecard_df.iterrows():
        status = str(row["operational_status"])

        if status not in {"AT_RISK", "CRITICAL_ATTENTION"}:
            continue

        decision_summary = (
            f"Priorizar mejora {row['primary_improvement_area']} para "
            f"{row['service_name']} por score operativo {row['operational_health_score']}."
        )

        evidence = {
            "operational_health_score": row["operational_health_score"],
            "operational_status": row["operational_status"],
            "primary_improvement_area": row["primary_improvement_area"],
            "slo_compliance_pct": row["slo_compliance_pct"],
            "avg_mttr_minutes": row["avg_mttr_minutes"],
            "false_positive_pct": row["false_positive_pct"],
            "high_risk_changes_30d": row["high_risk_changes_30d"],
            "changes_without_approval_30d": row["changes_without_approval_30d"],
            "remediation_success_pct": row["remediation_success_pct"],
            "feedback_coverage_pct": row["feedback_coverage_pct"],
            "open_improvement_items": row["open_improvement_items"],
        }

        rows.append(
            {
                "decision_id": f"DEC-{uuid.uuid4().hex[:12].upper()}",
                "decision_ts": utc_now().isoformat(),
                "service_name": row["service_name"],
                "owner_team": row["owner_team"],
                "decision_type": "OPERATING_REVIEW_ACTION",
                "decision_summary": decision_summary,
                "evidence_json": json.dumps(evidence, ensure_ascii=False, default=str),
                "status": "OPEN",
            }
        )

    if not rows:
        print("No se han generado decisiones operativas. Todos los servicios están en estado aceptable.")
        return

    errors = client.insert_rows_json(table, rows)
    if errors:
        raise RuntimeError(f"Error insertando decision log: {errors}")

    print(f"Decisiones operativas registradas: {len(rows)}")


def main() -> int:
    try:
        project_id = require_env("PROJECT_ID")
        dataset_id = os.getenv("AIOPS_DATASET", "aiops_lab")
        signals_table = os.getenv("AIOPS_OPERATIONAL_SIGNALS_TABLE", "operational_signals")
        scorecard_table = os.getenv("AIOPS_SCORECARD_TABLE", "operational_scorecard")
        decision_log_table = os.getenv("AIOPS_DECISION_LOG_TABLE", "operational_decision_log")

        client = bigquery.Client(project=project_id)

        ensure_dataset(client, project_id, dataset_id)

        signals_df = generate_operational_signals()

        load_operational_signals(
            client=client,
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=signals_table,
            df=signals_df,
        )

        scorecard_df = create_scorecard(
            client=client,
            project_id=project_id,
            dataset_id=dataset_id,
            signals_table=signals_table,
            scorecard_table=scorecard_table,
        )

        decision_log_bq_table = ensure_decision_log_table(
            client=client,
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=decision_log_table,
        )

        create_decision_log_entries(client, decision_log_bq_table, scorecard_df)

        print("\nOperational scorecard")
        print(scorecard_df.to_string(index=False))

        critical_or_risk = scorecard_df[
            scorecard_df["operational_status"].isin(["AT_RISK", "CRITICAL_ATTENTION"])
        ]

        if not critical_or_risk.empty:
            print(f"\nServicios que requieren revisión: {len(critical_or_risk)}")
            return 2

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())