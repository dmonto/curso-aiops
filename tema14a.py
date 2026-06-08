import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def generate_incidents(n: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    services = ["checkout", "payments", "inventory", "search", "identity"]
    severities = ["P1", "P2", "P3"]
    environments = ["prod", "preprod"]

    now = datetime.now(timezone.utc)
    rows = []

    for i in range(n):
        service = rng.choice(services, p=[0.25, 0.2, 0.2, 0.2, 0.15])
        severity = rng.choice(severities, p=[0.15, 0.35, 0.50])
        environment = rng.choice(environments, p=[0.85, 0.15])

        start_offset_hours = int(rng.integers(1, 24 * 30))
        started_at = now - timedelta(hours=start_offset_hours)

        # Incidencias más críticas tienden a ser detectadas antes,
        # pero tardan más en resolverse.
        if severity == "P1":
            detect_minutes = max(1, rng.normal(6, 3))
            acknowledge_minutes = max(1, rng.normal(5, 2))
            resolve_minutes = max(15, rng.normal(65, 20))
        elif severity == "P2":
            detect_minutes = max(2, rng.normal(12, 6))
            acknowledge_minutes = max(2, rng.normal(10, 5))
            resolve_minutes = max(20, rng.normal(95, 35))
        else:
            detect_minutes = max(5, rng.normal(30, 15))
            acknowledge_minutes = max(5, rng.normal(25, 10))
            resolve_minutes = max(30, rng.normal(180, 60))

        detected_at = started_at + timedelta(minutes=float(detect_minutes))
        acknowledged_at = detected_at + timedelta(minutes=float(acknowledge_minutes))
        resolved_at = started_at + timedelta(minutes=float(resolve_minutes))

        alert_generated = rng.random() < 0.88
        alert_actionable = alert_generated and (rng.random() < 0.72)
        predicted = rng.random() < 0.38

        # Simula si la automatización resolvió o ayudó.
        auto_remediation_attempted = rng.random() < 0.30
        auto_remediation_success = auto_remediation_attempted and (rng.random() < 0.68)

        rows.append(
            {
                "incident_id": f"INC-{i + 1:04d}",
                "service": service,
                "severity": severity,
                "environment": environment,
                "started_at": started_at,
                "detected_at": detected_at,
                "acknowledged_at": acknowledged_at,
                "resolved_at": resolved_at,
                "alert_generated": alert_generated,
                "alert_actionable": alert_actionable,
                "predicted": predicted,
                "auto_remediation_attempted": auto_remediation_attempted,
                "auto_remediation_success": auto_remediation_success,
            }
        )

    return pd.DataFrame(rows)


def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["mttd_min"] = (df["detected_at"] - df["started_at"]).dt.total_seconds() / 60
    df["mtta_min"] = (df["acknowledged_at"] - df["detected_at"]).dt.total_seconds() / 60
    df["mttr_min"] = (df["resolved_at"] - df["started_at"]).dt.total_seconds() / 60

    group_cols = ["service", "severity", "environment"]

    kpis = (
        df.groupby(group_cols)
        .agg(
            incidents=("incident_id", "count"),
            mttd_min=("mttd_min", "mean"),
            mtta_min=("mtta_min", "mean"),
            mttr_min=("mttr_min", "mean"),
            alerts_generated=("alert_generated", "sum"),
            actionable_alerts=("alert_actionable", "sum"),
            predicted_incidents=("predicted", "sum"),
            auto_attempts=("auto_remediation_attempted", "sum"),
            auto_successes=("auto_remediation_success", "sum"),
        )
        .reset_index()
    )

    kpis["actionable_alert_rate"] = np.where(
        kpis["alerts_generated"] > 0,
        kpis["actionable_alerts"] / kpis["alerts_generated"],
        0,
    )

    kpis["prediction_coverage"] = kpis["predicted_incidents"] / kpis["incidents"]

    kpis["auto_remediation_success_rate"] = np.where(
        kpis["auto_attempts"] > 0,
        kpis["auto_successes"] / kpis["auto_attempts"],
        0,
    )

    # Score operativo simple para priorizar revisión.
    # Cuanto mayor, más atención requiere.
    kpis["operational_risk_score"] = (
        kpis["incidents"] * 0.25
        + kpis["mttd_min"] * 0.20
        + kpis["mtta_min"] * 0.15
        + kpis["mttr_min"] * 0.30
        + (1 - kpis["actionable_alert_rate"]) * 20
    )

    numeric_cols = [
        "mttd_min",
        "mtta_min",
        "mttr_min",
        "actionable_alert_rate",
        "prediction_coverage",
        "auto_remediation_success_rate",
        "operational_risk_score",
    ]

    kpis[numeric_cols] = kpis[numeric_cols].round(2)

    return kpis.sort_values("operational_risk_score", ascending=False)


def write_to_bigquery(kpis: pd.DataFrame) -> None:
    project_id = os.getenv("PROJECT_ID")
    dataset = os.getenv("BQ_DATASET", "aiops_kpis")
    table = os.getenv("BQ_TABLE", "operational_kpis")

    if not project_id:
        print("PROJECT_ID no definido. Se omite carga en BigQuery.")
        return

    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    dataset_id = f"{project_id}.{dataset}"
    table_id = f"{dataset_id}.{table}"

    client.create_dataset(dataset_id, exists_ok=True)

    job = client.load_table_from_dataframe(
        kpis,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()

    print(f"KPIs cargados en BigQuery: {table_id}")


def main() -> None:
    incidents = generate_incidents()
    kpis = calculate_kpis(incidents)

    incidents.to_csv("incidents_sample.csv", index=False)
    kpis.to_csv("operational_kpis.csv", index=False)

    print("\nTop 10 combinaciones con mayor riesgo operativo:\n")
    print(
        kpis[
            [
                "service",
                "severity",
                "environment",
                "incidents",
                "mttd_min",
                "mtta_min",
                "mttr_min",
                "actionable_alert_rate",
                "prediction_coverage",
                "auto_remediation_success_rate",
                "operational_risk_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    write_to_bigquery(kpis)


if __name__ == "__main__":
    main()