import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EconomicAssumptions:
    sre_hourly_cost_eur: float = 65.0
    alert_review_minutes: float = 4.0
    automation_manual_minutes_saved: float = 18.0
    aiops_monthly_platform_cost_eur: float = 5200.0
    aiops_monthly_operations_cost_eur: float = 3800.0
    initial_investment_eur: float = 28000.0


def generate_monthly_service_data(seed: int = 52) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    services = [
        {
            "service": "checkout",
            "criticality": "alta",
            "cost_per_downtime_min_eur": 420,
            "baseline_cloud_spend_eur": 18000,
        },
        {
            "service": "payments",
            "criticality": "alta",
            "cost_per_downtime_min_eur": 520,
            "baseline_cloud_spend_eur": 22000,
        },
        {
            "service": "identity",
            "criticality": "alta",
            "cost_per_downtime_min_eur": 260,
            "baseline_cloud_spend_eur": 12000,
        },
        {
            "service": "search",
            "criticality": "media",
            "cost_per_downtime_min_eur": 90,
            "baseline_cloud_spend_eur": 15000,
        },
        {
            "service": "inventory",
            "criticality": "media",
            "cost_per_downtime_min_eur": 70,
            "baseline_cloud_spend_eur": 9500,
        },
    ]

    rows = []

    for svc in services:
        baseline_incidents = int(rng.integers(12, 28))
        incident_reduction_pct = rng.uniform(0.12, 0.35)

        current_incidents = int(round(baseline_incidents * (1 - incident_reduction_pct)))

        baseline_avg_mttr = rng.uniform(55, 130)
        mttr_reduction_pct = rng.uniform(0.20, 0.45)
        current_avg_mttr = baseline_avg_mttr * (1 - mttr_reduction_pct)

        baseline_alerts = int(rng.integers(600, 1600))
        alert_noise_reduction_pct = rng.uniform(0.15, 0.40)
        current_alerts = int(round(baseline_alerts * (1 - alert_noise_reduction_pct)))

        automation_executions = int(rng.integers(20, 160))

        cloud_optimization_pct = rng.uniform(0.03, 0.16)
        current_cloud_spend = svc["baseline_cloud_spend_eur"] * (1 - cloud_optimization_pct)

        rows.append(
            {
                "service": svc["service"],
                "criticality": svc["criticality"],
                "cost_per_downtime_min_eur": svc["cost_per_downtime_min_eur"],
                "baseline_incidents": baseline_incidents,
                "current_incidents": current_incidents,
                "baseline_avg_mttr_min": round(baseline_avg_mttr, 2),
                "current_avg_mttr_min": round(current_avg_mttr, 2),
                "baseline_alerts": baseline_alerts,
                "current_alerts": current_alerts,
                "automation_executions": automation_executions,
                "baseline_cloud_spend_eur": svc["baseline_cloud_spend_eur"],
                "current_cloud_spend_eur": round(current_cloud_spend, 2),
            }
        )

    return pd.DataFrame(rows)


def calculate_savings(
    df: pd.DataFrame,
    assumptions: EconomicAssumptions,
) -> pd.DataFrame:
    result = df.copy()

    result["incidents_avoided"] = (
        result["baseline_incidents"] - result["current_incidents"]
    )

    result["baseline_downtime_min"] = (
        result["baseline_incidents"] * result["baseline_avg_mttr_min"]
    )

    result["current_downtime_min"] = (
        result["current_incidents"] * result["current_avg_mttr_min"]
    )

    result["downtime_min_reduced"] = (
        result["baseline_downtime_min"] - result["current_downtime_min"]
    )

    result["downtime_savings_eur"] = (
        result["downtime_min_reduced"] * result["cost_per_downtime_min_eur"]
    )

    result["alerts_reduced"] = result["baseline_alerts"] - result["current_alerts"]

    result["alert_noise_savings_eur"] = (
        result["alerts_reduced"]
        * assumptions.alert_review_minutes
        * assumptions.sre_hourly_cost_eur
        / 60
    )

    result["automation_savings_eur"] = (
        result["automation_executions"]
        * assumptions.automation_manual_minutes_saved
        * assumptions.sre_hourly_cost_eur
        / 60
    )

    result["cloud_optimization_savings_eur"] = (
        result["baseline_cloud_spend_eur"] - result["current_cloud_spend_eur"]
    )

    result["gross_savings_eur"] = (
        result["downtime_savings_eur"]
        + result["alert_noise_savings_eur"]
        + result["automation_savings_eur"]
        + result["cloud_optimization_savings_eur"]
    )

    # Repartimos el coste mensual de AIOps entre servicios proporcionalmente al ahorro bruto.
    total_gross = result["gross_savings_eur"].sum()
    monthly_aiops_cost = (
        assumptions.aiops_monthly_platform_cost_eur
        + assumptions.aiops_monthly_operations_cost_eur
    )

    result["allocated_aiops_cost_eur"] = np.where(
        total_gross > 0,
        result["gross_savings_eur"] / total_gross * monthly_aiops_cost,
        monthly_aiops_cost / len(result),
    )

    result["net_savings_eur"] = (
        result["gross_savings_eur"] - result["allocated_aiops_cost_eur"]
    )

    result["roi"] = np.where(
        result["allocated_aiops_cost_eur"] > 0,
        result["net_savings_eur"] / result["allocated_aiops_cost_eur"],
        np.nan,
    )

    numeric_cols = [
        "baseline_downtime_min",
        "current_downtime_min",
        "downtime_min_reduced",
        "downtime_savings_eur",
        "alert_noise_savings_eur",
        "automation_savings_eur",
        "cloud_optimization_savings_eur",
        "gross_savings_eur",
        "allocated_aiops_cost_eur",
        "net_savings_eur",
        "roi",
    ]

    result[numeric_cols] = result[numeric_cols].round(2)

    return result.sort_values("net_savings_eur", ascending=False)


def create_executive_summary(
    savings: pd.DataFrame,
    assumptions: EconomicAssumptions,
) -> pd.DataFrame:
    total_gross = savings["gross_savings_eur"].sum()
    total_aiops_cost = (
        assumptions.aiops_monthly_platform_cost_eur
        + assumptions.aiops_monthly_operations_cost_eur
    )
    total_net = total_gross - total_aiops_cost

    roi = total_net / total_aiops_cost if total_aiops_cost > 0 else np.nan
    payback_months = (
        assumptions.initial_investment_eur / total_net if total_net > 0 else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "Ahorro bruto mensual",
                "value_eur": round(total_gross, 2),
                "value_text": f"{total_gross:,.2f} €",
            },
            {
                "metric": "Coste mensual AIOps",
                "value_eur": round(total_aiops_cost, 2),
                "value_text": f"{total_aiops_cost:,.2f} €",
            },
            {
                "metric": "Ahorro neto mensual",
                "value_eur": round(total_net, 2),
                "value_text": f"{total_net:,.2f} €",
            },
            {
                "metric": "ROI mensual",
                "value_eur": round(roi, 2),
                "value_text": f"{roi:.2f}x",
            },
            {
                "metric": "Payback estimado",
                "value_eur": round(payback_months, 2),
                "value_text": f"{payback_months:.2f} meses",
            },
        ]
    )

    return summary


def create_savings_breakdown(savings: pd.DataFrame) -> pd.DataFrame:
    breakdown = pd.DataFrame(
        [
            {
                "category": "Reducción de downtime",
                "amount_eur": savings["downtime_savings_eur"].sum(),
            },
            {
                "category": "Reducción de ruido de alertas",
                "amount_eur": savings["alert_noise_savings_eur"].sum(),
            },
            {
                "category": "Automatización operativa",
                "amount_eur": savings["automation_savings_eur"].sum(),
            },
            {
                "category": "Optimización cloud",
                "amount_eur": savings["cloud_optimization_savings_eur"].sum(),
            },
        ]
    )

    breakdown["amount_eur"] = breakdown["amount_eur"].round(2)
    breakdown["percent_of_gross_savings"] = (
        breakdown["amount_eur"] / breakdown["amount_eur"].sum() * 100
    ).round(2)

    return breakdown.sort_values("amount_eur", ascending=False)


def classify_confidence(row: pd.Series) -> str:
    if row["criticality"] == "alta" and row["baseline_incidents"] >= 15:
        return "Alta: baseline suficiente y servicio crítico"
    if row["baseline_incidents"] >= 10:
        return "Media: baseline razonable"
    return "Baja: requiere más histórico"


def add_governance_fields(savings: pd.DataFrame) -> pd.DataFrame:
    result = savings.copy()
    result["confidence_level"] = result.apply(classify_confidence, axis=1)

    result["executive_note"] = np.where(
        result["net_savings_eur"] > 50000,
        "Prioridad ejecutiva: mantener inversión y ampliar automatización",
        np.where(
            result["net_savings_eur"] > 15000,
            "Buen candidato para consolidar y monitorizar mensualmente",
            "Revisar supuestos antes de escalar inversión",
        ),
    )

    return result


def write_to_bigquery(df: pd.DataFrame, table_name: str) -> None:
    project_id = os.getenv("PROJECT_ID")
    dataset = os.getenv("BQ_DATASET", "aiops_kpis")

    if not project_id:
        print(f"PROJECT_ID no definido. Se omite carga en BigQuery para {table_name}.")
        return

    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    dataset_id = f"{project_id}.{dataset}"
    table_id = f"{dataset_id}.{table_name}"

    client.create_dataset(dataset_id, exists_ok=True)

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()

    print(f"Tabla cargada en BigQuery: {table_id}")


def main() -> None:
    assumptions = EconomicAssumptions()

    monthly_data = generate_monthly_service_data()
    savings = calculate_savings(monthly_data, assumptions)
    savings = add_governance_fields(savings)

    executive_summary = create_executive_summary(savings, assumptions)
    breakdown = create_savings_breakdown(savings)

    monthly_data.to_csv("economic_baseline_and_current.csv", index=False)
    savings.to_csv("economic_savings_by_service.csv", index=False)
    executive_summary.to_csv("economic_executive_summary.csv", index=False)
    breakdown.to_csv("economic_savings_breakdown.csv", index=False)

    print("\nResumen ejecutivo:\n")
    print(executive_summary.to_string(index=False))

    print("\nAhorro por servicio:\n")
    print(
        savings[
            [
                "service",
                "criticality",
                "incidents_avoided",
                "downtime_min_reduced",
                "downtime_savings_eur",
                "alert_noise_savings_eur",
                "automation_savings_eur",
                "cloud_optimization_savings_eur",
                "gross_savings_eur",
                "allocated_aiops_cost_eur",
                "net_savings_eur",
                "roi",
                "confidence_level",
                "executive_note",
            ]
        ].to_string(index=False)
    )

    print("\nDesglose de ahorro bruto:\n")
    print(breakdown.to_string(index=False))

    write_to_bigquery(monthly_data, "economic_baseline_and_current")
    write_to_bigquery(savings, "economic_savings_by_service")
    write_to_bigquery(executive_summary, "economic_executive_summary")
    write_to_bigquery(breakdown, "economic_savings_breakdown")


if __name__ == "__main__":
    main()