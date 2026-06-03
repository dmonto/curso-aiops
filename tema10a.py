import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google.cloud import bigquery


load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET", "aiops_cost")
BQ_LOCATION = os.getenv("BQ_LOCATION", "EU")
BILLING_EXPORT_TABLE = os.getenv("BILLING_EXPORT_TABLE", "").strip()

if not PROJECT_ID:
    raise RuntimeError(
        "Falta GOOGLE_CLOUD_PROJECT o PROJECT_ID en el .env"
    )

client = bigquery.Client(project=PROJECT_ID)

dataset_id = f"{PROJECT_ID}.{BQ_DATASET}"
raw_table = f"{PROJECT_ID}.{BQ_DATASET}.daily_consumption_raw"
analysis_table = f"{PROJECT_ID}.{BQ_DATASET}.daily_service_cost"

Path("outputs").mkdir(exist_ok=True)


def ensure_dataset() -> None:
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = BQ_LOCATION
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset preparado: {dataset_id}")


def run_query(sql: str) -> None:
    job = client.query(sql)
    job.result()


def create_from_billing_export() -> None:
    print(f"Usando Billing Export real: {BILLING_EXPORT_TABLE}")

    sql = f"""
    CREATE OR REPLACE TABLE `{raw_table}` AS
    SELECT
      DATE(usage_start_time) AS usage_date,
      COALESCE(project.id, 'sin_project_id') AS source_project,
      service.description AS service,
      sku.description AS sku,
      SUM(cost) AS gross_cost,
      SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS credits,
      SUM(cost) + SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS net_cost
    FROM `{BILLING_EXPORT_TABLE}`
    WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    GROUP BY
      usage_date,
      source_project,
      service,
      sku
    """
    run_query(sql)
    print(f"Tabla raw creada desde billing export: {raw_table}")


def create_synthetic_consumption() -> None:
    print("No se ha informado BILLING_EXPORT_TABLE. Generando datos sintéticos.")

    rng = np.random.default_rng(42)

    services = {
        "BigQuery": 16.0,
        "Vertex AI": 11.0,
        "Cloud Logging": 6.0,
        "Cloud Storage": 3.5,
        "Pub/Sub": 2.0,
    }

    skus = {
        "BigQuery": "Analysis Queries",
        "Vertex AI": "Online Prediction",
        "Cloud Logging": "Log Ingestion",
        "Cloud Storage": "Standard Storage",
        "Pub/Sub": "Message Delivery",
    }

    dates = pd.date_range(
        end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
        periods=90,
        freq="D",
    )

    rows = []

    for i, day in enumerate(dates):
        weekday_factor = 1.20 if day.weekday() < 5 else 0.65
        trend_factor = 1 + (i / 180)

        for service, base_cost in services.items():
            noise = rng.normal(0, base_cost * 0.08)
            gross_cost = max(0.05, base_cost * weekday_factor * trend_factor + noise)

            # Simulamos dos picos operativos.
            if service == "Vertex AI" and i in [48, 49]:
                gross_cost *= 3.4

            if service == "BigQuery" and i == 68:
                gross_cost *= 2.7

            credits = 0.0
            if service == "BigQuery" and i % 15 == 0:
                credits = -gross_cost * 0.08

            rows.append(
                {
                    "usage_date": day.date(),
                    "source_project": PROJECT_ID,
                    "service": service,
                    "sku": skus[service],
                    "gross_cost": round(float(gross_cost), 4),
                    "credits": round(float(credits), 4),
                    "net_cost": round(float(gross_cost + credits), 4),
                }
            )

    df = pd.DataFrame(rows)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"
    )

    job = client.load_table_from_dataframe(
        df,
        raw_table,
        job_config=job_config,
    )
    job.result()

    print(f"Tabla raw sintética creada: {raw_table}")
    print(f"Filas cargadas: {len(df)}")


def create_analysis_table() -> None:
    sql = f"""
    CREATE OR REPLACE TABLE `{analysis_table}` AS
    WITH daily_service AS (
      SELECT
        usage_date,
        service,
        SUM(gross_cost) AS gross_cost,
        SUM(credits) AS credits,
        SUM(net_cost) AS net_cost
      FROM `{raw_table}`
      GROUP BY usage_date, service
    ),
    daily_total AS (
      SELECT
        usage_date,
        SUM(net_cost) AS total_cost
      FROM daily_service
      GROUP BY usage_date
    )
    SELECT
      ds.usage_date,
      ds.service,
      ds.gross_cost,
      ds.credits,
      ds.net_cost,
      dt.total_cost,
      SAFE_DIVIDE(ds.net_cost, dt.total_cost) AS service_share
    FROM daily_service ds
    JOIN daily_total dt
    USING (usage_date)
    ORDER BY usage_date, service
    """
    run_query(sql)
    print(f"Tabla analítica creada: {analysis_table}")


def analyze_results() -> None:
    df_daily = client.query(
        f"""
        SELECT
        usage_date,
        SUM(net_cost) AS total_cost
        FROM `{analysis_table}`
        GROUP BY usage_date
        ORDER BY usage_date
        """
    ).to_dataframe(create_bqstorage_client=False)

    df_services = client.query(
        f"""
        SELECT
        service,
        SUM(net_cost) AS total_cost,
        AVG(service_share) AS avg_daily_share
        FROM `{analysis_table}`
        GROUP BY service
        ORDER BY total_cost DESC
        """
    ).to_dataframe(create_bqstorage_client=False)

    df_daily["usage_date"] = pd.to_datetime(df_daily["usage_date"])
    df_daily["rolling_mean_7d"] = (
        df_daily["total_cost"]
        .rolling(window=7, min_periods=7)
        .mean()
        .shift(1)
    )
    df_daily["rolling_std_7d"] = (
        df_daily["total_cost"]
        .rolling(window=7, min_periods=7)
        .std()
        .shift(1)
    )

    df_daily["threshold_high"] = (
        df_daily["rolling_mean_7d"] + 3 * df_daily["rolling_std_7d"]
    )

    df_daily["is_spike"] = (
        (df_daily["total_cost"] > df_daily["threshold_high"])
        & df_daily["threshold_high"].notna()
    )

    df_daily.to_csv("outputs/cost_daily_analysis.csv", index=False)
    df_services.to_csv("outputs/cost_by_service.csv", index=False)

    print("\nResumen por servicio")
    print(df_services.to_string(index=False))

    print("\nPosibles picos detectados")
    spikes = df_daily[df_daily["is_spike"]]
    if spikes.empty:
        print("No se han detectado picos con el umbral actual.")
    else:
        print(
            spikes[
                [
                    "usage_date",
                    "total_cost",
                    "rolling_mean_7d",
                    "threshold_high",
                ]
            ].to_string(index=False)
        )

    plt.figure(figsize=(11, 5))
    plt.plot(df_daily["usage_date"], df_daily["total_cost"], label="Coste diario")
    plt.plot(df_daily["usage_date"], df_daily["rolling_mean_7d"], label="Media móvil 7 días")
    plt.plot(df_daily["usage_date"], df_daily["threshold_high"], label="Umbral alto")

    spike_points = df_daily[df_daily["is_spike"]]
    if not spike_points.empty:
        plt.scatter(
            spike_points["usage_date"],
            spike_points["total_cost"],
            label="Posible pico",
        )

    plt.title("Análisis histórico de consumo diario")
    plt.xlabel("Fecha")
    plt.ylabel("Coste neto")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/cost_daily_total.png", dpi=140)

    print("\nFicheros generados:")
    print("- outputs/cost_daily_analysis.csv")
    print("- outputs/cost_by_service.csv")
    print("- outputs/cost_daily_total.png")


def main() -> None:
    ensure_dataset()

    if BILLING_EXPORT_TABLE:
        create_from_billing_export()
    else:
        create_synthetic_consumption()

    create_analysis_table()
    analyze_results()


if __name__ == "__main__":
    main()