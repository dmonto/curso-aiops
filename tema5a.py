import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from google.cloud import bigquery
import google.auth


DATASET_ID = "curso_aiops"
TABLE_ID = "metric_latency_hourly"
MODEL_ID = "model_latency_forecast_arima"
FORECAST_TABLE_ID = "forecast_latency_24h"

LOCATION = os.getenv("BIGQUERY_LOCATION", "EU")


def get_project_id() -> str:
    project_id = os.getenv("GCP_PROJECT_ID")

    if project_id:
        return project_id

    _, inferred_project = google.auth.default()

    if not inferred_project:
        raise RuntimeError(
            "No se ha podido inferir el proyecto. Define GCP_PROJECT_ID."
        )

    return inferred_project


def build_synthetic_latency_data() -> pd.DataFrame:
    """
    Genera una serie temporal sintética de latencia p95 por servicio.
    La forma intenta simular patrones reales:
    - estacionalidad diaria,
    - diferencias entre servicios,
    - ruido,
    - picos operativos.
    """
    rng = np.random.default_rng(42)

    services = [
        "api-login",
        "api-orders",
        "api-payments",
        "api-checkout",
        "frontend-web",
    ]

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=21)

    rows = []

    timestamps = pd.date_range(
        start=start,
        end=now,
        freq="h",
        tz="UTC",
        inclusive="left",
    )

    for service in services:
        base_latency = {
            "api-login": 180,
            "api-orders": 240,
            "api-payments": 320,
            "api-checkout": 380,
            "frontend-web": 150,
        }[service]

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day_of_week = ts.dayofweek

            # Patrón diario: más tráfico en horas centrales
            business_hour_effect = 80 if 9 <= hour <= 18 else 20

            # Patrón semanal: algo más de carga entre semana
            weekday_effect = 60 if day_of_week < 5 else 15

            # Tendencia suave creciente en pagos y checkout
            trend = 0
            if service in ["api-payments", "api-checkout"]:
                trend = i * 0.18

            # Picos artificiales en algunos momentos
            spike = 0
            if service == "api-payments" and i in range(250, 260):
                spike = 700
            if service == "api-checkout" and i in range(390, 398):
                spike = 900

            noise = rng.normal(0, 35)

            latency = max(
                20,
                base_latency
                + business_hour_effect
                + weekday_effect
                + trend
                + spike
                + noise,
            )

            error_rate = max(0, rng.normal(0.015, 0.01))
            request_count = int(max(10, rng.normal(1200, 250)))

            rows.append(
                {
                    "event_ts": ts.to_pydatetime(),
                    "service": service,
                    "latency_p95_ms": round(float(latency), 2),
                    "error_rate": round(float(error_rate), 4),
                    "request_count": request_count,
                }
            )

    return pd.DataFrame(rows)


def create_dataset_if_needed(client: bigquery.Client, project_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{project_id}.{DATASET_ID}")
    dataset_ref.location = LOCATION

    client.create_dataset(dataset_ref, exists_ok=True)
    print(f"Dataset disponible: {project_id}.{DATASET_ID}")


def upload_dataframe(
    client: bigquery.Client,
    project_id: str,
    df: pd.DataFrame,
) -> None:
    table_fqn = f"{project_id}.{DATASET_ID}.{TABLE_ID}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("event_ts", "TIMESTAMP"),
            bigquery.SchemaField("service", "STRING"),
            bigquery.SchemaField("latency_p95_ms", "FLOAT"),
            bigquery.SchemaField("error_rate", "FLOAT"),
            bigquery.SchemaField("request_count", "INTEGER"),
        ],
    )

    job = client.load_table_from_dataframe(
        df,
        table_fqn,
        job_config=job_config,
    )
    job.result()

    print(f"Tabla cargada: {table_fqn}")
    print(f"Filas cargadas: {len(df)}")


def train_arima_model(client: bigquery.Client, project_id: str) -> None:
    model_fqn = f"`{project_id}.{DATASET_ID}.{MODEL_ID}`"
    table_fqn = f"`{project_id}.{DATASET_ID}.{TABLE_ID}`"

    sql = f"""
    CREATE OR REPLACE MODEL {model_fqn}
    OPTIONS(
      MODEL_TYPE = 'ARIMA_PLUS',
      TIME_SERIES_TIMESTAMP_COL = 'event_ts',
      TIME_SERIES_DATA_COL = 'latency_p95_ms',
      TIME_SERIES_ID_COL = 'service',
      DATA_FREQUENCY = 'HOURLY',
      HORIZON = 24,
      AUTO_ARIMA = TRUE,
      CLEAN_SPIKES_AND_DIPS = TRUE,
      ADJUST_STEP_CHANGES = TRUE,
      DECOMPOSE_TIME_SERIES = TRUE
    ) AS
    SELECT
      event_ts,
      service,
      latency_p95_ms
    FROM {table_fqn}
    ORDER BY
      service,
      event_ts
    """

    print("Entrenando modelo ARIMA_PLUS en BigQuery ML...")
    job = client.query(sql, location=LOCATION)
    job.result()
    print(f"Modelo entrenado: {model_fqn}")


def create_forecast_table(client: bigquery.Client, project_id: str) -> None:
    model_fqn = f"`{project_id}.{DATASET_ID}.{MODEL_ID}`"
    forecast_table_fqn = f"`{project_id}.{DATASET_ID}.{FORECAST_TABLE_ID}`"

    sql = f"""
    CREATE OR REPLACE TABLE {forecast_table_fqn} AS
    SELECT
      service,
      forecast_timestamp,
      forecast_value,
      prediction_interval_lower_bound,
      prediction_interval_upper_bound,
      confidence_level
    FROM ML.FORECAST(
      MODEL {model_fqn},
      STRUCT(24 AS horizon, 0.80 AS confidence_level)
    )
    """

    print("Generando forecast de próximas 24 horas...")
    job = client.query(sql, location=LOCATION)
    job.result()
    print(f"Forecast guardado en: {forecast_table_fqn}")

def query_to_dataframe(client: bigquery.Client, sql: str) -> pd.DataFrame:
    job = client.query(sql, location=LOCATION)
    return job.to_dataframe(create_bqstorage_client=False)

def show_operational_risks(client: bigquery.Client, project_id: str) -> None:
    forecast_table_fqn = f"`{project_id}.{DATASET_ID}.{FORECAST_TABLE_ID}`"

    sql = f"""
    SELECT
      service,
      forecast_timestamp,
      ROUND(forecast_value, 2) AS forecast_latency_ms,
      ROUND(prediction_interval_lower_bound, 2) AS lower_bound,
      ROUND(prediction_interval_upper_bound, 2) AS upper_bound,
      CASE
        WHEN forecast_value >= 1000 THEN 'ALTO'
        WHEN prediction_interval_upper_bound >= 1000 THEN 'MEDIO'
        ELSE 'BAJO'
      END AS operational_risk
    FROM {forecast_table_fqn}
    ORDER BY
      operational_risk DESC,
      service,
      forecast_timestamp
    LIMIT 50
    """

    df = query_to_dataframe(client, sql)

    print("\nRiesgo operativo según forecast:")
    print(df.to_string(index=False))


def show_model_evaluation(client: bigquery.Client, project_id: str) -> None:
    model_fqn = f"`{project_id}.{DATASET_ID}.{MODEL_ID}`"

    sql = f"""
    SELECT
      *
    FROM ML.ARIMA_EVALUATE(MODEL {model_fqn})
    """

    df = query_to_dataframe(client, sql)

    # Ordenamos en Python para evitar errores si BigQuery cambia nombres de columnas
    if "service" in df.columns:
        df = df.sort_values("service")
    elif "time_series_id" in df.columns:
        df = df.sort_values("time_series_id")

    print("\nEvaluación ARIMA por serie:")
    print("Columnas devueltas por ML.ARIMA_EVALUATE:")
    print(list(df.columns))
    print()
    print(df.head(20).to_string(index=False))


def main() -> None:
    project_id = get_project_id()
    client = bigquery.Client(project=project_id, location=LOCATION)

    print(f"Proyecto: {project_id}")
    print(f"Location BigQuery: {LOCATION}")

    df = build_synthetic_latency_data()
    print("\nMuestra del dataset:")
    print(df.head().to_string(index=False))

    create_dataset_if_needed(client, project_id)
    upload_dataframe(client, project_id, df)
    train_arima_model(client, project_id)
    create_forecast_table(client, project_id)
    show_operational_risks(client, project_id)
    show_model_evaluation(client, project_id)


if __name__ == "__main__":
    main()