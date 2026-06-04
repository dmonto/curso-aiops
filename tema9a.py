import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno obligatoria: {name}")
    return value


def ensure_dataset(client: bigquery.Client, project_id: str, dataset_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset_ref.location = os.getenv("AIOPS_BQ_LOCATION", "EU")
        client.create_dataset(dataset_ref)
        print(f"Dataset creado: {project_id}.{dataset_id}")


def generate_synthetic_requests(
    service_name: str = "checkout-api",
    days: int = 7,
    rows_per_day: int = 1500,
) -> pd.DataFrame:
    """
    Genera eventos sintéticos de peticiones HTTP.

    Se introduce degradación artificial en los dos últimos días para que
    el cálculo de SLO muestre escenarios OK, WARNING y VIOLATION.
    """

    now = datetime.now(timezone.utc)
    rows: List[Dict] = []

    for day_offset in range(days):
        day = now - timedelta(days=days - day_offset - 1)

        for _ in range(rows_per_day):
            event_ts = day.replace(
                hour=random.randint(0, 23),
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
                microsecond=0,
            )

            # Días normales: pocos errores y latencia razonable.
            error_probability = 0.004
            slow_probability = 0.045

            # Últimos días: degradación simulada.
            if day_offset == days - 2:
                error_probability = 0.012
                slow_probability = 0.080

            if day_offset == days - 1:
                error_probability = 0.030
                slow_probability = 0.160

            is_error = random.random() < error_probability
            is_slow = random.random() < slow_probability

            if is_error:
                status_code = random.choice([500, 502, 503, 504])
            else:
                status_code = random.choice([200, 200, 200, 201, 204, 400, 401, 404])

            if is_slow:
                latency_ms = random.randint(600, 2500)
            else:
                latency_ms = max(20, int(random.gauss(220, 80)))

            rows.append(
                {
                    "event_ts": event_ts.isoformat(),
                    "service_name": service_name,
                    "route": random.choice(
                        ["/checkout", "/cart", "/payment", "/shipping"]
                    ),
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "region": random.choice(["europe-west1", "europe-west4"]),
                    "release_version": random.choice(["1.4.5", "1.4.6", "1.4.7"]),
                }
            )

    return pd.DataFrame(rows)


def load_requests_to_bigquery(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
    df: pd.DataFrame,
) -> None:
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    df = df.copy()

    # BigQuery espera TIMESTAMP, no object/string.
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)

    df["service_name"] = df["service_name"].astype(str)
    df["route"] = df["route"].astype(str)
    df["status_code"] = df["status_code"].astype("int64")
    df["latency_ms"] = df["latency_ms"].astype("int64")
    df["region"] = df["region"].astype(str)
    df["release_version"] = df["release_version"].astype(str)

    print("\nTipos antes de cargar a BigQuery:")
    print(df.dtypes)

    schema = [
        bigquery.SchemaField("event_ts", "TIMESTAMP"),
        bigquery.SchemaField("service_name", "STRING"),
        bigquery.SchemaField("route", "STRING"),
        bigquery.SchemaField("status_code", "INTEGER"),
        bigquery.SchemaField("latency_ms", "INTEGER"),
        bigquery.SchemaField("region", "STRING"),
        bigquery.SchemaField("release_version", "STRING"),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    print(f"Eventos cargados en BigQuery: {table_ref}")
    print(f"Filas: {len(df)}")


def calculate_daily_slo(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    request_table: str,
    slo_table: str,
) -> pd.DataFrame:
    source = f"`{project_id}.{dataset_id}.{request_table}`"
    target = f"`{project_id}.{dataset_id}.{slo_table}`"

    query = f"""
    CREATE OR REPLACE TABLE {target} AS
    WITH base AS (
      SELECT
        DATE(event_ts) AS event_date,
        service_name,
        COUNT(*) AS total_requests,

        COUNTIF(status_code < 500) AS availability_good_requests,
        COUNTIF(latency_ms <= 500) AS latency_good_requests,
        COUNTIF(status_code < 500 AND latency_ms <= 500) AS combined_good_requests,

        COUNTIF(status_code >= 500) AS server_errors,
        COUNTIF(latency_ms > 500) AS slow_requests,
        APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)] AS latency_p95_ms,
        APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)] AS latency_p99_ms
      FROM {source}
      GROUP BY event_date, service_name
    ),
    calculated AS (
      SELECT
        *,
        SAFE_DIVIDE(availability_good_requests, total_requests) AS availability_sli,
        SAFE_DIVIDE(latency_good_requests, total_requests) AS latency_sli,
        SAFE_DIVIDE(combined_good_requests, total_requests) AS combined_sli,

        0.995 AS availability_slo,
        0.950 AS latency_slo,
        0.990 AS combined_slo,

        total_requests * (1 - 0.990) AS combined_error_budget_allowed,
        total_requests - combined_good_requests AS combined_bad_events
      FROM base
    )
    SELECT
      *,
      SAFE_DIVIDE(combined_bad_events, combined_error_budget_allowed) AS combined_budget_consumed_ratio,

      CASE
        WHEN combined_sli < combined_slo THEN 'VIOLATION'
        WHEN SAFE_DIVIDE(combined_bad_events, combined_error_budget_allowed) >= 0.75 THEN 'WARNING'
        ELSE 'OK'
      END AS slo_status
    FROM calculated
    ORDER BY event_date;
    """

    client.query(query).result()

    result_query = f"""
    SELECT
      event_date,
      service_name,
      total_requests,
      ROUND(availability_sli * 100, 3) AS availability_sli_pct,
      ROUND(latency_sli * 100, 3) AS latency_sli_pct,
      ROUND(combined_sli * 100, 3) AS combined_sli_pct,
      latency_p95_ms,
      latency_p99_ms,
      server_errors,
      slow_requests,
      ROUND(combined_budget_consumed_ratio * 100, 2) AS combined_budget_consumed_pct,
      slo_status
    FROM {target}
    ORDER BY event_date;
    """

    job = client.query(result_query)
    return job.to_dataframe(create_bqstorage_client=False)


def main() -> int:
    try:
        project_id = require_env("PROJECT_ID")
        dataset_id = os.getenv("AIOPS_DATASET", "aiops_lab")
        request_table = os.getenv("AIOPS_REQUEST_TABLE", "service_requests")
        slo_table = os.getenv("AIOPS_SLO_TABLE", "slo_daily")

        client = bigquery.Client(project=project_id)

        ensure_dataset(client, project_id, dataset_id)

        df = generate_synthetic_requests()
        load_requests_to_bigquery(client, project_id, dataset_id, request_table, df)

        slo_df = calculate_daily_slo(
            client=client,
            project_id=project_id,
            dataset_id=dataset_id,
            request_table=request_table,
            slo_table=slo_table,
        )

        print("\nResultado diario de SLI/SLO")
        print(slo_df.to_string(index=False))

        print("\nInterpretación rápida:")
        print("- OK: el servicio cumple el SLO combinado.")
        print("- WARNING: todavía puede cumplir, pero consume demasiado error budget.")
        print("- VIOLATION: el SLO combinado se ha incumplido.")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())