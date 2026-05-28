import os
from datetime import datetime, timedelta, timezone

import google.auth
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import aiplatform


DATASET_ID = "curso_aiops"
FEATURE_TABLE_ID = "prod_aiops_features"
SCORE_TABLE_ID = "prod_aiops_risk_scores"
MODEL_ID = "prod_failure_model_bqml"
VERTEX_MODEL_ID = "prod_failure_model_aiops"
ENDPOINT_DISPLAY_NAME = "endpoint-aiops-failure-prod"

BQ_LOCATION = os.getenv("BIGQUERY_LOCATION", "EU")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "europe-west1")
DEPLOY_TO_VERTEX_ENDPOINT = os.getenv("DEPLOY_TO_VERTEX_ENDPOINT", "false").lower() == "true"

RANDOM_SEED = 42


def get_project_id() -> str:
    project_id = os.getenv("GCP_PROJECT_ID")
    if project_id:
        return project_id

    _, inferred_project = google.auth.default()
    if not inferred_project:
        raise RuntimeError("No se ha podido inferir el proyecto. Define GCP_PROJECT_ID.")

    return inferred_project


def build_features_dataset(n_rows: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    services = [
        "api-login",
        "api-orders",
        "api-payments",
        "api-checkout",
        "worker-billing",
        "frontend-web",
    ]

    regions = ["europe-west1", "europe-southwest1"]
    environments = ["prod", "test"]

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(
        start=now - timedelta(days=20),
        periods=n_rows,
        freq="15min",
        tz="UTC",
    )

    df = pd.DataFrame({
        "feature_ts": timestamps,
        "service": rng.choice(services, size=n_rows),
        "region": rng.choice(regions, size=n_rows),
        "environment": rng.choice(environments, size=n_rows, p=[0.75, 0.25]),
        "business_hour": rng.choice([0, 1], size=n_rows, p=[0.55, 0.45]),
        "service_criticality_score": rng.choice([1, 2, 3], size=n_rows, p=[0.20, 0.50, 0.30]),
        "cpu_avg_15m": rng.normal(55, 17, size=n_rows).clip(1, 100),
        "cpu_max_15m": rng.normal(72, 17, size=n_rows).clip(1, 100),
        "memory_avg_15m": rng.normal(58, 14, size=n_rows).clip(1, 100),
        "latency_p95_15m": rng.gamma(2.1, 230, size=n_rows).clip(20, 5000),
        "latency_trend_15m": rng.normal(50, 180, size=n_rows),
        "error_rate_avg_15m": rng.beta(1.2, 22, size=n_rows),
        "queue_depth_max_15m": rng.gamma(1.8, 130, size=n_rows).clip(0, 4000),
        "queue_trend_15m": rng.normal(20, 140, size=n_rows),
        "dependency_errors_sum_15m": rng.poisson(0.8, size=n_rows),
        "deploy_recent_30m": rng.choice([0, 1], size=n_rows, p=[0.84, 0.16]),
    })

    prod = df["environment"].eq("prod").astype(int)
    critical = df["service_criticality_score"]

    risk_score = (
        0.020 * df["cpu_avg_15m"]
        + 0.025 * df["cpu_max_15m"]
        + 0.0022 * df["latency_p95_15m"]
        + 0.0050 * df["latency_trend_15m"].clip(lower=0)
        + 8.0 * df["error_rate_avg_15m"]
        + 0.0015 * df["queue_depth_max_15m"]
        + 0.0040 * df["queue_trend_15m"].clip(lower=0)
        + 0.65 * df["dependency_errors_sum_15m"]
        + 1.00 * df["deploy_recent_30m"]
        + 0.65 * prod
        + 0.45 * critical
        + rng.normal(0, 1.3, size=n_rows)
    )

    probability = 1 / (1 + np.exp(-(risk_score - 8.4)))
    df["failure_within_30m"] = rng.binomial(1, probability)

    return df


def create_dataset(client: bigquery.Client, project_id: str) -> None:
    dataset = bigquery.Dataset(f"{project_id}.{DATASET_ID}")
    dataset.location = BQ_LOCATION
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset disponible: {project_id}.{DATASET_ID}")


def upload_features(client: bigquery.Client, project_id: str, df: pd.DataFrame) -> None:
    table_fqn = f"{project_id}.{DATASET_ID}.{FEATURE_TABLE_ID}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("feature_ts", "TIMESTAMP"),
            bigquery.SchemaField("service", "STRING"),
            bigquery.SchemaField("region", "STRING"),
            bigquery.SchemaField("environment", "STRING"),
            bigquery.SchemaField("business_hour", "INTEGER"),
            bigquery.SchemaField("service_criticality_score", "INTEGER"),
            bigquery.SchemaField("cpu_avg_15m", "FLOAT"),
            bigquery.SchemaField("cpu_max_15m", "FLOAT"),
            bigquery.SchemaField("memory_avg_15m", "FLOAT"),
            bigquery.SchemaField("latency_p95_15m", "FLOAT"),
            bigquery.SchemaField("latency_trend_15m", "FLOAT"),
            bigquery.SchemaField("error_rate_avg_15m", "FLOAT"),
            bigquery.SchemaField("queue_depth_max_15m", "FLOAT"),
            bigquery.SchemaField("queue_trend_15m", "FLOAT"),
            bigquery.SchemaField("dependency_errors_sum_15m", "INTEGER"),
            bigquery.SchemaField("deploy_recent_30m", "INTEGER"),
            bigquery.SchemaField("failure_within_30m", "INTEGER"),
        ],
    )

    job = client.load_table_from_dataframe(df, table_fqn, job_config=job_config)
    job.result()

    print(f"Features cargadas en: {table_fqn}")
    print(f"Filas: {len(df)}")

def run_query(client: bigquery.Client, sql: str) -> pd.DataFrame:
    job = client.query(sql, location=BQ_LOCATION)
    return job.to_dataframe(create_bqstorage_client=False)


def train_and_register_bqml_model(client: bigquery.Client, project_id: str) -> None:
    model_fqn = f"`{project_id}.{DATASET_ID}.{MODEL_ID}`"
    table_fqn = f"`{project_id}.{DATASET_ID}.{FEATURE_TABLE_ID}`"

    sql = f"""
    CREATE OR REPLACE MODEL {model_fqn}
    OPTIONS(
      MODEL_TYPE = 'LOGISTIC_REG',
      INPUT_LABEL_COLS = ['failure_within_30m'],
      AUTO_CLASS_WEIGHTS = TRUE,
      MAX_ITERATIONS = 30,
      EARLY_STOP = TRUE,
      MODEL_REGISTRY = 'VERTEX_AI',
      VERTEX_AI_MODEL_ID = '{VERTEX_MODEL_ID}',
      VERTEX_AI_MODEL_VERSION_ALIASES = ['candidate']
    ) AS
    SELECT
      service,
      region,
      environment,
      business_hour,
      service_criticality_score,
      cpu_avg_15m,
      cpu_max_15m,
      memory_avg_15m,
      latency_p95_15m,
      latency_trend_15m,
      error_rate_avg_15m,
      queue_depth_max_15m,
      queue_trend_15m,
      dependency_errors_sum_15m,
      deploy_recent_30m,
      failure_within_30m
    FROM {table_fqn}
    WHERE feature_ts < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
    """

    print("Entrenando y registrando modelo BigQuery ML en Vertex AI Model Registry...")
    client.query(sql, location=BQ_LOCATION).result()
    print(f"Modelo BQML creado: {model_fqn}")
    print(f"Vertex AI model id: {VERTEX_MODEL_ID}")


def evaluate_model(client: bigquery.Client, project_id: str) -> pd.DataFrame:
    model_fqn = f"`{project_id}.{DATASET_ID}.{MODEL_ID}`"
    table_fqn = f"`{project_id}.{DATASET_ID}.{FEATURE_TABLE_ID}`"

    sql = f"""
    SELECT
      *
    FROM ML.EVALUATE(
      MODEL {model_fqn},
      (
        SELECT
          service,
          region,
          environment,
          business_hour,
          service_criticality_score,
          cpu_avg_15m,
          cpu_max_15m,
          memory_avg_15m,
          latency_p95_15m,
          latency_trend_15m,
          error_rate_avg_15m,
          queue_depth_max_15m,
          queue_trend_15m,
          dependency_errors_sum_15m,
          deploy_recent_30m,
          failure_within_30m
        FROM {table_fqn}
        WHERE feature_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
      )
    )
    """

    df = run_query(client, sql)
    print("\nEvaluación del modelo candidato:")
    print(df.to_string(index=False))
    return df


def create_risk_scores_table(client: bigquery.Client, project_id: str, threshold: float = 0.70) -> None:
    model_fqn = f"`{project_id}.{DATASET_ID}.{MODEL_ID}`"
    table_fqn = f"`{project_id}.{DATASET_ID}.{FEATURE_TABLE_ID}`"
    score_fqn = f"`{project_id}.{DATASET_ID}.{SCORE_TABLE_ID}`"

    sql = f"""
    CREATE OR REPLACE TABLE {score_fqn} AS
    WITH predictions AS (
      SELECT
        feature_ts,
        service,
        region,
        environment,
        failure_within_30m,
        predicted_failure_within_30m,
        (
          SELECT prob
          FROM UNNEST(predicted_failure_within_30m_probs)
          WHERE label = 1
        ) AS failure_probability
      FROM ML.PREDICT(
        MODEL {model_fqn},
        (
          SELECT
            feature_ts,
            service,
            region,
            environment,
            business_hour,
            service_criticality_score,
            cpu_avg_15m,
            cpu_max_15m,
            memory_avg_15m,
            latency_p95_15m,
            latency_trend_15m,
            error_rate_avg_15m,
            queue_depth_max_15m,
            queue_trend_15m,
            dependency_errors_sum_15m,
            deploy_recent_30m,
            failure_within_30m
          FROM {table_fqn}
          WHERE feature_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
        )
      )
    )
    SELECT
      CURRENT_TIMESTAMP() AS scored_at,
      feature_ts,
      service,
      region,
      environment,
      failure_probability,
      CASE
        WHEN failure_probability >= 0.85 THEN 'CRITICO'
        WHEN failure_probability >= {threshold} THEN 'ALTO'
        WHEN failure_probability >= 0.45 THEN 'MEDIO'
        ELSE 'BAJO'
      END AS risk_level,
      CASE
        WHEN failure_probability >= 0.85 THEN 'abrir incidente preventivo'
        WHEN failure_probability >= {threshold} THEN 'notificar a SRE'
        WHEN failure_probability >= 0.45 THEN 'mostrar en dashboard'
        ELSE 'solo registrar'
      END AS recommended_action,
      '{VERTEX_MODEL_ID}' AS model_version,
      failure_within_30m
    FROM predictions
    """

    client.query(sql, location=BQ_LOCATION).result()
    print(f"\nTabla de scores creada: {score_fqn}")


def show_high_risk_services(client: bigquery.Client, project_id: str) -> None:
    score_fqn = f"`{project_id}.{DATASET_ID}.{SCORE_TABLE_ID}`"

    sql = f"""
    SELECT
      feature_ts,
      service,
      region,
      environment,
      ROUND(failure_probability, 4) AS failure_probability,
      risk_level,
      recommended_action
    FROM {score_fqn}
    WHERE risk_level IN ('ALTO', 'CRITICO')
    ORDER BY failure_probability DESC
    LIMIT 20
    """

    df = run_query(client, sql)

    print("\nServicios con riesgo alto o crítico:")
    if df.empty:
        print("No hay servicios en riesgo alto o crítico.")
    else:
        print(df.to_string(index=False))


def deploy_registered_model_to_endpoint(project_id: str) -> None:
    """
    Despliegue opcional en Vertex AI Endpoint.
    Mantener DEPLOY_TO_VERTEX_ENDPOINT=false en laboratorio si no se quiere consumir recursos.
    """
    print("\nBuscando modelo registrado en Vertex AI Model Registry...")

    aiplatform.init(project=project_id, location=VERTEX_LOCATION)

    models = aiplatform.Model.list(
        filter=f'display_name="{VERTEX_MODEL_ID}"',
        order_by="create_time desc",
    )

    if not models:
        print("No se ha encontrado el modelo en Vertex AI Model Registry.")
        print("Revisa que el registro desde BigQuery ML haya finalizado correctamente.")
        return

    model = models[0]
    print(f"Modelo encontrado: {model.resource_name}")

    print("Creando endpoint...")
    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        project=project_id,
        location=VERTEX_LOCATION,
    )

    print(f"Endpoint creado: {endpoint.resource_name}")

    print("Desplegando modelo en endpoint...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{VERTEX_MODEL_ID}-deployed",
        machine_type="n1-standard-2",
        traffic_percentage=100,
        sync=True,
    )

    print("Modelo desplegado correctamente.")
    print(f"Endpoint resource name: {endpoint.resource_name}")


def main() -> None:
    project_id = get_project_id()

    bq_client = bigquery.Client(project=project_id, location=BQ_LOCATION)

    print(f"Proyecto: {project_id}")
    print(f"BigQuery location: {BQ_LOCATION}")
    print(f"Vertex location: {VERTEX_LOCATION}")
    print(f"Desplegar endpoint Vertex AI: {DEPLOY_TO_VERTEX_ENDPOINT}")

    df = build_features_dataset()

    create_dataset(bq_client, project_id)
    upload_features(bq_client, project_id, df)
    train_and_register_bqml_model(bq_client, project_id)
    evaluate_model(bq_client, project_id)
    create_risk_scores_table(bq_client, project_id, threshold=0.70)
    show_high_risk_services(bq_client, project_id)

    if DEPLOY_TO_VERTEX_ENDPOINT:
        deploy_registered_model_to_endpoint(project_id)
    else:
        print("\nDEPLOY_TO_VERTEX_ENDPOINT=false.")
        print("Se ha creado el modelo, se ha registrado y se ha generado la tabla de scores.")
        print("Activa DEPLOY_TO_VERTEX_ENDPOINT=true solo si quieres crear un endpoint real.")


if __name__ == "__main__":
    main()