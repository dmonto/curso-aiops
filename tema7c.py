import os
import numpy as np
import pandas as pd

from google.cloud import bigquery
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
STAGING_BUCKET = os.getenv("BUCKET_NAME")

BQ_DATASET = "aiops_lab"
BQ_TABLE = "automl_anomalias_operativas"
MODEL_DISPLAY_NAME = "automl-aiops-anomaly-classifier"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def generar_dataset() -> pd.DataFrame:
    np.random.seed(42)

    services = ["checkout-api", "catalog-api", "auth-api"]
    rows = []

    timestamps = pd.date_range(
        start="2026-05-25 00:00:00",
        periods=7 * 24 * 12,
        freq="5min"
    )

    for service in services:
        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.dayofweek

            business_hours = 9 <= hour <= 18
            base_traffic = 1200 if business_hours else 350

            if service == "auth-api":
                base_traffic *= 1.4
            elif service == "catalog-api":
                base_traffic *= 0.9

            request_count = max(10, np.random.normal(base_traffic, base_traffic * 0.12))
            cpu_percent = 20 + request_count / 35 + np.random.normal(0, 5)
            memory_percent = np.random.normal(55, 7)
            latency_ms = 140 + request_count / 30 + np.random.normal(0, 25)
            error_rate = max(0, np.random.normal(0.004, 0.002))

            incident_type = "NORMAL"

            rows.append({
                "timestamp": ts,
                "service": service,
                "environment": "prod",
                "hour": hour,
                "day_of_week": day_of_week,
                "request_count": request_count,
                "latency_ms": latency_ms,
                "error_rate": error_rate,
                "cpu_percent": min(cpu_percent, 100),
                "memory_percent": min(memory_percent, 100),
                "minutes_since_deployment": np.random.randint(30, 10080),
                "incident_type": incident_type,
            })

    df = pd.DataFrame(rows)

    # Anomalía 1: checkout-api con latencia y errores altos
    mask = (
        (df["service"] == "checkout-api") &
        (df["timestamp"] >= "2026-05-27 10:00:00") &
        (df["timestamp"] <= "2026-05-27 11:30:00")
    )
    df.loc[mask, "latency_ms"] += 350
    df.loc[mask, "error_rate"] += 0.08
    df.loc[mask, "incident_type"] = "LATENCY_DEGRADATION"

    # Anomalía 2: catalog-api con caída de tráfico
    mask = (
        (df["service"] == "catalog-api") &
        (df["timestamp"] >= "2026-05-28 15:00:00") &
        (df["timestamp"] <= "2026-05-28 16:00:00")
    )
    df.loc[mask, "request_count"] *= 0.15
    df.loc[mask, "latency_ms"] += 120
    df.loc[mask, "incident_type"] = "TRAFFIC_DROP"

    # Anomalía 3: auth-api con saturación de CPU y errores
    mask = (
        (df["service"] == "auth-api") &
        (df["timestamp"] >= "2026-05-30 09:30:00") &
        (df["timestamp"] <= "2026-05-30 10:45:00")
    )
    df.loc[mask, "cpu_percent"] = np.random.normal(94, 2, mask.sum())
    df.loc[mask, "memory_percent"] = np.random.normal(88, 3, mask.sum())
    df.loc[mask, "error_rate"] += 0.05
    df.loc[mask, "incident_type"] = "CPU_SATURATION"

    df["is_anomaly"] = (df["incident_type"] != "NORMAL").astype(int)

    return df


def crear_dataset_bigquery(project_id: str, dataset_id: str) -> None:
    client = bigquery.Client(project=project_id)

    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset_ref.location = "US"

    try:
        client.create_dataset(dataset_ref, exists_ok=True)
        print(f"Dataset BigQuery disponible: {project_id}.{dataset_id}")
    except Exception as ex:
        raise RuntimeError(f"No se pudo crear/verificar el dataset de BigQuery: {ex}") from ex


def subir_a_bigquery(df: pd.DataFrame, project_id: str) -> str:
    crear_dataset_bigquery(project_id, BQ_DATASET)

    table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True
    )

    client = bigquery.Client(project=project_id)
    load_job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=job_config
    )
    load_job.result()

    print(f"Tabla BigQuery cargada: {table_id}")
    return f"bq://{table_id}"


def crear_dataset_vertex_ai(bq_uri: str) -> aiplatform.TabularDataset:
    dataset = aiplatform.TabularDataset.create(
        display_name="aiops-automl-anomaly-dataset",
        bq_source=bq_uri,
    )

    print(f"Dataset Vertex AI creado: {dataset.resource_name}")
    return dataset


def entrenar_automl_tabular(dataset: aiplatform.TabularDataset) -> aiplatform.Model:
    """
    Entrena un modelo AutoML Tabular de clasificación binaria.

    Nota:
    - Puede tardar bastante.
    - Consume presupuesto del proyecto.
    - Ejecutar solo en un proyecto preparado para laboratorio.
    """
    training_job = aiplatform.AutoMLTabularTrainingJob(
        display_name="aiops-automl-anomaly-training",
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-prc",
    )

    model = training_job.run(
        dataset=dataset,
        target_column="is_anomaly",
        budget_milli_node_hours=1000,
        model_display_name=MODEL_DISPLAY_NAME,
        disable_early_stopping=False,
        sync=True,
    )

    print(f"Modelo entrenado: {model.resource_name}")
    return model


def main() -> None:
    project_id = require_env("PROJECT_ID")
    require_env("BUCKET_NAME")

    aiplatform.init(
        project=project_id,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET
    )

    df = generar_dataset()

    print("Resumen del dataset")
    print("-------------------")
    print(df["incident_type"].value_counts())
    print()
    print(df.head())

    df.to_csv("automl_anomalias_operativas.csv", index=False)
    print("\nCSV local generado: automl_anomalias_operativas.csv")

    bq_uri = subir_a_bigquery(df, project_id)
    dataset = crear_dataset_vertex_ai(bq_uri)

    print("\nDataset preparado para AutoML.")
    print("Para entrenar el modelo real, descomenta la siguiente línea en el script:")
    print("# model = entrenar_automl_tabular(dataset)")

    # Descomentar solo cuando quieras lanzar entrenamiento real:
    # model = entrenar_automl_tabular(dataset)


if __name__ == "__main__":
    main()