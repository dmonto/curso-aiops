import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud import storage

import re
import uuid

TARGET_COLUMN = "incident_next_30m"
SPLIT_COLUMN = "split"

FEATURE_COLUMNS = [
    "service",
    "environment",
    "region",
    "cpu_avg_5m",
    "memory_avg_5m",
    "latency_p95_5m",
    "error_rate_5m",
    "log_error_count_5m",
    "deploy_last_30m",
    "previous_incidents_24h",
]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value

def slugify(value: str) -> str:
    """
    Normaliza STUDENT_ID para usarlo en rutas, nombres y labels.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")

    if not value:
        raise RuntimeError("STUDENT_ID no puede quedar vacío.")

    return value


def build_run_context() -> tuple[str, str]:
    """
    Crea un identificador estable de alumno y un identificador único de ejecución.
    """
    student_id = slugify(require_env("STUDENT_ID"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]

    run_id = f"{student_id}-{timestamp}-{short_uuid}"

    return student_id, run_id


def build_labels(student_id: str, run_id: str) -> dict:
    """
    Labels comunes para localizar recursos de Vertex AI.
    """
    return {
        "course": "aiops",
        "topic": "custom-training",
        "case": "incident-risk",
        "student_id": student_id[:63],
        "run_id": run_id[:63],
    }

def build_operational_dataset(rows: int = 1500) -> pd.DataFrame:
    services = ["checkout", "payments", "catalog", "search", "api-gateway"]
    environments = ["prod", "prod", "prod", "pre"]
    regions = ["europe-west1", "europe-west1", "europe-southwest1"]

    start = datetime.now(timezone.utc) - timedelta(minutes=rows * 5)
    records = []

    for i in range(rows):
        service = services[i % len(services)]
        environment = environments[i % len(environments)]
        region = regions[i % len(regions)]
        ts = start + timedelta(minutes=i * 5)

        degradation = i % 127 in [101, 102, 103, 104, 105, 106, 107, 108]
        deploy_last_30m = 1 if i % 71 in [0, 1, 2, 3, 4, 5] else 0
        previous_incidents_24h = 1 if i % 210 > 175 else 0

        cpu = 35 + (i % 35) + (38 if degradation else 0)
        memory = 45 + (i % 25) + (27 if degradation else 0)
        latency = 110 + (i % 60) * 5 + (720 if degradation else 0)
        error_rate = 0.01 + ((i % 9) / 1000) + (0.17 if degradation else 0)
        log_errors = 5 + (i % 18) + (180 if degradation else 0)

        incident_next_30m = 1 if (
            environment == "prod"
            and cpu > 82
            and memory > 76
            and latency > 700
            and error_rate > 0.10
        ) else 0

        records.append(
            {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "service": service,
                "environment": environment,
                "region": region,
                "cpu_avg_5m": round(cpu, 2),
                "memory_avg_5m": round(memory, 2),
                "latency_p95_5m": round(latency, 2),
                "error_rate_5m": round(error_rate, 4),
                "log_error_count_5m": int(log_errors),
                "deploy_last_30m": str(deploy_last_30m),
                "previous_incidents_24h": int(previous_incidents_24h),
                TARGET_COLUMN: int(incident_next_30m),
            }
        )

    df = pd.DataFrame(records)
    return add_temporal_split(df)


def add_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    df[SPLIT_COLUMN] = "training"
    df.loc[train_end:validation_end - 1, SPLIT_COLUMN] = "validation"
    df.loc[validation_end:, SPLIT_COLUMN] = "test"

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_distribution": {
            str(k): int(v)
            for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()
        },
        "split_distribution": {
            str(k): int(v)
            for k, v in df[SPLIT_COLUMN].value_counts().to_dict().items()
        },
        "passed": True,
    }

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        report["passed"] = False
        report["missing_columns"] = missing

    if df[TARGET_COLUMN].sum() == 0:
        report["passed"] = False
        report["error"] = "No hay ejemplos positivos."

    return report


def upload_file_to_gcs(local_path: Path, bucket_name: str, destination_blob: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{destination_blob}"


def main() -> None:
    load_dotenv()

    project_id = require_env("PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "europe-west1")
    bucket_name = require_env("BUCKET_NAME")

    run_vertex_training = (
        os.getenv("RUN_VERTEX_CUSTOM_TRAINING", "false").lower() == "true"
    )
    
    machine_type = os.getenv("VERTEX_TRAINING_MACHINE_TYPE", "e2-standard-4")
    training_container = os.getenv(
        "VERTEX_TRAINING_CONTAINER",
        "europe-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    )
    serving_container = os.getenv(
        "VERTEX_SERVING_CONTAINER",
        "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest",
    )

    student_id, run_id = build_run_context()
    labels = build_labels(student_id, run_id)

    output_dir = Path("data") / student_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Región Vertex AI: {location}")
    print(f"Machine type entrenamiento: {machine_type}")
    print(f"Training container: {training_container}")
    print(f"Alumno: {student_id}")
    print(f"Ejecución: {run_id}")

    print("1. Generando dataset operativo...")
    df = build_operational_dataset(rows=1500)

    print("2. Validando dataset...")
    report = validate_dataset(df)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["passed"]:
        raise RuntimeError("El dataset no ha pasado la validación.")

    dataset_path = output_dir / "aiops_custom_training_dataset.csv"
    df.to_csv(dataset_path, index=False)

    print("3. Subiendo dataset a Cloud Storage...")
    dataset_gcs_uri = upload_file_to_gcs(
        dataset_path,
        bucket_name,
        (
            f"students/{student_id}/custom-training/input/"
            f"{run_id}/aiops_custom_training_dataset.csv"
        ),
    )

    print(f"Dataset GCS: {dataset_gcs_uri}")

    trainer_script = "tema2c_trainer.py"
    if not Path(trainer_script).is_file():
        raise RuntimeError("No existe tema2c_trainer.py")

    if not run_vertex_training:
        print("\nRUN_VERTEX_CUSTOM_TRAINING=false")
        print("No se lanza entrenamiento remoto para evitar consumo accidental.")
        print("\nPrueba primero el entrenamiento local con:")
        print(
            "python tema2c_trainer.py "
            f"--train-data-uri .\\{dataset_path} "
            "--model-output-dir .\\data\\local_model"
        )
        print("\nPara entrenar en Vertex AI, cambia en .env:")
        print("RUN_VERTEX_CUSTOM_TRAINING=true")
        return

    print("4. Inicializando Vertex AI...")
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket_name}",
    )

    base_output_dir = (
        f"gs://{bucket_name}/students/{student_id}/"
        f"custom-training/output/{run_id}"
    )

    print("5. Creando CustomTrainingJob...")
    job = aiplatform.CustomTrainingJob(
        display_name=f"aiops-custom-training-{run_id}",
        script_path=str(trainer_script),
        container_uri=training_container,
        requirements=[
            "google-cloud-storage>=2.18.0",
            "joblib>=1.4.2",
        ],
        model_serving_container_image_uri=serving_container,
        labels=labels,
    )

    print("6. Lanzando entrenamiento remoto...")
    model = job.run(
        args=[
            "--train-data-uri",
            dataset_gcs_uri,
            "--target-column",
            TARGET_COLUMN,
            "--split-column",
            SPLIT_COLUMN,
            "--feature-columns",
            ",".join(FEATURE_COLUMNS),
            "--n-estimators",
            "200",
        ],
        replica_count=1,
        machine_type=machine_type,
        base_output_dir=base_output_dir,
        model_display_name=f"aiops-incident-risk-custom-{run_id}",
        model_labels=labels,
        sync=True,
    )

    print("\nEntrenamiento remoto terminado.")
    print(f"Alumno:            {student_id}")
    print(f"Ejecución:         {run_id}")
    print(f"Dataset GCS:       {dataset_gcs_uri}")
    print(f"Artefactos GCS:    {base_output_dir}")
    print(f"Modelo registrado: {model.resource_name}")
    print(f"Modelo visible:    aiops-incident-risk-custom-{run_id}")


if __name__ == "__main__":
    main()