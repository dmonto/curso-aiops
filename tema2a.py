import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud import storage
import re
import uuid

EXPECTED_COLUMNS = [
    "timestamp",
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
    "incident_next_30m",
]

TARGET_COLUMN = "incident_next_30m"

POTENTIAL_LEAKAGE_COLUMNS = [
    "incident_id",
    "root_cause",
    "resolution_time_minutes",
    "postmortem_summary",
    "ticket_closed_at",
]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value

def slugify(value: str) -> str:
    """
    Convierte un identificador en algo seguro para rutas, nombres de dataset
    y prefijos de Cloud Storage.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")

    if not value:
        raise RuntimeError("STUDENT_ID no puede quedar vacío después de normalizarlo.")

    return value


def build_run_context() -> tuple[str, str]:
    """
    Crea un identificador único de ejecución.

    Devuelve:
    - student_id: identificador estable del alumno
    - run_id: identificador único para esta ejecución
    """
    student_id = slugify(require_env("STUDENT_ID"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]

    run_id = f"{student_id}-{timestamp}-{short_uuid}"

    return student_id, run_id

def build_aiops_dataset(rows: int = 600) -> pd.DataFrame:
    """
    Genera un dataset sintético con señales operativas.
    Cada fila representa una ventana de 5 minutos por servicio.
    """
    services = ["checkout", "payments", "catalog", "search"]
    environments = ["prod", "prod", "prod", "pre"]
    regions = ["europe-west1", "europe-west1", "europe-southwest1"]

    start = datetime.now(timezone.utc) - timedelta(minutes=rows * 5)
    records = []

    for i in range(rows):
        service = services[i % len(services)]
        environment = environments[i % len(environments)]
        region = regions[i % len(regions)]
        ts = start + timedelta(minutes=i * 5)

        degradation = i % 97 in [81, 82, 83, 84, 85, 86]
        deploy_last_30m = 1 if i % 53 in [0, 1, 2, 3, 4, 5] else 0

        cpu = 33 + (i % 33) + (34 if degradation else 0)
        memory = 50 + (i % 25) + (20 if degradation else 0)
        latency = 120 + (i % 60) * 5 + (650 if degradation else 0)
        error_rate = 0.01 + ((i % 7) / 1000) + (0.14 if degradation else 0)
        log_errors = 5 + (i % 15) + (140 if degradation else 0)
        previous_incidents = 1 if i % 120 > 95 else 0

        incident_next_30m = 1 if (
            environment == "prod"
            and cpu > 80
            and latency > 600
            and error_rate > 0.08
        ) else 0

        records.append(
            {
                "timestamp": ts.isoformat(),
                "service": service,
                "environment": environment,
                "region": region,
                "cpu_avg_5m": round(cpu, 2),
                "memory_avg_5m": round(memory, 2),
                "latency_p95_5m": round(latency, 2),
                "error_rate_5m": round(error_rate, 4),
                "log_error_count_5m": int(log_errors),
                "deploy_last_30m": int(deploy_last_30m),
                "previous_incidents_24h": int(previous_incidents),
                "incident_next_30m": int(incident_next_30m),
            }
        )

    return pd.DataFrame(records)


def validate_dataset(df: pd.DataFrame) -> Dict:
    report = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_expected_columns": [],
        "unexpected_columns": [],
        "nulls_by_column": {},
        "duplicate_service_timestamp_rows": 0,
        "target_distribution": {},
        "potential_leakage_columns_found": [],
        "range_warnings": [],
        "passed": True,
    }

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    unexpected = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    report["missing_expected_columns"] = missing
    report["unexpected_columns"] = unexpected

    if missing:
        report["passed"] = False

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            report["nulls_by_column"][col] = null_count

    if report["nulls_by_column"]:
        report["passed"] = False

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        invalid_ts = int(df["timestamp"].isna().sum())
        if invalid_ts > 0:
            report["range_warnings"].append(
                f"Hay {invalid_ts} timestamps no interpretables."
            )
            report["passed"] = False

    if {"service", "timestamp"}.issubset(df.columns):
        dup_count = int(df.duplicated(subset=["service", "timestamp"]).sum())
        report["duplicate_service_timestamp_rows"] = dup_count
        if dup_count > 0:
            report["passed"] = False

    if TARGET_COLUMN in df.columns:
        distribution = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        report["target_distribution"] = {
            str(k): int(v) for k, v in distribution.items()
        }

        positive_rate = float(df[TARGET_COLUMN].mean())
        if positive_rate < 0.01:
            report["range_warnings"].append(
                "El target positivo es muy bajo. Puede haber problema de desbalance."
            )
        if positive_rate > 0.80:
            report["range_warnings"].append(
                "El target positivo es muy alto. Revisa la definición de incidente."
            )
    else:
        report["passed"] = False

    found_leakage = [
        col for col in POTENTIAL_LEAKAGE_COLUMNS if col in df.columns
    ]
    report["potential_leakage_columns_found"] = found_leakage

    if found_leakage:
        report["passed"] = False

    numeric_ranges = {
        "cpu_avg_5m": (0, 100),
        "memory_avg_5m": (0, 100),
        "error_rate_5m": (0, 1),
    }

    for col, (min_value, max_value) in numeric_ranges.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_value) | (df[col] > max_value)]
            if len(out_of_range) > 0:
                report["range_warnings"].append(
                    f"{col} tiene {len(out_of_range)} valores fuera de rango."
                )
                report["passed"] = False

    return report


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    validation_df = df_sorted.iloc[train_end:validation_end].copy()
    test_df = df_sorted.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def save_artifacts(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: Dict,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "full": output_dir / "aiops_dataset_full.csv",
        "train": output_dir / "aiops_dataset_train.csv",
        "validation": output_dir / "aiops_dataset_validation.csv",
        "test": output_dir / "aiops_dataset_test.csv",
        "quality_report": output_dir / "aiops_dataset_quality_report.json",
    }

    df.to_csv(paths["full"], index=False)
    train_df.to_csv(paths["train"], index=False)
    validation_df.to_csv(paths["validation"], index=False)
    test_df.to_csv(paths["test"], index=False)

    with open(paths["quality_report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return paths


def upload_file_to_gcs(local_path: Path, bucket_name: str, destination_blob: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{destination_blob}"


def upload_artifacts(
    paths: Dict[str, Path],
    bucket_name: str,
    student_id: str,
    run_id: str,
) -> Dict[str, str]:
    gcs_uris = {}

    for artifact_name, local_path in paths.items():
        destination_blob = (
            f"students/{student_id}/datasets/aiops/{run_id}/{local_path.name}"
        )

        gcs_uris[artifact_name] = upload_file_to_gcs(
            local_path=local_path,
            bucket_name=bucket_name,
            destination_blob=destination_blob,
        )

    return gcs_uris


def create_vertex_tabular_dataset(
    project_id: str,
    location: str,
    bucket_name: str,
    display_name: str,
    gcs_uri: str,
):
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket_name}",
    )

    print(f"Creando Vertex AI TabularDataset: {display_name}")

    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=[gcs_uri],
    )

    return dataset


def main() -> None:
    load_dotenv()

    project_id = require_env("PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "europe-west1")
    bucket_name = require_env("BUCKET_NAME")
    display_name = os.getenv("VERTEX_DATASET_DISPLAY_NAME", "aiops-incident-dataset")
    create_vertex_dataset = os.getenv("CREATE_VERTEX_DATASET", "true").lower() == "true"

    student_id, run_id = build_run_context()

    output_dir = Path("data") / student_id / run_id

    print(f"Alumno: {student_id}")
    print(f"Ejecución: {run_id}")

    print("1. Generando dataset sintético...")
    df = build_aiops_dataset(rows=600)

    print("2. Validando calidad del dataset...")
    report = validate_dataset(df.copy())

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["passed"]:
        raise RuntimeError(
            "El dataset no ha pasado la validación. Revisa el informe antes de continuar."
        )

    print("3. Creando split temporal...")
    train_df, validation_df, test_df = temporal_split(df)

    print(f"Train:      {len(train_df)} filas")
    print(f"Validation: {len(validation_df)} filas")
    print(f"Test:       {len(test_df)} filas")

    print("4. Guardando artefactos locales...")
    paths = save_artifacts(
        df=df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        report=report,
        output_dir=output_dir,
    )

    print("5. Subiendo artefactos a Cloud Storage...")
    gcs_uris = upload_artifacts(
        paths=paths,
        bucket_name=bucket_name,
        student_id=student_id,
        run_id=run_id,
)

    for name, uri in gcs_uris.items():
        print(f"{name}: {uri}")

    if create_vertex_dataset:
        print("6. Creando dataset gestionado en Vertex AI...")
        dataset = create_vertex_tabular_dataset(
            project_id=project_id,
            location=location,
            bucket_name=bucket_name,
            display_name=f"{display_name}-{run_id}",
            gcs_uri=gcs_uris["full"],
        )

        print("Dataset creado en Vertex AI")
        print(f"Nombre recurso: {dataset.resource_name}")
    else:
        print("6. CREATE_VERTEX_DATASET=false. Se omite creación en Vertex AI.")

    print("\nResumen final")
    print("-------------")
    print(f"Alumno:          {student_id}")
    print(f"Ejecución:       {run_id}")
    print(f"Target:          {TARGET_COLUMN}")
    print(f"GCS full:        {gcs_uris['full']}")
    print(f"Quality report:  {gcs_uris['quality_report']}")


if __name__ == "__main__":
    main()