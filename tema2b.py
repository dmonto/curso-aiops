import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud import storage
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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

COLUMN_SPECS = {
    "service": "categorical",
    "environment": "categorical",
    "region": "categorical",
    "cpu_avg_5m": "numeric",
    "memory_avg_5m": "numeric",
    "latency_p95_5m": "numeric",
    "error_rate_5m": "numeric",
    "log_error_count_5m": "numeric",
    "deploy_last_30m": "categorical",
    "previous_incidents_24h": "numeric",
}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value


def build_operational_dataset(rows: int = 1200) -> pd.DataFrame:
    """
    Dataset sintético AIOps.
    Cada fila representa una ventana de 5 minutos para un servicio.
    """
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

        # Patrón sintético de degradación.
        degradation = i % 113 in [91, 92, 93, 94, 95, 96, 97]
        deploy_last_30m = 1 if i % 67 in [0, 1, 2, 3, 4, 5] else 0
        previous_incidents_24h = 1 if i % 180 > 150 else 0

        cpu = 35 + (i % 35) + (35 if degradation else 0)
        memory = 45 + (i % 25) + (25 if degradation else 0)
        latency = 110 + (i % 60) * 5 + (700 if degradation else 0)
        error_rate = 0.01 + ((i % 9) / 1000) + (0.16 if degradation else 0)
        log_errors = 5 + (i % 18) + (160 if degradation else 0)

        incident_next_30m = 1 if (
            environment == "prod"
            and cpu > 80
            and memory > 75
            and latency > 650
            and error_rate > 0.09
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


def validate_for_automl(df: pd.DataFrame) -> Dict:
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_feature_columns": [],
        "target_distribution": {},
        "split_distribution": {},
        "nulls": {},
        "passed": True,
    }

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN, "timestamp"]
    missing = [col for col in expected_columns if col not in df.columns]
    report["missing_feature_columns"] = missing

    if missing:
        report["passed"] = False

    for col in expected_columns:
        if col in df.columns:
            nulls = int(df[col].isna().sum())
            if nulls > 0:
                report["nulls"][col] = nulls

    if report["nulls"]:
        report["passed"] = False

    if TARGET_COLUMN in df.columns:
        report["target_distribution"] = {
            str(k): int(v)
            for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()
        }

    if SPLIT_COLUMN in df.columns:
        report["split_distribution"] = {
            str(k): int(v)
            for k, v in df[SPLIT_COLUMN].value_counts().to_dict().items()
        }

        valid_splits = {"training", "validation", "test"}
        actual_splits = set(df[SPLIT_COLUMN].unique())
        if not actual_splits.issubset(valid_splits):
            report["passed"] = False

    positive_rate = float(df[TARGET_COLUMN].mean())
    if positive_rate == 0:
        report["passed"] = False
        report["warning"] = "No hay ejemplos positivos. No se puede entrenar clasificación útil."
    elif positive_rate < 0.01:
        report["warning"] = "Clase positiva muy baja. Revisa AU PRC, precision y recall."

    return report


def run_local_baseline(df: pd.DataFrame) -> Dict:
    """
    Baseline local rápido para comparar contra AutoML.
    No sustituye AutoML, pero ayuda a validar si el dataset tiene señal predictiva.
    """
    model_df = df[FEATURE_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN]].copy()

    # One-hot encoding simple para columnas categóricas.
    model_df = pd.get_dummies(
        model_df,
        columns=["service", "environment", "region", "deploy_last_30m"],
        drop_first=False,
    )

    train_df = model_df[model_df[SPLIT_COLUMN] == "training"].drop(columns=[SPLIT_COLUMN])
    test_df = model_df[model_df[SPLIT_COLUMN] == "test"].drop(columns=[SPLIT_COLUMN])

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "confusion_matrix": cm,
        "classification_report": report,
    }


def upload_file_to_gcs(local_path: Path, bucket_name: str, destination_blob: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{destination_blob}"


def create_vertex_dataset(
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

    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=[gcs_uri],
        sync=True,
    )

    return dataset


def run_vertex_automl_training(
    dataset,
    display_name: str,
    model_display_name: str,
    budget_milli_node_hours: int,
):
    training_job = aiplatform.AutoMLTabularTrainingJob(
        display_name=display_name,
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-prc",
        column_specs=COLUMN_SPECS,
        labels={
            "course": "aiops",
            "topic": "automl",
            "case": "incident-risk",
        },
    )

    model = training_job.run(
        dataset=dataset,
        target_column=TARGET_COLUMN,
        predefined_split_column_name=SPLIT_COLUMN,
        budget_milli_node_hours=budget_milli_node_hours,
        model_display_name=model_display_name,
        model_labels={
            "course": "aiops",
            "topic": "automl",
            "case": "incident-risk",
        },
        sync=True,
    )

    return model


def main() -> None:
    load_dotenv()

    project_id = require_env("PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "europe-west1")
    bucket_name = require_env("BUCKET_NAME")

    run_automl = os.getenv("RUN_VERTEX_AUTOML", "false").lower() == "true"
    budget = int(os.getenv("AUTOML_BUDGET_MILLI_NODE_HOURS", "1000"))

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    output_dir = Path("data") / version
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Generando dataset operativo sintético...")
    df = build_operational_dataset(rows=1200)

    print("2. Validando dataset para AutoML...")
    quality_report = validate_for_automl(df)
    print(json.dumps(quality_report, indent=2, ensure_ascii=False))

    if not quality_report["passed"]:
        raise RuntimeError("El dataset no pasa la validación mínima para AutoML.")

    dataset_path = output_dir / "aiops_automl_incident_dataset.csv"
    report_path = output_dir / "aiops_automl_quality_report.json"
    baseline_path = output_dir / "aiops_automl_local_baseline.json"

    df.to_csv(dataset_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    print("3. Ejecutando baseline local rápido...")
    baseline = run_local_baseline(df)

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    print("Matriz de confusión baseline local:")
    print(baseline["confusion_matrix"])

    print("4. Subiendo artefactos a Cloud Storage...")
    dataset_gcs_uri = upload_file_to_gcs(
        dataset_path,
        bucket_name,
        f"datasets/automl/{version}/aiops_automl_incident_dataset.csv",
    )

    report_gcs_uri = upload_file_to_gcs(
        report_path,
        bucket_name,
        f"datasets/automl/{version}/aiops_automl_quality_report.json",
    )

    baseline_gcs_uri = upload_file_to_gcs(
        baseline_path,
        bucket_name,
        f"datasets/automl/{version}/aiops_automl_local_baseline.json",
    )

    print(f"Dataset GCS:  {dataset_gcs_uri}")
    print(f"Quality GCS:  {report_gcs_uri}")
    print(f"Baseline GCS: {baseline_gcs_uri}")

    print("5. Creando Vertex AI TabularDataset...")
    vertex_dataset = create_vertex_dataset(
        project_id=project_id,
        location=location,
        bucket_name=bucket_name,
        display_name=f"aiops-automl-incident-dataset-{version}",
        gcs_uri=dataset_gcs_uri,
    )

    print(f"Dataset Vertex AI: {vertex_dataset.resource_name}")

    if not run_automl:
        print("\nRUN_VERTEX_AUTOML=false")
        print("No se lanza entrenamiento AutoML para evitar consumo accidental.")
        print("Para entrenar, cambia en .env:")
        print("RUN_VERTEX_AUTOML=true")
        print("\nConfiguración preparada:")
        print(f"target_column: {TARGET_COLUMN}")
        print(f"split_column:  {SPLIT_COLUMN}")
        print(f"budget:        {budget} milli node hours")
        print(f"objective:     maximize-au-prc")
        return

    print("6. Lanzando entrenamiento AutoML en Vertex AI...")
    model = run_vertex_automl_training(
        dataset=vertex_dataset,
        display_name=f"aiops-automl-training-{version}",
        model_display_name=f"aiops-incident-risk-automl-{version}",
        budget_milli_node_hours=budget,
    )

    print("Modelo entrenado:")
    print(model.resource_name)

    print("7. Leyendo evaluaciones del modelo...")
    try:
        evaluations = model.list_model_evaluations()
        for idx, evaluation in enumerate(evaluations, start=1):
            print(f"\nEvaluación {idx}")
            print(json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False))
    except Exception as ex:
        print(f"No se pudieron leer evaluaciones desde el SDK: {ex}")

    print("\nProceso terminado.")


if __name__ == "__main__":
    main()