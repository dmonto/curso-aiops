import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FEATURE_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data-uri", required=True)
    parser.add_argument("--target-column", default="incident_next_30m")
    parser.add_argument("--split-column", default="split")
    parser.add_argument(
        "--feature-columns",
        default=",".join(DEFAULT_FEATURE_COLUMNS),
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output-dir", default="local_model")

    return parser.parse_args()


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"No es una URI de Cloud Storage: {uri}")

    without_scheme = uri.replace("gs://", "", 1)
    bucket_name, blob_name = without_scheme.split("/", 1)
    return bucket_name, blob_name


def download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))


def upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))


def load_training_data(uri: str) -> pd.DataFrame:
    if uri.startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "train.csv"
            download_from_gcs(uri, local_path)
            return pd.read_csv(local_path)

    return pd.read_csv(uri)


def build_pipeline(feature_columns: List[str], n_estimators: int, random_state: int) -> Pipeline:
    categorical_feature_names = [
        "service",
        "environment",
        "region",
        "deploy_last_30m",
    ]

    categorical_features = [
        feature_columns.index(col)
        for col in categorical_feature_names
    ]

    numeric_features = [
        idx
        for idx, col in enumerate(feature_columns)
        if col not in categorical_feature_names
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

def evaluate_model(model: Pipeline, test_df: pd.DataFrame, feature_columns: List[str], target_column: str) -> dict:
    x_test = test_df[feature_columns].to_numpy()
    y_test = test_df[target_column]

    y_pred = model.predict(x_test)

    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
        y_score = model.predict_proba(x_test)[:, 1]
        metrics["average_precision"] = float(average_precision_score(y_test, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))

    return metrics


def save_outputs(model: Pipeline, metrics: dict, output_dir: str) -> None:
    local_tmp = Path(tempfile.mkdtemp())
    local_model = local_tmp / "model.joblib"
    local_metrics = local_tmp / "metrics.json"

    joblib.dump(model, local_model)

    with open(local_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if output_dir.startswith("gs://"):
        upload_to_gcs(local_model, output_dir.rstrip("/") + "/model.joblib")
        upload_to_gcs(local_metrics, output_dir.rstrip("/") + "/metrics.json")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, output_path / "model.joblib")

        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    feature_columns = [
        col.strip()
        for col in args.feature_columns.split(",")
        if col.strip()
    ]

    print("Cargando dataset...")
    df = load_training_data(args.train_data_uri)

    required_columns = feature_columns + [args.target_column, args.split_column]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise RuntimeError(f"Faltan columnas en el dataset: {missing_columns}")

    train_df = df[df[args.split_column] == "training"].copy()
    validation_df = df[df[args.split_column] == "validation"].copy()
    test_df = df[df[args.split_column] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("El dataset debe tener filas de training y test.")

    print(f"Filas training:   {len(train_df)}")
    print(f"Filas validation: {len(validation_df)}")
    print(f"Filas test:       {len(test_df)}")

    model = build_pipeline(
        feature_columns=feature_columns,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print("Entrenando modelo...")
    model.fit(
        train_df[feature_columns].to_numpy(),
        train_df[args.target_column],
    )

    print("Evaluando modelo...")
    metrics = evaluate_model(
        model=model,
        test_df=test_df,
        feature_columns=feature_columns,
        target_column=args.target_column,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    vertex_model_dir = os.getenv("AIP_MODEL_DIR")
    output_dir = vertex_model_dir or args.model_output_dir

    print(f"Guardando artefactos en: {output_dir}")
    save_outputs(model, metrics, output_dir)

    print("Entrenamiento terminado correctamente.")


if __name__ == "__main__":
    main()