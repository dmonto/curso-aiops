from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARTIFACT_DIR = Path("experiment_artifacts")
RESULTS_CSV = ARTIFACT_DIR / "experiment_results.csv"
BEST_RUN_JSON = ARTIFACT_DIR / "best_run.json"
BEST_MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"


def build_dataset() -> pd.DataFrame:
    rows = [
        ["checkout-api", "europe-west1", "prod", 1200, 0.3, 42, 58, 3, 0, 210],
        ["checkout-api", "europe-west1", "prod", 2500, 0.8, 55, 62, 3, 0, 285],
        ["checkout-api", "europe-west1", "prod", 4300, 1.4, 68, 71, 4, 0, 390],
        ["checkout-api", "europe-west1", "prod", 6200, 2.5, 79, 78, 4, 1, 610],
        ["checkout-api", "europe-west1", "prod", 8500, 4.2, 88, 84, 5, 1, 890],
        ["checkout-api", "europe-west1", "prod", 9600, 5.1, 93, 89, 5, 1, 1120],
        ["payment-api", "europe-west1", "prod", 900, 0.2, 38, 51, 2, 0, 180],
        ["payment-api", "europe-west1", "prod", 1600, 0.7, 48, 57, 2, 0, 245],
        ["payment-api", "europe-west1", "prod", 2900, 1.2, 63, 66, 3, 0, 330],
        ["payment-api", "europe-west1", "prod", 4800, 2.8, 76, 75, 3, 1, 570],
        ["payment-api", "europe-west1", "prod", 6100, 3.9, 85, 82, 4, 1, 760],
        ["payment-api", "europe-west1", "prod", 7400, 5.4, 91, 87, 4, 1, 980],
        ["catalog-api", "europe-west1", "prod", 1800, 0.1, 35, 44, 2, 0, 140],
        ["catalog-api", "europe-west1", "prod", 3600, 0.4, 49, 53, 2, 0, 190],
        ["catalog-api", "europe-west1", "prod", 5200, 0.9, 61, 64, 3, 0, 265],
        ["catalog-api", "europe-west1", "prod", 7900, 1.8, 73, 72, 3, 0, 410],
        ["catalog-api", "europe-west1", "prod", 10300, 2.7, 83, 80, 4, 1, 650],
        ["catalog-api", "europe-west1", "prod", 12100, 3.5, 90, 86, 4, 1, 830],
        ["orders-api", "europe-west1", "prod", 1500, 0.4, 44, 49, 2, 0, 220],
        ["orders-api", "europe-west1", "prod", 2700, 0.9, 57, 61, 2, 0, 310],
        ["orders-api", "europe-west1", "prod", 4100, 1.6, 70, 70, 3, 0, 455],
        ["orders-api", "europe-west1", "prod", 6900, 3.1, 82, 79, 3, 1, 720],
        ["orders-api", "europe-west1", "prod", 8300, 4.3, 89, 85, 4, 1, 940],
        ["orders-api", "europe-west1", "prod", 9700, 5.8, 95, 90, 4, 1, 1230],
        ["checkout-api", "us-central1", "prod", 2200, 0.6, 52, 60, 3, 0, 300],
        ["checkout-api", "us-central1", "prod", 5700, 2.1, 77, 76, 4, 1, 640],
        ["payment-api", "us-central1", "prod", 2500, 0.8, 56, 63, 3, 0, 310],
        ["payment-api", "us-central1", "prod", 6600, 3.7, 87, 84, 4, 1, 850],
        ["catalog-api", "us-central1", "prod", 4400, 0.7, 58, 62, 3, 0, 250],
        ["catalog-api", "us-central1", "prod", 9900, 2.4, 81, 79, 4, 1, 620],
        ["orders-api", "us-central1", "prod", 3900, 1.4, 69, 68, 3, 0, 430],
        ["orders-api", "us-central1", "prod", 8800, 4.9, 92, 88, 4, 1, 1080],
        ["checkout-api", "europe-west1", "test", 700, 0.2, 30, 42, 1, 0, 160],
        ["checkout-api", "europe-west1", "test", 1600, 0.5, 45, 54, 1, 0, 230],
        ["payment-api", "europe-west1", "test", 600, 0.1, 28, 39, 1, 0, 140],
        ["payment-api", "europe-west1", "test", 1400, 0.4, 42, 50, 1, 1, 240],
        ["catalog-api", "europe-west1", "test", 1100, 0.2, 34, 46, 1, 0, 150],
        ["orders-api", "europe-west1", "test", 1300, 0.6, 47, 55, 1, 1, 260],
    ]

    columns = [
        "service",
        "region",
        "environment",
        "request_count",
        "error_rate",
        "cpu_percent",
        "memory_percent",
        "instances",
        "is_after_deployment",
        "latency_p95_ms",
    ]

    return pd.DataFrame(rows, columns=columns)


def feature_columns() -> list[str]:
    return [
        "service",
        "region",
        "environment",
        "request_count",
        "error_rate",
        "cpu_percent",
        "memory_percent",
        "instances",
        "is_after_deployment",
    ]


def build_preprocessor() -> ColumnTransformer:
    categorical_features = ["service", "region", "environment"]
    numeric_features = [
        "request_count",
        "error_rate",
        "cpu_percent",
        "memory_percent",
        "instances",
        "is_after_deployment",
    ]

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )


def build_model(params: dict[str, Any]):
    model_type = params["model_type"]

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
        )

    if model_type == "linear_regression":
        return LinearRegression()

    raise ValueError(f"Tipo de modelo no soportado: {model_type}")


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_preprocessor()),
            ("model", build_model(params)),
        ]
    )


def get_experiment_grid() -> list[dict[str, Any]]:
    return [
        {
            "run_name": "rf-n100-depth4-leaf1",
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 4,
            "min_samples_leaf": 1,
        },
        {
            "run_name": "rf-n300-depth6-leaf2",
            "model_type": "random_forest",
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 2,
        },
        {
            "run_name": "rf-n500-depth8-leaf2",
            "model_type": "random_forest",
            "n_estimators": 500,
            "max_depth": 8,
            "min_samples_leaf": 2,
        },
        {
            "run_name": "rf-n300-depth-none-leaf2",
            "model_type": "random_forest",
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
        },
        {
            "run_name": "linear-regression-baseline",
            "model_type": "linear_regression",
            "n_estimators": 0,
            "max_depth": 0,
            "min_samples_leaf": 0,
        },
    ]


def evaluate_run(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, float]]:
    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "mae": float(round(mae, 4)),
        "rmse": float(round(rmse, 4)),
        "r2": float(round(r2, 4)),
    }

    return pipeline, metrics


def vertex_experiments_enabled() -> bool:
    return os.getenv("ENABLE_VERTEX_EXPERIMENTS", "1") == "1"


def log_to_vertex_experiments(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    from google.cloud import aiplatform

    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("VERTEX_LOCATION", "europe-west1")

    if not project_id:
        raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT")

    aiplatform.init(
        project=project_id,
        location=region,
        experiment=experiment_name,
    )

    safe_params = {
        key: "none" if value is None else value
        for key, value in params.items()
        if key != "run_name"
    }

    with aiplatform.start_run(run=run_name, resume=False):
        aiplatform.log_params(safe_params)
        aiplatform.log_metrics(metrics)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()

    X = df[feature_columns()]
    y = df["latency_p95_ms"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    experiment_name = os.getenv(
        "VERTEX_EXPERIMENT_NAME",
        "aiops-latency-regression",
    )

    results: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_record: dict[str, Any] | None = None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    for params in get_experiment_grid():
        run_name = f"{params['run_name']}-{timestamp}"

        print(f"\n=== Ejecutando run: {run_name} ===")

        model, metrics = evaluate_run(
            params=params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        record = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "target": "latency_p95_ms",
            "rows_total": int(len(df)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            **params,
            **metrics,
        }

        results.append(record)

        print("Parámetros:")
        print(json.dumps(params, indent=2))

        print("Métricas:")
        print(json.dumps(metrics, indent=2))

        if vertex_experiments_enabled():
            log_to_vertex_experiments(
                experiment_name=experiment_name,
                run_name=run_name,
                params={
                    **params,
                    "rows_total": len(df),
                    "rows_train": len(X_train),
                    "rows_test": len(X_test),
                    "target": "latency_p95_ms",
                },
                metrics=metrics,
            )
            print("Run registrado en Vertex AI Experiments.")

        if best_record is None or metrics["mae"] < best_record["mae"]:
            best_model = model
            best_record = record

    results_df = pd.DataFrame(results).sort_values("mae", ascending=True)
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    if best_model is None or best_record is None:
        raise RuntimeError("No se ha generado ningún modelo válido.")

    joblib.dump(best_model, BEST_MODEL_PATH)
    save_json(BEST_RUN_JSON, best_record)

    print("\n=== Ranking de experimentos ===")
    print(results_df)

    print("\n=== Mejor run ===")
    print(json.dumps(best_record, indent=2, ensure_ascii=False))

    print(f"\nResultados guardados en: {RESULTS_CSV}")
    print(f"Mejor modelo guardado en: {BEST_MODEL_PATH}")
    print(f"Mejor run guardado en: {BEST_RUN_JSON}")


if __name__ == "__main__":
    main()