from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_PATH = Path("latency_regression_model.joblib")
PREDICTIONS_PATH = Path("latency_predictions.csv")


def build_dataset() -> pd.DataFrame:
    """
    Dataset sintético de señales operativas.
    En un caso real, estos datos vendrían de Cloud Monitoring, Cloud Logging,
    BigQuery, Pub/Sub o una tabla curada de observabilidad.
    """

    rows = [
        # checkout-api
        ["checkout-api", "europe-west1", "prod", 1200, 0.3, 42, 58, 3, 0, 210],
        ["checkout-api", "europe-west1", "prod", 2500, 0.8, 55, 62, 3, 0, 285],
        ["checkout-api", "europe-west1", "prod", 4300, 1.4, 68, 71, 4, 0, 390],
        ["checkout-api", "europe-west1", "prod", 6200, 2.5, 79, 78, 4, 1, 610],
        ["checkout-api", "europe-west1", "prod", 8500, 4.2, 88, 84, 5, 1, 890],
        ["checkout-api", "europe-west1", "prod", 9600, 5.1, 93, 89, 5, 1, 1120],

        # payment-api
        ["payment-api", "europe-west1", "prod", 900, 0.2, 38, 51, 2, 0, 180],
        ["payment-api", "europe-west1", "prod", 1600, 0.7, 48, 57, 2, 0, 245],
        ["payment-api", "europe-west1", "prod", 2900, 1.2, 63, 66, 3, 0, 330],
        ["payment-api", "europe-west1", "prod", 4800, 2.8, 76, 75, 3, 1, 570],
        ["payment-api", "europe-west1", "prod", 6100, 3.9, 85, 82, 4, 1, 760],
        ["payment-api", "europe-west1", "prod", 7400, 5.4, 91, 87, 4, 1, 980],

        # catalog-api
        ["catalog-api", "europe-west1", "prod", 1800, 0.1, 35, 44, 2, 0, 140],
        ["catalog-api", "europe-west1", "prod", 3600, 0.4, 49, 53, 2, 0, 190],
        ["catalog-api", "europe-west1", "prod", 5200, 0.9, 61, 64, 3, 0, 265],
        ["catalog-api", "europe-west1", "prod", 7900, 1.8, 73, 72, 3, 0, 410],
        ["catalog-api", "europe-west1", "prod", 10300, 2.7, 83, 80, 4, 1, 650],
        ["catalog-api", "europe-west1", "prod", 12100, 3.5, 90, 86, 4, 1, 830],

        # orders-api
        ["orders-api", "europe-west1", "prod", 1500, 0.4, 44, 49, 2, 0, 220],
        ["orders-api", "europe-west1", "prod", 2700, 0.9, 57, 61, 2, 0, 310],
        ["orders-api", "europe-west1", "prod", 4100, 1.6, 70, 70, 3, 0, 455],
        ["orders-api", "europe-west1", "prod", 6900, 3.1, 82, 79, 3, 1, 720],
        ["orders-api", "europe-west1", "prod", 8300, 4.3, 89, 85, 4, 1, 940],
        ["orders-api", "europe-west1", "prod", 9700, 5.8, 95, 90, 4, 1, 1230],

        # Same services in dev/test with lower impact
        ["checkout-api", "europe-west1", "test", 700, 0.2, 30, 42, 1, 0, 160],
        ["checkout-api", "europe-west1", "test", 1600, 0.5, 45, 54, 1, 0, 230],
        ["payment-api", "europe-west1", "test", 600, 0.1, 28, 39, 1, 0, 140],
        ["payment-api", "europe-west1", "test", 1400, 0.4, 42, 50, 1, 1, 240],
        ["catalog-api", "europe-west1", "test", 1100, 0.2, 34, 46, 1, 0, 150],
        ["orders-api", "europe-west1", "test", 1300, 0.6, 47, 55, 1, 1, 260],

        # Another region
        ["checkout-api", "us-central1", "prod", 2200, 0.6, 52, 60, 3, 0, 300],
        ["checkout-api", "us-central1", "prod", 5700, 2.1, 77, 76, 4, 1, 640],
        ["payment-api", "us-central1", "prod", 2500, 0.8, 56, 63, 3, 0, 310],
        ["payment-api", "us-central1", "prod", 6600, 3.7, 87, 84, 4, 1, 850],
        ["catalog-api", "us-central1", "prod", 4400, 0.7, 58, 62, 3, 0, 250],
        ["catalog-api", "us-central1", "prod", 9900, 2.4, 81, 79, 4, 1, 620],
        ["orders-api", "us-central1", "prod", 3900, 1.4, 69, 68, 3, 0, 430],
        ["orders-api", "us-central1", "prod", 8800, 4.9, 92, 88, 4, 1, 1080],
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


def build_pipeline() -> Pipeline:
    categorical_features = ["service", "region", "environment"]
    numeric_features = [
        "request_count",
        "error_rate",
        "cpu_percent",
        "memory_percent",
        "instances",
        "is_after_deployment",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )

    pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(df: pd.DataFrame) -> Pipeline:
    feature_columns = [
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

    target_column = "latency_p95_ms"

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n=== Evaluación del modelo ===")
    print(f"MAE  : {mae:.2f} ms")
    print(f"RMSE : {rmse:.2f} ms")
    print(f"R2   : {r2:.3f}")

    comparison = X_test.copy()
    comparison["real_latency_p95_ms"] = y_test.values
    comparison["predicted_latency_p95_ms"] = predictions.round(1)
    comparison["absolute_error_ms"] = (
        comparison["real_latency_p95_ms"] - comparison["predicted_latency_p95_ms"]
    ).abs().round(1)

    print("\n=== Comparación real vs predicción ===")
    print(comparison.sort_values("absolute_error_ms", ascending=False))

    return pipeline


def train_final_model(df: pd.DataFrame) -> Pipeline:
    feature_columns = [
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

    target_column = "latency_p95_ms"

    pipeline = build_pipeline()
    pipeline.fit(df[feature_columns], df[target_column])

    return pipeline


def predict_new_situations(pipeline: Pipeline) -> pd.DataFrame:
    new_data = pd.DataFrame(
        [
            {
                "service": "checkout-api",
                "region": "europe-west1",
                "environment": "prod",
                "request_count": 9200,
                "error_rate": 4.8,
                "cpu_percent": 91,
                "memory_percent": 86,
                "instances": 5,
                "is_after_deployment": 1,
            },
            {
                "service": "catalog-api",
                "region": "europe-west1",
                "environment": "prod",
                "request_count": 5200,
                "error_rate": 0.8,
                "cpu_percent": 58,
                "memory_percent": 61,
                "instances": 3,
                "is_after_deployment": 0,
            },
            {
                "service": "orders-api",
                "region": "us-central1",
                "environment": "prod",
                "request_count": 8700,
                "error_rate": 4.5,
                "cpu_percent": 90,
                "memory_percent": 87,
                "instances": 4,
                "is_after_deployment": 1,
            },
        ]
    )

    predictions = pipeline.predict(new_data)

    result = new_data.copy()
    result["predicted_latency_p95_ms"] = predictions.round(1)

    # Regla operativa sencilla para interpretar la predicción
    result["risk_level"] = pd.cut(
        result["predicted_latency_p95_ms"],
        bins=[0, 300, 700, float("inf")],
        labels=["low", "medium", "high"],
        right=False,
    )

    result["recommended_action"] = result["risk_level"].map(
        {
            "low": "Sin acción inmediata",
            "medium": "Revisar métricas y tendencia",
            "high": "Generar alerta predictiva y revisar escalado",
        }
    )

    return result


def main() -> None:
    df = build_dataset()

    print("=== Dataset de entrenamiento ===")
    print(df)

    _ = evaluate_model(df)

    final_model = train_final_model(df)
    joblib.dump(final_model, MODEL_PATH)

    print(f"\nModelo guardado en: {MODEL_PATH}")

    predictions = predict_new_situations(final_model)

    print("\n=== Predicciones nuevas ===")
    print(predictions)

    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

    print(f"\nPredicciones exportadas a: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()