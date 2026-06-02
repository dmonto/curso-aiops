from __future__ import annotations

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_dataset() -> pd.DataFrame:
    """
    Dataset pequeño y didáctico de eventos operativos.
    En un entorno real vendría de BigQuery, Cloud Logging exportado,
    Pub/Sub, tickets históricos o una tabla curada de incidentes.
    """

    rows = [
        {
            "message": "HTTP 500 rate increased above baseline in checkout API",
            "severity": "ERROR",
            "service": "checkout-api",
            "resource_type": "cloud_run_revision",
            "environment": "prod",
            "metric_value": 18.2,
            "label": "app_error",
        },
        {
            "message": "Unhandled exception in payment validation flow",
            "severity": "ERROR",
            "service": "payment-api",
            "resource_type": "cloud_run_revision",
            "environment": "prod",
            "metric_value": 1.0,
            "label": "app_error",
        },
        {
            "message": "Null pointer exception after new deployment",
            "severity": "ERROR",
            "service": "orders-api",
            "resource_type": "gke_container",
            "environment": "prod",
            "metric_value": 1.0,
            "label": "app_error",
        },
        {
            "message": "CPU utilization above 95 percent for ten minutes",
            "severity": "WARNING",
            "service": "orders-api",
            "resource_type": "gce_instance",
            "environment": "prod",
            "metric_value": 96.7,
            "label": "capacity",
        },
        {
            "message": "Memory pressure detected in product catalog service",
            "severity": "WARNING",
            "service": "catalog-api",
            "resource_type": "gke_container",
            "environment": "prod",
            "metric_value": 91.3,
            "label": "capacity",
        },
        {
            "message": "Disk usage reached critical capacity threshold",
            "severity": "ERROR",
            "service": "batch-loader",
            "resource_type": "gce_instance",
            "environment": "prod",
            "metric_value": 94.5,
            "label": "capacity",
        },
        {
            "message": "Permission denied for service account while accessing bucket",
            "severity": "ERROR",
            "service": "batch-loader",
            "resource_type": "service_account",
            "environment": "prod",
            "metric_value": 1.0,
            "label": "security",
        },
        {
            "message": "Multiple failed authentication attempts detected",
            "severity": "WARNING",
            "service": "admin-portal",
            "resource_type": "iam_service_account",
            "environment": "prod",
            "metric_value": 37.0,
            "label": "security",
        },
        {
            "message": "Access denied from unexpected region",
            "severity": "ERROR",
            "service": "admin-portal",
            "resource_type": "iam_policy",
            "environment": "prod",
            "metric_value": 1.0,
            "label": "security",
        },
        {
            "message": "Packet loss detected between frontend and backend subnet",
            "severity": "WARNING",
            "service": "frontend",
            "resource_type": "vpc_network",
            "environment": "prod",
            "metric_value": 12.4,
            "label": "network",
        },
        {
            "message": "Connection timeout calling inventory service",
            "severity": "ERROR",
            "service": "checkout-api",
            "resource_type": "vpc_network",
            "environment": "prod",
            "metric_value": 8.0,
            "label": "network",
        },
        {
            "message": "High latency detected in internal load balancer",
            "severity": "WARNING",
            "service": "frontend",
            "resource_type": "load_balancer",
            "environment": "prod",
            "metric_value": 850.0,
            "label": "network",
        },
        {
            "message": "Dataflow job failed while processing orders stream",
            "severity": "ERROR",
            "service": "orders-pipeline",
            "resource_type": "dataflow_job",
            "environment": "prod",
            "metric_value": 1.0,
            "label": "data_pipeline",
        },
        {
            "message": "BigQuery load job failed due to invalid schema",
            "severity": "ERROR",
            "service": "analytics-loader",
            "resource_type": "bigquery_job",
            "environment": "prod",
            "metric_value": 1.0,
            "label": "data_pipeline",
        },
        {
            "message": "Pub/Sub backlog growing in ingestion topic",
            "severity": "WARNING",
            "service": "ingestion-pipeline",
            "resource_type": "pubsub_subscription",
            "environment": "prod",
            "metric_value": 45000.0,
            "label": "data_pipeline",
        },
        {
            "message": "Deployment completed successfully",
            "severity": "INFO",
            "service": "checkout-api",
            "resource_type": "cloud_deploy",
            "environment": "prod",
            "metric_value": 0.0,
            "label": "informational",
        },
        {
            "message": "Scheduled maintenance window started",
            "severity": "INFO",
            "service": "platform",
            "resource_type": "maintenance_event",
            "environment": "prod",
            "metric_value": 0.0,
            "label": "informational",
        },
        {
            "message": "Health check passed after restart",
            "severity": "INFO",
            "service": "orders-api",
            "resource_type": "cloud_run_revision",
            "environment": "prod",
            "metric_value": 0.0,
            "label": "informational",
        },
    ]

    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame) -> Pipeline:
    feature_columns = [
        "message",
        "severity",
        "service",
        "resource_type",
        "environment",
        "metric_value",
    ]

    X = df[feature_columns]
    y = df["label"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("message_tfidf", TfidfVectorizer(ngram_range=(1, 2)), "message"),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["severity", "service", "resource_type", "environment"],
            ),
            ("numeric", "passthrough", ["metric_value"]),
        ]
    )

    model = LogisticRegression(
        max_iter=10000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    return pipeline


def evaluate_model(df: pd.DataFrame) -> Pipeline:
    feature_columns = [
        "message",
        "severity",
        "service",
        "resource_type",
        "environment",
        "metric_value",
    ]

    X = df[feature_columns]
    y = df["label"]

    # Dataset pequeño: test_size alto para que se vea el flujo.
    # En un caso real usaríamos más datos y validación más sólida.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=42,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["label"] = y_train

    pipeline = train_model(train_df)

    predictions = pipeline.predict(X_test)

    print("\n=== Matriz de confusión ===")
    print(confusion_matrix(y_test, predictions))

    print("\n=== Informe de clasificación ===")
    print(classification_report(y_test, predictions, zero_division=0))

    return pipeline


def predict_new_events(pipeline: Pipeline) -> None:
    new_events = pd.DataFrame(
        [
            {
                "message": "Cloud Run service returning many HTTP 500 errors",
                "severity": "ERROR",
                "service": "checkout-api",
                "resource_type": "cloud_run_revision",
                "environment": "prod",
                "metric_value": 22.0,
            },
            {
                "message": "Service account cannot read object from storage bucket",
                "severity": "ERROR",
                "service": "batch-loader",
                "resource_type": "service_account",
                "environment": "prod",
                "metric_value": 1.0,
            },
            {
                "message": "Pub/Sub subscription has a large unacked messages backlog",
                "severity": "WARNING",
                "service": "ingestion-pipeline",
                "resource_type": "pubsub_subscription",
                "environment": "prod",
                "metric_value": 62000.0,
            },
            {
                "message": "Instance CPU is above threshold",
                "severity": "WARNING",
                "service": "orders-api",
                "resource_type": "gce_instance",
                "environment": "prod",
                "metric_value": 98.0,
            },
        ]
    )

    predicted_labels = pipeline.predict(new_events)
    probabilities = pipeline.predict_proba(new_events)
    classes = pipeline.classes_

    result = new_events.copy()
    result["predicted_label"] = predicted_labels
    result["confidence"] = probabilities.max(axis=1).round(3)

    print("\n=== Nuevos eventos clasificados ===")
    print(result[["message", "severity", "service", "predicted_label", "confidence"]])

    print("\n=== Probabilidades por clase ===")
    probabilities_df = pd.DataFrame(probabilities, columns=classes)
    print(probabilities_df.round(3))


def main() -> None:
    df = build_dataset()

    print("Eventos de entrenamiento:")
    print(df[["message", "severity", "service", "label"]])

    pipeline = evaluate_model(df)

    # Reentrenamos con todo el dataset antes de guardar,
    # porque la evaluación anterior separó parte para test.
    final_pipeline = train_model(df)

    model_path = "event_classifier.joblib"
    joblib.dump(final_pipeline, model_path)

    print(f"\nModelo guardado en: {model_path}")

    predict_new_events(final_pipeline)


if __name__ == "__main__":
    main()