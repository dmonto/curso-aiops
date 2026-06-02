from __future__ import annotations

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_events() -> pd.DataFrame:
    rows = [
        {
            "message": "HTTP 500 rate increased in checkout API",
            "severity": "ERROR",
            "service": "checkout-api",
            "resource_type": "cloud_run_revision",
            "metric_value": 18.2,
        },
        {
            "message": "Unhandled exception in payment validation flow",
            "severity": "ERROR",
            "service": "payment-api",
            "resource_type": "cloud_run_revision",
            "metric_value": 1.0,
        },
        {
            "message": "Null pointer exception after deployment",
            "severity": "ERROR",
            "service": "orders-api",
            "resource_type": "gke_container",
            "metric_value": 1.0,
        },
        {
            "message": "Application error returned by backend service",
            "severity": "ERROR",
            "service": "catalog-api",
            "resource_type": "gke_container",
            "metric_value": 1.0,
        },
        {
            "message": "CPU utilization above 95 percent for ten minutes",
            "severity": "WARNING",
            "service": "orders-api",
            "resource_type": "gce_instance",
            "metric_value": 96.7,
        },
        {
            "message": "Memory pressure detected in product catalog service",
            "severity": "WARNING",
            "service": "catalog-api",
            "resource_type": "gke_container",
            "metric_value": 91.3,
        },
        {
            "message": "Disk usage reached critical threshold",
            "severity": "ERROR",
            "service": "batch-loader",
            "resource_type": "gce_instance",
            "metric_value": 94.5,
        },
        {
            "message": "Instance saturated by high CPU and memory usage",
            "severity": "WARNING",
            "service": "worker-service",
            "resource_type": "gce_instance",
            "metric_value": 89.0,
        },
        {
            "message": "Permission denied for service account while accessing bucket",
            "severity": "ERROR",
            "service": "batch-loader",
            "resource_type": "service_account",
            "metric_value": 1.0,
        },
        {
            "message": "Multiple failed authentication attempts detected",
            "severity": "WARNING",
            "service": "admin-portal",
            "resource_type": "iam_service_account",
            "metric_value": 37.0,
        },
        {
            "message": "Access denied from unexpected region",
            "severity": "ERROR",
            "service": "admin-portal",
            "resource_type": "iam_policy",
            "metric_value": 1.0,
        },
        {
            "message": "Token validation failed for workload identity",
            "severity": "ERROR",
            "service": "internal-api",
            "resource_type": "iam_service_account",
            "metric_value": 1.0,
        },
        {
            "message": "Packet loss detected between frontend and backend subnet",
            "severity": "WARNING",
            "service": "frontend",
            "resource_type": "vpc_network",
            "metric_value": 12.4,
        },
        {
            "message": "Connection timeout calling inventory service",
            "severity": "ERROR",
            "service": "checkout-api",
            "resource_type": "vpc_network",
            "metric_value": 8.0,
        },
        {
            "message": "High latency detected in internal load balancer",
            "severity": "WARNING",
            "service": "frontend",
            "resource_type": "load_balancer",
            "metric_value": 850.0,
        },
        {
            "message": "DNS resolution failed for dependency service",
            "severity": "ERROR",
            "service": "orders-api",
            "resource_type": "vpc_network",
            "metric_value": 1.0,
        },
        {
            "message": "Dataflow job failed while processing orders stream",
            "severity": "ERROR",
            "service": "orders-pipeline",
            "resource_type": "dataflow_job",
            "metric_value": 1.0,
        },
        {
            "message": "BigQuery load job failed due to invalid schema",
            "severity": "ERROR",
            "service": "analytics-loader",
            "resource_type": "bigquery_job",
            "metric_value": 1.0,
        },
        {
            "message": "Pub/Sub backlog growing in ingestion subscription",
            "severity": "WARNING",
            "service": "ingestion-pipeline",
            "resource_type": "pubsub_subscription",
            "metric_value": 45000.0,
        },
        {
            "message": "Streaming pipeline delayed due to unacked messages",
            "severity": "WARNING",
            "service": "events-pipeline",
            "resource_type": "pubsub_subscription",
            "metric_value": 38000.0,
        },
    ]

    return pd.DataFrame(rows)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "message_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
                "message",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["severity", "service", "resource_type"],
            ),
            (
                "numeric",
                StandardScaler(),
                ["metric_value"],
            ),
        ]
    )


def choose_number_of_clusters(df: pd.DataFrame) -> None:
    feature_columns = [
        "message",
        "severity",
        "service",
        "resource_type",
        "metric_value",
    ]

    X = df[feature_columns]
    preprocessor = build_preprocessor()
    X_vectorized = preprocessor.fit_transform(X)

    print("\n=== Evaluación rápida de número de clusters ===")

    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_vectorized)

        score = silhouette_score(X_vectorized, labels)
        print(f"k={k} -> silhouette={score:.3f}")


def cluster_events(df: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame, Pipeline]:
    feature_columns = [
        "message",
        "severity",
        "service",
        "resource_type",
        "metric_value",
    ]

    pipeline = Pipeline(
        steps=[
            ("features", build_preprocessor()),
            (
                "cluster",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init="auto",
                ),
            ),
        ]
    )

    X = df[feature_columns]
    cluster_ids = pipeline.fit_predict(X)

    result = df.copy()
    result["cluster_id"] = cluster_ids

    return result, pipeline


def show_clusters(clustered_df: pd.DataFrame) -> None:
    print("\n=== Eventos agrupados por cluster ===")

    for cluster_id in sorted(clustered_df["cluster_id"].unique()):
        subset = clustered_df[clustered_df["cluster_id"] == cluster_id]

        print(f"\n--- Cluster {cluster_id} | eventos={len(subset)} ---")

        for _, row in subset.iterrows():
            print(
                f"[{row['severity']}] "
                f"{row['service']} | "
                f"{row['resource_type']} | "
                f"{row['message']}"
            )


def reduce_to_2d_for_inspection(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "message",
        "severity",
        "service",
        "resource_type",
        "metric_value",
    ]

    X = df[feature_columns]

    preprocessor = build_preprocessor()
    X_vectorized = preprocessor.fit_transform(X)

    reducer = TruncatedSVD(n_components=2, random_state=42)
    coords = reducer.fit_transform(X_vectorized)

    reduced = df.copy()
    reduced["x"] = coords[:, 0].round(4)
    reduced["y"] = coords[:, 1].round(4)

    return reduced


def assign_new_events(pipeline: Pipeline) -> pd.DataFrame:
    new_events = pd.DataFrame(
        [
            {
                "message": "Service account cannot read object from storage bucket",
                "severity": "ERROR",
                "service": "batch-loader",
                "resource_type": "service_account",
                "metric_value": 1.0,
            },
            {
                "message": "Cloud Run service returning many HTTP 500 errors",
                "severity": "ERROR",
                "service": "checkout-api",
                "resource_type": "cloud_run_revision",
                "metric_value": 25.0,
            },
            {
                "message": "Pub/Sub subscription has many unacked messages",
                "severity": "WARNING",
                "service": "ingestion-pipeline",
                "resource_type": "pubsub_subscription",
                "metric_value": 60000.0,
            },
            {
                "message": "Instance CPU and memory are above expected baseline",
                "severity": "WARNING",
                "service": "worker-service",
                "resource_type": "gce_instance",
                "metric_value": 93.0,
            },
        ]
    )

    feature_columns = [
        "message",
        "severity",
        "service",
        "resource_type",
        "metric_value",
    ]

    new_events["cluster_id"] = pipeline.predict(new_events[feature_columns])

    return new_events


def main() -> None:
    df = build_events()

    print("=== Dataset inicial ===")
    print(df)

    choose_number_of_clusters(df)

    clustered_df, pipeline = cluster_events(df, n_clusters=5)

    show_clusters(clustered_df)

    reduced_df = reduce_to_2d_for_inspection(clustered_df)

    print("\n=== Coordenadas 2D para inspección o visualización ===")
    print(
        reduced_df[
            [
                "message",
                "severity",
                "service",
                "resource_type",
                "cluster_id",
                "x",
                "y",
            ]
        ]
    )

    new_events = assign_new_events(pipeline)

    print("\n=== Nuevos eventos asignados a clusters existentes ===")
    print(new_events)

    output_path = "clustered_events.csv"
    clustered_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nCSV generado: {output_path}")


if __name__ == "__main__":
    main()