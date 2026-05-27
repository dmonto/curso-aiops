import os
from dotenv import load_dotenv

from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


PROJECT_ID = require_env("PROJECT_ID")
REGION = os.getenv("VERTEX_LOCATION", "europe-west1")
PIPELINE_ROOT = require_env("PIPELINE_ROOT")

PIPELINE_PACKAGE = "aiops_incident_risk_pipeline.json"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def generate_operational_dataset(
    output_dataset: Output[Dataset],
    n_rows: int = 2000,
) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)

    cpu_utilization = rng.uniform(20, 99, n_rows)
    memory_utilization = rng.uniform(25, 98, n_rows)
    latency_ms = rng.uniform(100, 2200, n_rows)
    error_rate = rng.uniform(0.0, 0.25, n_rows)
    recent_alerts = rng.poisson(2, n_rows)
    deploy_in_last_hour = rng.integers(0, 2, n_rows)

    # Regla sintética para simular probabilidad de incidencia.
    risk_score = (
        0.025 * cpu_utilization
        + 0.020 * memory_utilization
        + 0.0020 * latency_ms
        + 12.0 * error_rate
        + 0.35 * recent_alerts
        + 1.2 * deploy_in_last_hour
        + rng.normal(0, 1.2, n_rows)
    )

    incident_next_hour = (risk_score > 7.5).astype(int)

    df = pd.DataFrame(
        {
            "cpu_utilization": cpu_utilization,
            "memory_utilization": memory_utilization,
            "latency_ms": latency_ms,
            "error_rate": error_rate,
            "recent_alerts": recent_alerts,
            "deploy_in_last_hour": deploy_in_last_hour,
            "incident_next_hour": incident_next_hour,
        }
    )

    df.to_csv(output_dataset.path, index=False)

    output_dataset.metadata["rows"] = int(len(df))
    output_dataset.metadata["positive_rate"] = float(df["incident_next_hour"].mean())


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
)
def train_and_evaluate_model(
    input_dataset: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    evaluation_dataset: Output[Dataset],
) -> None:
    import json
    import os

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_dataset.path)

    target = "incident_next_hour"
    features = [
        "cpu_utilization",
        "memory_utilization",
        "latency_ms",
        "error_rate",
        "recent_alerts",
        "deploy_in_last_hour",
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    os.makedirs(model_artifact.path, exist_ok=True)
    joblib.dump(clf, os.path.join(model_artifact.path, "model.joblib"))

    model_artifact.metadata["framework"] = "scikit-learn"
    model_artifact.metadata["model_type"] = "RandomForestClassifier"
    model_artifact.metadata["target"] = target

    metrics_artifact.log_metric("accuracy", float(accuracy))
    metrics_artifact.log_metric("precision", float(precision))
    metrics_artifact.log_metric("recall", float(recall))
    metrics_artifact.log_metric("auc", float(auc))

    evaluation = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    with open(evaluation_dataset.path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)


@dsl.component(
    base_image="python:3.11",
)
def decide_model_promotion(
    evaluation_dataset: Input[Dataset],
    decision_dataset: Output[Dataset],
    min_auc: float = 0.75,
) -> None:
    import json
    from datetime import datetime, timezone

    with open(evaluation_dataset.path, "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    auc = float(evaluation["auc"])
    accepted = auc >= min_auc

    decision = {
        "accepted": accepted,
        "reason": (
            f"Modelo aceptado: AUC {auc:.4f} >= {min_auc:.4f}"
            if accepted
            else f"Modelo rechazado: AUC {auc:.4f} < {min_auc:.4f}"
        ),
        "auc": auc,
        "min_auc": min_auc,
        "evaluation": evaluation,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "next_step": (
            "register_candidate_model"
            if accepted
            else "review_features_or_training_data"
        ),
    }

    with open(decision_dataset.path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print(json.dumps(decision, indent=2))


@dsl.pipeline(
    name="aiops-incident-risk-pipeline",
    description="Pipeline ML de laboratorio para predecir riesgo de incidencia operativa.",
)
def aiops_incident_risk_pipeline(
    n_rows: int = 2000,
    min_auc: float = 0.75,
):
    dataset_task = generate_operational_dataset(n_rows=n_rows)

    train_task = train_and_evaluate_model(
        input_dataset=dataset_task.outputs["output_dataset"],
    )

    decide_model_promotion(
        evaluation_dataset=train_task.outputs["evaluation_dataset"],
        min_auc=min_auc,
    )


def compile_pipeline() -> None:
    compiler.Compiler().compile(
        pipeline_func=aiops_incident_risk_pipeline,
        package_path=PIPELINE_PACKAGE,
    )

    print(f"Pipeline compilado en: {PIPELINE_PACKAGE}")


def submit_pipeline() -> None:
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=PIPELINE_ROOT,
    )

    job = aiplatform.PipelineJob(
        display_name="aiops-incident-risk-pipeline-run",
        template_path=PIPELINE_PACKAGE,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "n_rows": 2000,
            "min_auc": 0.75,
        },
        enable_caching=False,
    )

    job.submit(service_account="aiops-runtime-sa@asteci-capacitacion-ia.iam.gserviceaccount.com")

    print("Pipeline enviado a Vertex AI.")
    print(f"Resource name: {job.resource_name}")


if __name__ == "__main__":
    compile_pipeline()
    submit_pipeline()