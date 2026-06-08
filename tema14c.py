import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

load_dotenv()


@dataclass
class CostAssumptions:
    false_positive_review_minutes: float = 6.0
    sre_hourly_cost_eur: float = 65.0
    false_negative_impact_eur: float = 4500.0
    true_positive_prevention_value_eur: float = 1800.0


def generate_predictions(n: int = 2500, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    services = ["checkout", "payments", "identity", "search", "inventory"]
    environments = ["prod", "preprod"]
    severities = ["P1_P2", "none"]

    rows = []

    for i in range(n):
        service = rng.choice(services, p=[0.24, 0.22, 0.18, 0.20, 0.16])
        environment = rng.choice(environments, p=[0.86, 0.14])

        cpu = rng.normal(55, 18)
        memory = rng.normal(60, 16)
        latency_ms = rng.normal(350, 180)
        error_rate = abs(rng.normal(0.015, 0.025))
        recent_deploy = rng.random() < 0.18
        dependency_errors = rng.poisson(1.2)

        # Riesgo real latente.
        risk = (
            0.015
            + max(cpu - 75, 0) * 0.003
            + max(memory - 80, 0) * 0.004
            + max(latency_ms - 600, 0) * 0.0008
            + error_rate * 5
            + dependency_errors * 0.025
            + (0.08 if recent_deploy else 0)
            + (0.05 if service in ["checkout", "payments"] else 0)
        )

        risk = min(max(risk, 0.01), 0.85)

        incident_real = rng.random() < risk
        severity = "P1_P2" if incident_real else "none"

        # Probabilidad estimada por el modelo.
        # Simulamos un modelo razonable, pero imperfecto.
        model_noise = rng.normal(0, 0.12)
        predicted_probability = risk + model_noise

        if incident_real:
            predicted_probability += rng.normal(0.18, 0.10)

        predicted_probability = float(np.clip(predicted_probability, 0.001, 0.999))

        rows.append(
            {
                "event_id": f"EVT-{i + 1:06d}",
                "service": service,
                "environment": environment,
                "cpu": round(float(cpu), 2),
                "memory": round(float(memory), 2),
                "latency_ms": round(float(latency_ms), 2),
                "error_rate": round(float(error_rate), 4),
                "recent_deploy": recent_deploy,
                "dependency_errors": dependency_errors,
                "incident_real": int(incident_real),
                "severity": severity,
                "predicted_probability": round(predicted_probability, 4),
            }
        )

    return pd.DataFrame(rows)


def calculate_global_metrics(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    y_true = df["incident_real"]
    y_score = df["predicted_probability"]
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": threshold,
        "events": len(df),
        "positive_rate": y_true.mean(),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
    }

    result = pd.DataFrame([metrics])

    numeric_cols = [
        "positive_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
        "false_negative_rate",
    ]

    result[numeric_cols] = result[numeric_cols].round(4)

    return result


def evaluate_thresholds(
    df: pd.DataFrame,
    assumptions: CostAssumptions,
) -> pd.DataFrame:
    rows = []

    y_true = df["incident_real"].to_numpy()
    y_score = df["predicted_probability"].to_numpy()

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        false_positive_cost = (
            fp
            * assumptions.false_positive_review_minutes
            * assumptions.sre_hourly_cost_eur
            / 60
        )

        false_negative_cost = fn * assumptions.false_negative_impact_eur
        true_positive_value = tp * assumptions.true_positive_prevention_value_eur

        net_value = true_positive_value - false_positive_cost - false_negative_cost

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_cost_eur": false_positive_cost,
                "false_negative_cost_eur": false_negative_cost,
                "true_positive_value_eur": true_positive_value,
                "net_value_eur": net_value,
            }
        )

    result = pd.DataFrame(rows)

    numeric_cols = [
        "precision",
        "recall",
        "f1",
        "false_positive_cost_eur",
        "false_negative_cost_eur",
        "true_positive_value_eur",
        "net_value_eur",
    ]

    result[numeric_cols] = result[numeric_cols].round(2)

    return result.sort_values("net_value_eur", ascending=False)


def calculate_metrics_by_service(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []

    for service, group in df.groupby("service"):
        y_true = group["incident_real"]
        y_score = group["predicted_probability"]
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        rows.append(
            {
                "service": service,
                "events": len(group),
                "incident_rate": y_true.mean(),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_score)
                if y_true.nunique() > 1
                else np.nan,
                "pr_auc": average_precision_score(y_true, y_score)
                if y_true.nunique() > 1
                else np.nan,
            }
        )

    result = pd.DataFrame(rows)

    numeric_cols = [
        "incident_rate",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    result[numeric_cols] = result[numeric_cols].round(4)

    return result.sort_values("f1", ascending=False)


def assign_recommendation(row: pd.Series) -> str:
    if row["recall"] < 0.60:
        return "Subir recall: revisar features, bajar threshold o entrenar con más incidentes reales"
    if row["precision"] < 0.50:
        return "Reducir falsos positivos: subir threshold o añadir señales de confirmación"
    if row["f1"] < 0.55:
        return "Modelo poco equilibrado: revisar datos, etiquetas y segmentación"
    if row["pr_auc"] < 0.50:
        return "Dataset desbalanceado o señales débiles: mejorar features y muestreo"
    return "Métrica aceptable: monitorizar drift y rendimiento por servicio"


def create_model_quality_report(metrics_by_service: pd.DataFrame) -> pd.DataFrame:
    report = metrics_by_service.copy()
    report["recommendation"] = report.apply(assign_recommendation, axis=1)

    report["quality_status"] = np.where(
        (report["precision"] >= 0.60) & (report["recall"] >= 0.70),
        "Bueno",
        np.where(
            (report["precision"] >= 0.45) & (report["recall"] >= 0.55),
            "Revisar",
            "Crítico",
        ),
    )

    return report


def write_to_bigquery(df: pd.DataFrame, table_name: str) -> None:
    project_id = os.getenv("PROJECT_ID")
    dataset = os.getenv("BQ_DATASET", "aiops_kpis")

    if not project_id:
        print(f"PROJECT_ID no definido. Se omite carga en BigQuery para {table_name}.")
        return

    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    dataset_id = f"{project_id}.{dataset}"
    table_id = f"{dataset_id}.{table_name}"

    client.create_dataset(dataset_id, exists_ok=True)

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()

    print(f"Tabla cargada en BigQuery: {table_id}")


def main() -> None:
    assumptions = CostAssumptions()

    predictions = generate_predictions()

    default_metrics = calculate_global_metrics(predictions, threshold=0.50)
    threshold_table = evaluate_thresholds(predictions, assumptions)

    best_threshold = float(threshold_table.iloc[0]["threshold"])

    metrics_best = calculate_global_metrics(predictions, threshold=best_threshold)
    metrics_by_service = calculate_metrics_by_service(predictions, threshold=best_threshold)
    quality_report = create_model_quality_report(metrics_by_service)

    predictions.to_csv("model_predictions_sample.csv", index=False)
    default_metrics.to_csv("model_metrics_threshold_050.csv", index=False)
    threshold_table.to_csv("model_threshold_evaluation.csv", index=False)
    metrics_best.to_csv("model_metrics_best_threshold.csv", index=False)
    quality_report.to_csv("model_quality_by_service.csv", index=False)

    print("\nMétricas globales con threshold 0.50:\n")
    print(default_metrics.to_string(index=False))

    print("\nMejores thresholds por valor económico estimado:\n")
    print(
        threshold_table[
            [
                "threshold",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "false_positive_cost_eur",
                "false_negative_cost_eur",
                "true_positive_value_eur",
                "net_value_eur",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nThreshold recomendado por valor económico: {best_threshold:.2f}")

    print("\nCalidad del modelo por servicio:\n")
    print(
        quality_report[
            [
                "service",
                "events",
                "incident_rate",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "quality_status",
                "recommendation",
            ]
        ].to_string(index=False)
    )

    write_to_bigquery(predictions, "model_predictions_sample")
    write_to_bigquery(default_metrics, "model_metrics_threshold_050")
    write_to_bigquery(threshold_table, "model_threshold_evaluation")
    write_to_bigquery(metrics_best, "model_metrics_best_threshold")
    write_to_bigquery(quality_report, "model_quality_by_service")


if __name__ == "__main__":
    main()