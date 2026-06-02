from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARTIFACT_DIR = Path("drift_artifacts")
MODEL_PATH = ARTIFACT_DIR / "latency_model.joblib"
REPORT_PATH = ARTIFACT_DIR / "drift_report.json"
FEATURE_DRIFT_CSV = ARTIFACT_DIR / "feature_drift_report.csv"

PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
MAE_DEGRADATION_FACTOR = 1.5


def build_baseline_dataset() -> pd.DataFrame:
    rows = [
        ["checkout-api", "europe-west1", "prod", 1200, 0.3, 42, 58, 3, 0, 210],
        ["checkout-api", "europe-west1", "prod", 2500, 0.8, 55, 62, 3, 0, 285],
        ["checkout-api", "europe-west1", "prod", 4300, 1.4, 68, 71, 4, 0, 390],
        ["checkout-api", "europe-west1", "prod", 6200, 2.5, 79, 78, 4, 1, 610],
        ["payment-api", "europe-west1", "prod", 900, 0.2, 38, 51, 2, 0, 180],
        ["payment-api", "europe-west1", "prod", 1600, 0.7, 48, 57, 2, 0, 245],
        ["payment-api", "europe-west1", "prod", 2900, 1.2, 63, 66, 3, 0, 330],
        ["payment-api", "europe-west1", "prod", 4800, 2.8, 76, 75, 3, 1, 570],
        ["catalog-api", "europe-west1", "prod", 1800, 0.1, 35, 44, 2, 0, 140],
        ["catalog-api", "europe-west1", "prod", 3600, 0.4, 49, 53, 2, 0, 190],
        ["catalog-api", "europe-west1", "prod", 5200, 0.9, 61, 64, 3, 0, 265],
        ["catalog-api", "europe-west1", "prod", 7900, 1.8, 73, 72, 3, 0, 410],
        ["orders-api", "europe-west1", "prod", 1500, 0.4, 44, 49, 2, 0, 220],
        ["orders-api", "europe-west1", "prod", 2700, 0.9, 57, 61, 2, 0, 310],
        ["orders-api", "europe-west1", "prod", 4100, 1.6, 70, 70, 3, 0, 455],
        ["orders-api", "europe-west1", "prod", 6900, 3.1, 82, 79, 3, 1, 720],
        ["checkout-api", "us-central1", "prod", 2200, 0.6, 52, 60, 3, 0, 300],
        ["checkout-api", "us-central1", "prod", 5700, 2.1, 77, 76, 4, 1, 640],
        ["payment-api", "us-central1", "prod", 2500, 0.8, 56, 63, 3, 0, 310],
        ["payment-api", "us-central1", "prod", 6600, 3.7, 87, 84, 4, 1, 850],
        ["catalog-api", "us-central1", "prod", 4400, 0.7, 58, 62, 3, 0, 250],
        ["catalog-api", "us-central1", "prod", 9900, 2.4, 81, 79, 4, 1, 620],
        ["orders-api", "us-central1", "prod", 3900, 1.4, 69, 68, 3, 0, 430],
        ["orders-api", "us-central1", "prod", 8800, 4.9, 92, 88, 4, 1, 1080],
    ]

    return build_dataframe(rows)


def build_current_dataset_with_drift() -> pd.DataFrame:
    """
    Simula producción actual con más tráfico, CPU más alta y mayor error_rate.
    También incluye valores reales de latency_p95_ms para medir performance drift.
    """

    rows = [
        ["checkout-api", "europe-west1", "prod", 7600, 3.8, 86, 83, 5, 1, 930],
        ["checkout-api", "europe-west1", "prod", 8400, 4.5, 90, 86, 5, 1, 1050],
        ["checkout-api", "europe-west1", "prod", 9200, 5.2, 94, 89, 6, 1, 1210],
        ["checkout-api", "europe-west1", "prod", 10100, 5.9, 96, 91, 6, 1, 1360],
        ["payment-api", "europe-west1", "prod", 5400, 3.1, 82, 79, 4, 1, 760],
        ["payment-api", "europe-west1", "prod", 6200, 3.9, 87, 83, 4, 1, 890],
        ["payment-api", "europe-west1", "prod", 7100, 4.7, 91, 87, 5, 1, 1030],
        ["payment-api", "europe-west1", "prod", 7900, 5.4, 94, 90, 5, 1, 1170],
        ["catalog-api", "europe-west1", "prod", 9100, 2.6, 80, 78, 4, 1, 670],
        ["catalog-api", "europe-west1", "prod", 10400, 3.2, 85, 82, 4, 1, 790],
        ["catalog-api", "europe-west1", "prod", 11600, 3.8, 89, 85, 5, 1, 910],
        ["catalog-api", "europe-west1", "prod", 12800, 4.4, 92, 88, 5, 1, 1030],
        ["orders-api", "europe-west1", "prod", 7700, 3.9, 88, 85, 4, 1, 980],
        ["orders-api", "europe-west1", "prod", 8600, 4.6, 92, 88, 4, 1, 1120],
        ["orders-api", "europe-west1", "prod", 9400, 5.2, 95, 90, 5, 1, 1280],
        ["orders-api", "europe-west1", "prod", 10300, 6.0, 97, 92, 5, 1, 1440],
        ["checkout-api", "us-central1", "prod", 7200, 3.6, 84, 81, 5, 1, 880],
        ["payment-api", "us-central1", "prod", 6900, 4.2, 89, 86, 5, 1, 960],
        ["catalog-api", "us-central1", "prod", 10900, 3.4, 86, 83, 5, 1, 820],
        ["orders-api", "us-central1", "prod", 9800, 5.5, 96, 91, 5, 1, 1320],
    ]

    return build_dataframe(rows)


def build_dataframe(rows: list[list[object]]) -> pd.DataFrame:
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


def numeric_feature_columns() -> list[str]:
    return [
        "request_count",
        "error_rate",
        "cpu_percent",
        "memory_percent",
        "instances",
        "is_after_deployment",
    ]


def train_model(baseline_df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    categorical_features = ["service", "region", "environment"]
    numeric_features = numeric_feature_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
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

    X = baseline_df[feature_columns()]
    y = baseline_df["latency_p95_ms"]

    pipeline.fit(X, y)

    baseline_predictions = pipeline.predict(X)

    baseline_metrics = {
        "mae": float(mean_absolute_error(y, baseline_predictions)),
        "rmse": float(mean_squared_error(y, baseline_predictions) ** 0.5),
        "r2": float(r2_score(y, baseline_predictions)),
    }

    return pipeline, baseline_metrics


def calculate_psi(
    baseline_values: pd.Series,
    current_values: pd.Series,
    bins: int = 10,
) -> float:
    baseline = pd.to_numeric(baseline_values, errors="coerce").dropna()
    current = pd.to_numeric(current_values, errors="coerce").dropna()

    if baseline.empty or current.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(baseline, quantiles))

    if len(breakpoints) < 3:
        return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)

    baseline_pct = baseline_counts / max(len(baseline), 1)
    current_pct = current_counts / max(len(current), 1)

    epsilon = 0.0001
    baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
    current_pct = np.where(current_pct == 0, epsilon, current_pct)

    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)

    return float(np.sum(psi_values))


def drift_level_from_psi(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi >= PSI_CRITICAL_THRESHOLD:
        return "critical"
    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "ok"


def calculate_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in numeric_feature_columns():
        baseline_values = baseline_df[column]
        current_values = current_df[column]

        psi = calculate_psi(baseline_values, current_values)
        ks_result = ks_2samp(baseline_values, current_values)

        baseline_mean = float(baseline_values.mean())
        current_mean = float(current_values.mean())
        mean_diff = current_mean - baseline_mean

        rows.append(
            {
                "feature": column,
                "baseline_mean": round(baseline_mean, 4),
                "current_mean": round(current_mean, 4),
                "mean_diff": round(mean_diff, 4),
                "psi": round(psi, 4),
                "psi_level": drift_level_from_psi(psi),
                "ks_statistic": round(float(ks_result.statistic), 4),
                "ks_pvalue": round(float(ks_result.pvalue), 6),
            }
        )

    return pd.DataFrame(rows)


def calculate_prediction_and_performance_drift(
    model: Pipeline,
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    baseline_metrics: dict[str, float],
) -> dict[str, object]:
    baseline_pred = model.predict(baseline_df[feature_columns()])
    current_pred = model.predict(current_df[feature_columns()])

    prediction_psi = calculate_psi(
        pd.Series(baseline_pred),
        pd.Series(current_pred),
        bins=10,
    )

    current_mae = mean_absolute_error(current_df["latency_p95_ms"], current_pred)
    current_rmse = mean_squared_error(current_df["latency_p95_ms"], current_pred) ** 0.5
    current_r2 = r2_score(current_df["latency_p95_ms"], current_pred)

    mae_degradation = current_mae / max(baseline_metrics["mae"], 0.0001)

    return {
        "prediction_drift": {
            "baseline_prediction_mean": float(np.mean(baseline_pred)),
            "current_prediction_mean": float(np.mean(current_pred)),
            "prediction_psi": float(prediction_psi),
            "prediction_psi_level": drift_level_from_psi(prediction_psi),
        },
        "performance_drift": {
            "baseline_mae": float(baseline_metrics["mae"]),
            "current_mae": float(current_mae),
            "baseline_rmse": float(baseline_metrics["rmse"]),
            "current_rmse": float(current_rmse),
            "baseline_r2": float(baseline_metrics["r2"]),
            "current_r2": float(current_r2),
            "mae_degradation_factor": float(mae_degradation),
            "performance_level": (
                "critical"
                if mae_degradation >= MAE_DEGRADATION_FACTOR
                else "ok"
            ),
        },
    }


def decide_action(
    feature_drift_df: pd.DataFrame,
    prediction_performance_report: dict[str, object],
) -> dict[str, object]:
    critical_features = feature_drift_df[
        feature_drift_df["psi_level"] == "critical"
    ]["feature"].tolist()

    warning_features = feature_drift_df[
        feature_drift_df["psi_level"] == "warning"
    ]["feature"].tolist()

    prediction_level = prediction_performance_report["prediction_drift"][
        "prediction_psi_level"
    ]

    performance_level = prediction_performance_report["performance_drift"][
        "performance_level"
    ]

    if performance_level == "critical":
        action = "retrain_required"
        severity = "critical"
        reason = "El rendimiento del modelo se ha degradado significativamente."
    elif critical_features or prediction_level == "critical":
        action = "review_and_prepare_retraining"
        severity = "high"
        reason = "Hay drift fuerte en features o predicciones."
    elif warning_features or prediction_level == "warning":
        action = "increase_monitoring"
        severity = "medium"
        reason = "Hay drift moderado que requiere seguimiento."
    else:
        action = "continue_monitoring"
        severity = "low"
        reason = "No se observa drift relevante."

    return {
        "severity": severity,
        "action": action,
        "reason": reason,
        "critical_features": critical_features,
        "warning_features": warning_features,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df = build_baseline_dataset()
    current_df = build_current_dataset_with_drift()

    model, baseline_metrics = train_model(baseline_df)
    joblib.dump(model, MODEL_PATH)

    feature_drift_df = calculate_feature_drift(baseline_df, current_df)

    prediction_performance_report = calculate_prediction_and_performance_drift(
        model=model,
        baseline_df=baseline_df,
        current_df=current_df,
        baseline_metrics=baseline_metrics,
    )

    decision = decide_action(feature_drift_df, prediction_performance_report)

    report = {
        "baseline_rows": int(len(baseline_df)),
        "current_rows": int(len(current_df)),
        "thresholds": {
            "psi_warning": PSI_WARNING_THRESHOLD,
            "psi_critical": PSI_CRITICAL_THRESHOLD,
            "mae_degradation_factor": MAE_DEGRADATION_FACTOR,
        },
        "baseline_metrics": baseline_metrics,
        "prediction_and_performance": prediction_performance_report,
        "decision": decision,
    }

    feature_drift_df.to_csv(FEATURE_DRIFT_CSV, index=False, encoding="utf-8")

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Feature drift ===")
    print(feature_drift_df)

    print("\n=== Prediction / Performance drift ===")
    print(json.dumps(prediction_performance_report, indent=2))

    print("\n=== Decisión operativa ===")
    print(json.dumps(decision, indent=2, ensure_ascii=False))

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"CSV de drift guardado en: {FEATURE_DRIFT_CSV}")
    print(f"Reporte JSON guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()