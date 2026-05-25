import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SERVICE_CATALOG = {
    "frontend": {
        "base_latency": 120,
        "base_error": 0.25,
        "base_requests": 900,
        "business_impact_score": 3,
        "owner_team": "web-sre",
    },
    "checkout": {
        "base_latency": 220,
        "base_error": 0.45,
        "base_requests": 600,
        "business_impact_score": 5,
        "owner_team": "commerce-sre",
    },
    "payments": {
        "base_latency": 260,
        "base_error": 0.60,
        "base_requests": 450,
        "business_impact_score": 5,
        "owner_team": "payments-sre",
    },
}


def generar_datos_operativos(seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2026-05-25 08:00:00",
        periods=10 * 60,
        freq="min",
    )

    rows = []

    for service, cfg in SERVICE_CATALOG.items():
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60
            load_factor = 1.0 + 0.25 * np.sin((hour - 8) / 10 * np.pi)

            latency = cfg["base_latency"] * load_factor + rng.normal(0, 18)
            error_rate = cfg["base_error"] + rng.normal(0, 0.10)
            requests = cfg["base_requests"] * load_factor + rng.normal(0, 45)

            rows.append({
                "timestamp": ts,
                "service": service,
                "latency_ms": max(20, latency),
                "error_rate_pct": max(0, error_rate),
                "requests_per_min": max(10, requests),
                "deployment_event": 0,
                "incident_active": 0,
            })

    df = pd.DataFrame(rows)

    # Simulamos despliegues
    deployments = [
        ("payments", "2026-05-25 10:50:00"),
        ("checkout", "2026-05-25 13:15:00"),
    ]

    for service, deploy_time in deployments:
        deploy_ts = pd.Timestamp(deploy_time)
        df.loc[
            (df["service"] == service) & (df["timestamp"] == deploy_ts),
            "deployment_event"
        ] = 1

    # Incidente 1: payments tras despliegue
    aplicar_incidente(
        df=df,
        service="payments",
        start="2026-05-25 11:05:00",
        end="2026-05-25 11:45:00",
        latency_increase_start=100,
        latency_increase_end=420,
        error_increase_start=1.0,
        error_increase_end=5.0,
    )

    # Propagación a checkout
    aplicar_incidente(
        df=df,
        service="checkout",
        start="2026-05-25 11:15:00",
        end="2026-05-25 11:55:00",
        latency_increase_start=70,
        latency_increase_end=260,
        error_increase_start=0.5,
        error_increase_end=2.6,
    )

    # Incidente 2: checkout tras despliegue
    aplicar_incidente(
        df=df,
        service="checkout",
        start="2026-05-25 13:30:00",
        end="2026-05-25 14:05:00",
        latency_increase_start=90,
        latency_increase_end=300,
        error_increase_start=0.8,
        error_increase_end=3.8,
    )

    return df


def aplicar_incidente(
    df: pd.DataFrame,
    service: str,
    start: str,
    end: str,
    latency_increase_start: float,
    latency_increase_end: float,
    error_increase_start: float,
    error_increase_end: float,
) -> None:
    mask = (
        (df["service"] == service)
        & (df["timestamp"] >= pd.Timestamp(start))
        & (df["timestamp"] <= pd.Timestamp(end))
    )

    n = int(mask.sum())

    if n == 0:
        return

    df.loc[mask, "latency_ms"] += np.linspace(
        latency_increase_start,
        latency_increase_end,
        n,
    )

    df.loc[mask, "error_rate_pct"] += np.linspace(
        error_increase_start,
        error_increase_end,
        n,
    )

    df.loc[mask, "incident_active"] = 1


def construir_features(df: pd.DataFrame, window: int = 45) -> pd.DataFrame:
    df = df.sort_values(["service", "timestamp"]).copy()
    parts = []

    for service, sdf in df.groupby("service"):
        sdf = sdf.copy()

        for metric in ["latency_ms", "error_rate_pct", "requests_per_min"]:
            rolling_mean = sdf[metric].rolling(window=window, min_periods=20).mean()
            rolling_std = sdf[metric].rolling(window=window, min_periods=20).std()

            sdf[f"{metric}_rolling_mean"] = rolling_mean
            sdf[f"{metric}_zscore"] = (
                (sdf[metric] - rolling_mean) / rolling_std.replace(0, np.nan)
            )

        # Minutos desde último despliegue
        sdf["deployment_group"] = sdf["deployment_event"].cumsum()
        sdf["last_deployment_time"] = sdf["timestamp"].where(
            sdf["deployment_event"].eq(1)
        ).ffill()

        sdf["minutes_since_deploy"] = (
            (sdf["timestamp"] - sdf["last_deployment_time"])
            .dt.total_seconds()
            .div(60)
        )

        sdf["minutes_since_deploy"] = sdf["minutes_since_deploy"].fillna(9999)

        sdf["recent_deploy"] = (
            (sdf["minutes_since_deploy"] >= 0)
            & (sdf["minutes_since_deploy"] <= 60)
        ).astype(int)

        sdf["business_impact_score"] = SERVICE_CATALOG[service]["business_impact_score"]
        sdf["owner_team"] = SERVICE_CATALOG[service]["owner_team"]

        parts.append(sdf)

    features = pd.concat(parts, ignore_index=True)

    # Label predictivo: queremos anticipar incidente en los próximos 10 minutos.
    # Para cada servicio, si incident_active aparece pronto, marcamos incident_next_10m.
    labeled_parts = []

    for service, sdf in features.groupby("service"):
        sdf = sdf.sort_values("timestamp").copy()
        future_incident = (
            sdf["incident_active"]
            .rolling(window=10, min_periods=1)
            .max()
            .shift(-9)
            .fillna(0)
        )
        sdf["incident_next_10m"] = future_incident.astype(int)
        labeled_parts.append(sdf)

    features = pd.concat(labeled_parts, ignore_index=True)

    features = features.replace([np.inf, -np.inf], np.nan)

    return features


def entrenar_modelo(features: pd.DataFrame) -> tuple[RandomForestClassifier, pd.DataFrame]:
    feature_cols = [
        "latency_ms",
        "error_rate_pct",
        "requests_per_min",
        "latency_ms_zscore",
        "error_rate_pct_zscore",
        "requests_per_min_zscore",
        "recent_deploy",
        "minutes_since_deploy",
        "business_impact_score",
    ]

    model_df = features.dropna(subset=feature_cols + ["incident_next_10m"]).copy()

    X = model_df[feature_cols]
    y = model_df["incident_next_10m"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nEvaluación del modelo")
    print("-" * 80)
    print(classification_report(y_test, y_pred, digits=3))

    print("Matriz de confusión")
    print(confusion_matrix(y_test, y_pred))

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nImportancia de features")
    print("-" * 80)
    print(importance.to_string(index=False))

    importance.to_csv("outputs/tema1b_feature_importance.csv", index=False)

    return model, model_df


def generar_predicciones(
    model: RandomForestClassifier,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = [
        "latency_ms",
        "error_rate_pct",
        "requests_per_min",
        "latency_ms_zscore",
        "error_rate_pct_zscore",
        "requests_per_min_zscore",
        "recent_deploy",
        "minutes_since_deploy",
        "business_impact_score",
    ]

    result = model_df.copy()

    result["incident_risk_score"] = model.predict_proba(result[feature_cols])[:, 1]

    result["risk_level"] = pd.cut(
        result["incident_risk_score"],
        bins=[-0.01, 0.35, 0.70, 1.01],
        labels=["low", "medium", "high"],
    )

    result["recommended_action"] = result.apply(recomendar_accion, axis=1)

    cols = [
        "timestamp",
        "service",
        "owner_team",
        "latency_ms",
        "error_rate_pct",
        "requests_per_min",
        "latency_ms_zscore",
        "error_rate_pct_zscore",
        "recent_deploy",
        "minutes_since_deploy",
        "business_impact_score",
        "incident_active",
        "incident_next_10m",
        "incident_risk_score",
        "risk_level",
        "recommended_action",
    ]

    return result[cols]


def recomendar_accion(row: pd.Series) -> str:
    risk = float(row["incident_risk_score"])
    impact = int(row["business_impact_score"])
    recent_deploy = int(row["recent_deploy"])

    if risk >= 0.80 and impact >= 5:
        return "abrir_incidente_y_lanzar_diagnostico"

    if risk >= 0.70 and recent_deploy == 1:
        return "revisar_cambio_reciente_y_crear_ticket"

    if risk >= 0.50:
        return "crear_ticket_con_evidencias"

    if risk >= 0.35:
        return "observar_y_revisar_si_se_repite"

    return "sin_accion"


def pintar_riesgo(predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 5))

    for service in SERVICE_CATALOG.keys():
        sdf = predictions[predictions["service"] == service]
        plt.plot(
            sdf["timestamp"],
            sdf["incident_risk_score"],
            label=service,
        )

    plt.axhline(0.70, linestyle="--", label="umbral riesgo alto")
    plt.title("Predicción de riesgo de incidente en los próximos 10 minutos")
    plt.xlabel("Tiempo")
    plt.ylabel("Riesgo estimado")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/tema1b_riesgo_incidente.png", dpi=150)
    plt.show()


def main() -> None:
    raw = generar_datos_operativos()
    features = construir_features(raw)

    model, model_df = entrenar_modelo(features)
    predictions = generar_predicciones(model, model_df)

    high_risk = predictions[predictions["risk_level"].eq("high")].copy()

    print("\nPrimeras predicciones de alto riesgo")
    print("-" * 100)

    if high_risk.empty:
        print("No hay predicciones de alto riesgo.")
    else:
        print(
            high_risk[
                [
                    "timestamp",
                    "service",
                    "owner_team",
                    "incident_risk_score",
                    "risk_level",
                    "recommended_action",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    raw.to_csv("outputs/tema1b_datos_raw.csv", index=False)
    features.to_csv("outputs/tema1b_features.csv", index=False)
    predictions.to_csv("outputs/tema1b_predicciones.csv", index=False)
    high_risk.to_csv("outputs/tema1b_predicciones_alto_riesgo.csv", index=False)

    pintar_riesgo(predictions)

    print("\nArchivos generados:")
    print("- outputs/tema1b_datos_raw.csv")
    print("- outputs/tema1b_features.csv")
    print("- outputs/tema1b_feature_importance.csv")
    print("- outputs/tema1b_predicciones.csv")
    print("- outputs/tema1b_predicciones_alto_riesgo.csv")
    print("- outputs/tema1b_riesgo_incidente.png")


if __name__ == "__main__":
    main()