import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SERVICE_CATALOG = {
    "frontend": {
        "owner_team": "web-sre",
        "dependency": "checkout",
        "business_impact": "medium",
        "base_latency": 120,
        "base_error": 0.3,
    },
    "checkout": {
        "owner_team": "commerce-sre",
        "dependency": "payments",
        "business_impact": "high",
        "base_latency": 230,
        "base_error": 0.5,
    },
    "payments": {
        "owner_team": "payments-sre",
        "dependency": None,
        "business_impact": "critical",
        "base_latency": 260,
        "base_error": 0.6,
    },
}

IMPACT_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 4,
    "critical": 5,
}


def generar_datos_operativos(seed: int = 32) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera métricas y eventos de cambio.
    Simula dos fuentes distintas:
    - métricas por minuto
    - eventos de despliegue
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2026-05-25 08:00:00",
        periods=8 * 60,
        freq="min",
    )

    metric_rows = []

    for service, cfg in SERVICE_CATALOG.items():
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60
            load_factor = 1.0 + 0.25 * np.sin((hour - 8) / 8 * np.pi)

            latency = cfg["base_latency"] * load_factor + rng.normal(0, 18)
            error_rate = cfg["base_error"] + rng.normal(0, 0.12)
            requests = 500 * load_factor + rng.normal(0, 35)

            metric_rows.append({
                "timestamp": ts,
                "service": service,
                "environment": "prod",
                "latency_ms": max(20, latency),
                "error_rate_pct": max(0, error_rate),
                "requests_per_min": max(10, requests),
            })

    metrics = pd.DataFrame(metric_rows)

    changes = pd.DataFrame([
        {
            "change_id": "CHG-2026-00017",
            "timestamp": pd.Timestamp("2026-05-25 10:55:00"),
            "service": "payments",
            "change_type": "deployment",
            "deployment_version": "2026.05.25.2",
            "risk": "medium",
        },
        {
            "change_id": "CHG-2026-00018",
            "timestamp": pd.Timestamp("2026-05-25 13:20:00"),
            "service": "frontend",
            "change_type": "config",
            "deployment_version": "2026.05.25.3",
            "risk": "low",
        },
    ])

    # Incidente raíz en payments después del cambio CHG-2026-00017
    payments_mask = (
        (metrics["service"] == "payments")
        & (metrics["timestamp"] >= pd.Timestamp("2026-05-25 11:05:00"))
        & (metrics["timestamp"] <= pd.Timestamp("2026-05-25 11:45:00"))
    )
    n = payments_mask.sum()
    metrics.loc[payments_mask, "latency_ms"] += np.linspace(120, 430, n)
    metrics.loc[payments_mask, "error_rate_pct"] += np.linspace(1.0, 5.0, n)

    # Propagación a checkout
    checkout_mask = (
        (metrics["service"] == "checkout")
        & (metrics["timestamp"] >= pd.Timestamp("2026-05-25 11:12:00"))
        & (metrics["timestamp"] <= pd.Timestamp("2026-05-25 11:52:00"))
    )
    n = checkout_mask.sum()
    metrics.loc[checkout_mask, "latency_ms"] += np.linspace(90, 280, n)
    metrics.loc[checkout_mask, "error_rate_pct"] += np.linspace(0.6, 3.0, n)

    # Propagación a frontend
    frontend_mask = (
        (metrics["service"] == "frontend")
        & (metrics["timestamp"] >= pd.Timestamp("2026-05-25 11:20:00"))
        & (metrics["timestamp"] <= pd.Timestamp("2026-05-25 12:00:00"))
    )
    n = frontend_mask.sum()
    metrics.loc[frontend_mask, "latency_ms"] += np.linspace(40, 170, n)
    metrics.loc[frontend_mask, "error_rate_pct"] += np.linspace(0.2, 1.1, n)

    return metrics, changes


def enriquecer_metricas(metrics: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """
    Añade contexto operativo:
    - owner
    - dependencia
    - impacto de negocio
    - último cambio conocido
    - minutos desde el cambio
    """
    df = metrics.copy()

    df["owner_team"] = df["service"].map(lambda s: SERVICE_CATALOG[s]["owner_team"])
    df["dependency"] = df["service"].map(lambda s: SERVICE_CATALOG[s]["dependency"])
    df["business_impact"] = df["service"].map(lambda s: SERVICE_CATALOG[s]["business_impact"])
    df["business_impact_score"] = df["business_impact"].map(IMPACT_SCORE)

    enriched_parts = []

    for service, sdf in df.groupby("service"):
        sdf = sdf.sort_values("timestamp").copy()
        service_changes = changes[changes["service"] == service].sort_values("timestamp").copy()

        if service_changes.empty:
            sdf["last_change_id"] = None
            sdf["last_change_type"] = None
            sdf["last_deployment_version"] = "2026.05.25.1"
            sdf["minutes_since_change"] = np.nan
            enriched_parts.append(sdf)
            continue

        merged = pd.merge_asof(
            sdf,
            service_changes.rename(columns={"timestamp": "change_timestamp"}),
            left_on="timestamp",
            right_on="change_timestamp",
            by="service",
            direction="backward",
        )

        merged["last_change_id"] = merged["change_id"]
        merged["last_change_type"] = merged["change_type"]
        merged["last_deployment_version"] = merged["deployment_version"].fillna("2026.05.25.1")
        merged["minutes_since_change"] = (
            (merged["timestamp"] - merged["change_timestamp"])
            .dt.total_seconds()
            .div(60)
        )

        enriched_parts.append(merged)

    result = pd.concat(enriched_parts, ignore_index=True)

    keep_cols = [
        "timestamp",
        "service",
        "environment",
        "latency_ms",
        "error_rate_pct",
        "requests_per_min",
        "owner_team",
        "dependency",
        "business_impact",
        "business_impact_score",
        "last_change_id",
        "last_change_type",
        "last_deployment_version",
        "minutes_since_change",
    ]

    return result[keep_cols]


def construir_features(df: pd.DataFrame, window: int = 45) -> pd.DataFrame:
    """
    Convierte datos enriquecidos en features operativas.
    """
    df = df.sort_values(["service", "timestamp"]).copy()
    result = []

    for service, sdf in df.groupby("service"):
        sdf = sdf.copy()

        for metric in ["latency_ms", "error_rate_pct", "requests_per_min"]:
            rolling_mean = sdf[metric].rolling(window=window, min_periods=20).mean()
            rolling_std = sdf[metric].rolling(window=window, min_periods=20).std()

            sdf[f"{metric}_rolling_mean"] = rolling_mean
            sdf[f"{metric}_zscore"] = (
                (sdf[metric] - rolling_mean) / rolling_std.replace(0, np.nan)
            )

        sdf["latency_anomaly"] = sdf["latency_ms_zscore"] > 2.5
        sdf["error_anomaly"] = sdf["error_rate_pct_zscore"] > 2.5
        sdf["traffic_anomaly"] = sdf["requests_per_min_zscore"].abs() > 2.8

        sdf["signals_count"] = (
            sdf["latency_anomaly"].astype(int)
            + sdf["error_anomaly"].astype(int)
            + sdf["traffic_anomaly"].astype(int)
        )

        # Cambio reciente: ventana operativa de 60 minutos
        sdf["recent_change"] = (
            sdf["minutes_since_change"].notna()
            & (sdf["minutes_since_change"] >= 0)
            & (sdf["minutes_since_change"] <= 60)
        )

        result.append(sdf)

    return pd.concat(result, ignore_index=True)

def calcular_riesgo_operativo(features: pd.DataFrame) -> pd.DataFrame:
    """
    Añade scoring operativo por timestamp y servicio.

    Esta tabla sigue siendo granular, es decir:
    una fila por servicio y minuto.
    """
    df = features.copy()

    df["risk_score"] = (
        df["signals_count"] * 2
        + df["business_impact_score"]
        + df["recent_change"].astype(int) * 2
    )

    df["is_aiops_candidate"] = df["risk_score"] >= 7

    return df

def crear_inteligencia_operativa(scored_features: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte features ya puntuadas en una tabla de inteligencia operativa.

    Entrada:
    - scored_features: datos por timestamp y servicio con risk_score.

    Salida:
    - incidents: tabla agregada de incidentes candidatos.
    """
    df = scored_features.copy()

    required_cols = {"risk_score", "is_aiops_candidate"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            "Faltan columnas de scoring operativo: "
            + ", ".join(sorted(missing_cols))
            + ". Ejecuta primero calcular_riesgo_operativo(features)."
        )

    candidates = df[df["is_aiops_candidate"]].copy()

    if candidates.empty:
        return pd.DataFrame(columns=[
            "incident_id",
            "start",
            "end",
            "probable_root_service",
            "affected_services",
            "owner_team",
            "severity",
            "risk_score_max",
            "evidence",
            "recommended_action",
        ])

    candidates = candidates.sort_values("timestamp")
    candidates["gap_min"] = candidates["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    candidates["new_incident"] = candidates["gap_min"] > 10
    candidates["incident_number"] = candidates["new_incident"].cumsum() + 1

    incidents = []

    for incident_number, group in candidates.groupby("incident_number"):
        group = group.sort_values("timestamp")

        first = group.iloc[0]
        root_service = first["service"]
        affected_services = sorted(group["service"].unique().tolist())

        risk_score_max = int(group["risk_score"].max())
        max_error = group["error_rate_pct"].max()

        if risk_score_max >= 10 or max_error >= 4:
            severity = "P1"
        elif risk_score_max >= 8:
            severity = "P2"
        else:
            severity = "P3"

        owner_team = SERVICE_CATALOG[root_service]["owner_team"]

        evidence_parts = [
            f"Primera señal relevante en {root_service}",
            f"servicios afectados: {', '.join(affected_services)}",
            f"risk_score_max={risk_score_max}",
            f"max_error_rate={max_error:.2f}%",
        ]

        change_ids = sorted(group["last_change_id"].dropna().unique().tolist())
        if change_ids:
            evidence_parts.append(f"cambios recientes: {', '.join(change_ids)}")

        evidence = "; ".join(evidence_parts)

        if severity == "P1":
            action = "abrir_incidente_p1_y_lanzar_diagnostico"
        elif severity == "P2":
            action = "crear_ticket_con_evidencias"
        else:
            action = "observar_y_revisar_si_se_repite"

        incidents.append({
            "incident_id": f"INC-AIOPS-{incident_number:04d}",
            "start": group["timestamp"].min(),
            "end": group["timestamp"].max(),
            "probable_root_service": root_service,
            "affected_services": ", ".join(affected_services),
            "owner_team": owner_team,
            "severity": severity,
            "risk_score_max": risk_score_max,
            "evidence": evidence,
            "recommended_action": action,
        })

    return pd.DataFrame(incidents)

def evaluar_calidad_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revisa campos mínimos para inteligencia operativa.
    """
    required_fields = [
        "timestamp",
        "service",
        "environment",
        "latency_ms",
        "error_rate_pct",
        "owner_team",
        "business_impact",
    ]

    rows = []

    for field in required_fields:
        missing = int(df[field].isna().sum())
        total = int(len(df))
        completeness = 1 - (missing / total)

        rows.append({
            "field": field,
            "missing_values": missing,
            "total_rows": total,
            "completeness_pct": round(completeness * 100, 2),
            "status": "OK" if completeness >= 0.99 else "REVIEW",
        })

    return pd.DataFrame(rows)


def pintar_resultado(features: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 5))

    for service in SERVICE_CATALOG.keys():
        sdf = features[features["service"] == service]
        plt.plot(sdf["timestamp"], sdf["risk_score"], label=service)

    plt.axhline(7, linestyle="--", label="umbral candidato AIOps")
    plt.title("Risk score operativo por servicio")
    plt.xlabel("Tiempo")
    plt.ylabel("Risk score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/tema1a_risk_score.png", dpi=150)
    plt.show()


def main() -> None:
    metrics, changes = generar_datos_operativos()

    enriched = enriquecer_metricas(metrics, changes)
    features = construir_features(enriched)
    scored_features = calcular_riesgo_operativo(features)
    intelligence = crear_inteligencia_operativa(scored_features)
    quality = evaluar_calidad_datos(enriched)

    print("\nMétricas enriquecidas")
    print("-" * 100)
    print(enriched.head(8).to_string(index=False))

    print("\nCalidad de datos operativos")
    print("-" * 100)
    print(quality.to_string(index=False))

    print("\nInteligencia operativa generada")
    print("-" * 100)

    if intelligence.empty:
        print("No se han generado incidentes candidatos.")
    else:
        print(intelligence.to_string(index=False))

    metrics.to_csv("outputs/tema1a_metricas_raw.csv", index=False)
    changes.to_csv("outputs/tema1a_cambios_raw.csv", index=False)
    enriched.to_csv("outputs/tema1a_datos_enriquecidos.csv", index=False)
    features.to_csv("outputs/tema1a_features_operativas.csv", index=False)
    scored_features.to_csv("outputs/tema1a_features_con_scoring.csv", index=False)
    intelligence.to_csv("outputs/tema1a_inteligencia_operativa.csv", index=False)
    quality.to_csv("outputs/tema1a_calidad_datos.csv", index=False)

    pintar_resultado(scored_features)

    print("\nArchivos generados:")
    print("- tema1a_metricas_raw.csv")
    print("- tema1a_cambios_raw.csv")
    print("- tema1a_datos_enriquecidos.csv")
    print("- tema1a_features_operativas.csv")
    print("- tema1a_features_con_scoring.csv")
    print("- tema1a_inteligencia_operativa.csv")
    print("- tema1a_calidad_datos.csv")
    print("- tema1a_risk_score.png")


if __name__ == "__main__":
    main()