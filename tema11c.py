import json
import os
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None


SENSITIVE_METHODS = {
    "google.iam.admin.v1.SetIamPolicy": 35,
    "google.iam.admin.v1.CreateServiceAccountKey": 45,
    "google.iam.credentials.v1.GenerateAccessToken": 30,
    "google.cloud.functions.v2.UpdateFunction": 20,
    "google.cloud.workflows.executions.v1.CreateExecution": 15,
    "google.cloud.aiplatform.v1.Endpoint.Predict": 8,
}

SENSITIVE_RESOURCES = {
    "prod",
    "security",
    "raw",
    "iam",
    "secrets",
}

EXPECTED_SERVICE_ACCOUNT_PREFIXES = (
    "sa-aiops-dataflow",
    "sa-aiops-training",
    "sa-aiops-remediation",
    "sa-aiops-monitoring",
)


@dataclass
class Detection:
    detection_ts: str
    principal: str
    window_start: str
    window_end: str
    event_count: int
    denied_count: int
    sensitive_method_count: int
    sensitive_resource_count: int
    off_hours_count: int
    novelty_count: int
    baseline_event_count: float
    volume_ratio: float
    risk_score: int
    severity: str
    reasons: str
    recommended_action: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_sample_events() -> pd.DataFrame:
    """
    Genera eventos sintéticos similares a auditoría.
    En un entorno real, esta parte se sustituye por lectura desde BigQuery.
    """
    now = utc_now().replace(minute=0, second=0, microsecond=0)
    principals = [
        "user:devops1@empresa.com",
        "user:sre1@empresa.com",
        "serviceAccount:sa-aiops-dataflow@project.iam.gserviceaccount.com",
        "serviceAccount:sa-aiops-training@project.iam.gserviceaccount.com",
        "serviceAccount:sa-aiops-remediation@project.iam.gserviceaccount.com",
    ]

    normal_methods = [
        "google.cloud.bigquery.v2.JobService.InsertJob",
        "google.cloud.logging.v2.ListLogEntries",
        "google.cloud.monitoring.v3.ListTimeSeries",
        "google.cloud.pubsub.v1.Publish",
        "google.cloud.storage.v1.GetObject",
    ]

    resources = [
        "projects/demo/datasets/aiops_curated",
        "projects/demo/topics/aiops-events",
        "projects/demo/buckets/aiops-staging",
        "projects/demo/models/incident-classifier",
    ]

    rows: List[Dict] = []

    for hours_ago in range(72, 0, -1):
        ts_base = now - timedelta(hours=hours_ago)

        for principal in principals:
            normal_count = random.randint(2, 10)

            for _ in range(normal_count):
                ts = ts_base + timedelta(minutes=random.randint(0, 59))
                rows.append(
                    {
                        "event_ts": ts.isoformat(),
                        "principal": principal,
                        "method_name": random.choice(normal_methods),
                        "resource_name": random.choice(resources),
                        "status_code": 0,
                    }
                )

    # Anomalía 1: muchos PERMISSION_DENIED
    attacker = "user:devops1@empresa.com"
    for i in range(35):
        ts = now - timedelta(minutes=random.randint(0, 50))
        rows.append(
            {
                "event_ts": ts.isoformat(),
                "principal": attacker,
                "method_name": "google.cloud.storage.v1.GetObject",
                "resource_name": f"projects/demo/buckets/prod-raw-secrets/object-{i}",
                "status_code": 7,
            }
        )

    # Anomalía 2: service account toca IAM
    sa = "serviceAccount:sa-aiops-dataflow@project.iam.gserviceaccount.com"
    for method in [
        "google.iam.admin.v1.SetIamPolicy",
        "google.iam.admin.v1.CreateServiceAccountKey",
    ]:
        rows.append(
            {
                "event_ts": (now - timedelta(minutes=15)).isoformat(),
                "principal": sa,
                "method_name": method,
                "resource_name": "projects/demo/iam/serviceAccounts/sa-prod",
                "status_code": 0,
            }
        )

    # Anomalía 3: principal nuevo
    for i in range(8):
        rows.append(
            {
                "event_ts": (now - timedelta(minutes=random.randint(0, 30))).isoformat(),
                "principal": "serviceAccount:unknown-runner@project.iam.gserviceaccount.com",
                "method_name": "google.cloud.bigquery.v2.JobService.InsertJob",
                "resource_name": "projects/demo/datasets/prod_finance",
                "status_code": 0,
            }
        )

    return pd.DataFrame(rows)


def load_events_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"event_ts", "principal", "method_name", "resource_name", "status_code"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"El CSV no contiene columnas obligatorias: {missing}")
    return df


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce").fillna(0).astype(int)
    df["hour"] = df["event_ts"].dt.floor("h")
    df["hour_of_day"] = df["event_ts"].dt.hour

    df["is_denied"] = df["status_code"].eq(7)
    df["is_sensitive_method"] = df["method_name"].isin(SENSITIVE_METHODS.keys())
    df["sensitive_method_score"] = df["method_name"].map(SENSITIVE_METHODS).fillna(0).astype(int)

    df["is_sensitive_resource"] = df["resource_name"].str.lower().apply(
        lambda x: any(token in x for token in SENSITIVE_RESOURCES)
    )

    # Simplificación: fuera de horario laboral 08:00-19:00 UTC.
    # En un caso real, ajusta zona horaria, calendario y ventanas de guardia.
    df["is_off_hours"] = ~df["hour_of_day"].between(8, 19)

    df["is_service_account"] = df["principal"].str.startswith("serviceAccount:")
    df["is_unknown_service_account"] = df["is_service_account"] & ~df["principal"].str.contains(
        "|".join(EXPECTED_SERVICE_ACCOUNT_PREFIXES), regex=True
    )

    return df


def compute_baselines(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    historical = df[df["event_ts"] < cutoff].copy()

    if historical.empty:
        return pd.DataFrame(columns=["principal", "baseline_event_count"])

    hourly_counts = (
        historical.groupby(["principal", "hour"])
        .size()
        .reset_index(name="events_per_hour")
    )

    baseline = (
        hourly_counts.groupby("principal")["events_per_hour"]
        .median()
        .reset_index(name="baseline_event_count")
    )

    return baseline


def detect(df: pd.DataFrame) -> List[Detection]:
    df = normalize_events(df)

    max_ts = df["event_ts"].max()
    window_start = max_ts.floor("h")
    window_end = window_start + pd.Timedelta(hours=1)
    cutoff = window_start

    current = df[(df["event_ts"] >= window_start) & (df["event_ts"] < window_end)].copy()
    baseline = compute_baselines(df, cutoff)

    if current.empty:
        return []

    # Métodos históricos por principal para detectar novedad simple
    historical = df[df["event_ts"] < cutoff]
    known_pairs = set(zip(historical["principal"], historical["method_name"]))

    current["is_novel_method_for_principal"] = current.apply(
        lambda row: (row["principal"], row["method_name"]) not in known_pairs,
        axis=1,
    )

    grouped = (
        current.groupby("principal")
        .agg(
            event_count=("method_name", "size"),
            denied_count=("is_denied", "sum"),
            sensitive_method_count=("is_sensitive_method", "sum"),
            sensitive_resource_count=("is_sensitive_resource", "sum"),
            off_hours_count=("is_off_hours", "sum"),
            novelty_count=("is_novel_method_for_principal", "sum"),
            sensitive_method_score=("sensitive_method_score", "sum"),
            unknown_service_account=("is_unknown_service_account", "max"),
        )
        .reset_index()
    )

    grouped = grouped.merge(baseline, on="principal", how="left")
    grouped["baseline_event_count"] = grouped["baseline_event_count"].fillna(1.0)
    grouped["volume_ratio"] = grouped["event_count"] / grouped["baseline_event_count"].clip(lower=1)

    detections: List[Detection] = []

    for _, row in grouped.iterrows():
        reasons: List[str] = []
        score = 0

        if row["volume_ratio"] >= 5:
            score += 20
            reasons.append(f"volumen {row['volume_ratio']:.1f}x sobre baseline")

        if row["denied_count"] >= 10:
            score += min(30, int(row["denied_count"]))
            reasons.append(f"{int(row['denied_count'])} eventos PERMISSION_DENIED")

        if row["sensitive_method_count"] > 0:
            score += int(row["sensitive_method_score"])
            reasons.append(f"{int(row['sensitive_method_count'])} métodos sensibles")

        if row["sensitive_resource_count"] > 0:
            score += min(20, int(row["sensitive_resource_count"]) * 4)
            reasons.append(f"{int(row['sensitive_resource_count'])} accesos a recursos sensibles")

        if row["off_hours_count"] > 0:
            score += min(10, int(row["off_hours_count"]) * 2)
            reasons.append(f"{int(row['off_hours_count'])} eventos fuera de horario")

        if row["novelty_count"] > 0:
            score += min(15, int(row["novelty_count"]) * 3)
            reasons.append(f"{int(row['novelty_count'])} métodos nuevos para el principal")

        if bool(row["unknown_service_account"]):
            score += 25
            reasons.append("service account no registrada en la allowlist del curso")

        if score >= 70:
            severity = "HIGH"
            action = "Abrir incidente, revisar IAM, validar origen y bloquear automatización si procede."
        elif score >= 35:
            severity = "MEDIUM"
            action = "Revisar en dashboard, confirmar cambio previsto y ajustar permisos o baseline."
        elif score >= 15:
            severity = "LOW"
            action = "Registrar como señal débil y observar recurrencia."
        else:
            severity = "INFO"
            action = "Sin acción inmediata."

        if score >= 15:
            detections.append(
                Detection(
                    detection_ts=datetime.now(timezone.utc).isoformat(),
                    principal=row["principal"],
                    window_start=window_start.isoformat(),
                    window_end=window_end.isoformat(),
                    event_count=int(row["event_count"]),
                    denied_count=int(row["denied_count"]),
                    sensitive_method_count=int(row["sensitive_method_count"]),
                    sensitive_resource_count=int(row["sensitive_resource_count"]),
                    off_hours_count=int(row["off_hours_count"]),
                    novelty_count=int(row["novelty_count"]),
                    baseline_event_count=float(row["baseline_event_count"]),
                    volume_ratio=round(float(row["volume_ratio"]), 2),
                    risk_score=int(score),
                    severity=severity,
                    reasons="; ".join(reasons),
                    recommended_action=action,
                )
            )

    return sorted(detections, key=lambda d: d.risk_score, reverse=True)


def write_reports(detections: List[Detection]) -> None:
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"behavior-anomalies-{timestamp}.csv"
    json_path = out_dir / f"behavior-anomalies-{timestamp}.json"
    md_path = out_dir / f"behavior-anomalies-{timestamp}.md"

    rows = [asdict(d) for d in detections]

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("### Informe de comportamientos anómalos\n\n")
        f.write(f"Fecha: {datetime.now().isoformat(timespec='seconds')}\n\n")

        if not detections:
            f.write("No se han detectado comportamientos anómalos con el umbral actual.\n")
        else:
            for d in detections:
                f.write(f"#### {d.severity} - {d.principal}\n\n")
                f.write(f"- Ventana: `{d.window_start}` a `{d.window_end}`\n")
                f.write(f"- Score: `{d.risk_score}`\n")
                f.write(f"- Eventos: `{d.event_count}`\n")
                f.write(f"- Motivos: {d.reasons}\n")
                f.write(f"- Acción recomendada: {d.recommended_action}\n\n")

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


def write_to_bigquery(project_id: str, dataset_id: str, table_id: str, detections: List[Detection]) -> None:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery no está instalado.")

    client = bigquery.Client(project=project_id)
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset_ref.location = "EU"
    client.create_dataset(dataset_ref, exists_ok=True)

    schema = [
        bigquery.SchemaField("detection_ts", "TIMESTAMP"),
        bigquery.SchemaField("principal", "STRING"),
        bigquery.SchemaField("window_start", "TIMESTAMP"),
        bigquery.SchemaField("window_end", "TIMESTAMP"),
        bigquery.SchemaField("event_count", "INTEGER"),
        bigquery.SchemaField("denied_count", "INTEGER"),
        bigquery.SchemaField("sensitive_method_count", "INTEGER"),
        bigquery.SchemaField("sensitive_resource_count", "INTEGER"),
        bigquery.SchemaField("off_hours_count", "INTEGER"),
        bigquery.SchemaField("novelty_count", "INTEGER"),
        bigquery.SchemaField("baseline_event_count", "FLOAT"),
        bigquery.SchemaField("volume_ratio", "FLOAT"),
        bigquery.SchemaField("risk_score", "INTEGER"),
        bigquery.SchemaField("severity", "STRING"),
        bigquery.SchemaField("reasons", "STRING"),
        bigquery.SchemaField("recommended_action", "STRING"),
    ]

    table = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
    client.create_table(table, exists_ok=True)

    rows = [asdict(d) for d in detections]
    errors = client.insert_rows_json(table, rows)

    if errors:
        raise RuntimeError(f"Errores insertando en BigQuery: {errors}")

    print(f"Insertadas {len(rows)} detecciones en BigQuery: {project_id}.{dataset_id}.{table_id}")


def print_console_summary(detections: List[Detection]) -> None:
    print("\nDetección de comportamientos anómalos\n")

    if not detections:
        print("No se han detectado anomalías con el umbral actual.")
        return

    for d in detections:
        print(f"[{d.severity}] {d.principal}")
        print(f"  Score: {d.risk_score}")
        print(f"  Eventos: {d.event_count} | Denegados: {d.denied_count}")
        print(f"  Ratio vs baseline: {d.volume_ratio}")
        print(f"  Motivos: {d.reasons}")
        print(f"  Acción: {d.recommended_action}")
        print()


def main() -> None:
    load_dotenv()

    input_csv = os.getenv("INPUT_EVENTS_CSV", "").strip()

    if input_csv:
        df = load_events_from_csv(input_csv)
    else:
        df = generate_sample_events()

    detections = detect(df)
    print_console_summary(detections)
    write_reports(detections)

    write_bq = os.getenv("WRITE_TO_BIGQUERY", "false").lower() == "true"

    if write_bq:
        project_id = os.getenv("PROJECT_ID")
        dataset_id = os.getenv("DATASET_ID", "aiops_security")
        table_id = os.getenv("BQ_TABLE", "behavior_anomalies")

        if not project_id:
            print("ERROR: PROJECT_ID es obligatorio para escribir en BigQuery.", file=sys.stderr)
            sys.exit(1)

        write_to_bigquery(project_id, dataset_id, table_id, detections)


if __name__ == "__main__":
    main()