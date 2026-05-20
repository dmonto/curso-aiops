import os
import re
import sys
import json
import time
import subprocess
from datetime import datetime

def load_env(path=".env"):
    if not os.path.exists(path):
        raise RuntimeError("No encuentro el fichero .env en la carpeta actual.")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

def result(status, name, detail=""):
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    return {"status": status, "name": name, "detail": detail}

def run_cmd(name, cmd, warn=False):
    try:
        completed = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        if completed.returncode == 0:
            return result("PASS", name)
        status = "WARN" if warn else "FAIL"
        detail = completed.stderr.strip() or completed.stdout.strip()
        return result(status, name, detail[:300])
    except Exception as e:
        return result("WARN" if warn else "FAIL", name, str(e))

def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable {name} en .env")
    return value

def safe_suffix():
    raw = os.getenv("USERNAME") or os.getenv("USER") or "student"
    raw = re.sub(r"[^a-zA-Z0-9_]", "_", raw.lower())
    return raw[:30]

def main():
    load_env()

    checks = []

    project_id = require_env("PROJECT_ID")
    region = require_env("REGION")
    vertex_location = os.getenv("VERTEX_LOCATION", region)
    bucket_name = require_env("BUCKET_NAME")
    dataset_id = require_env("DATASET_ID")
    pubsub_topic = require_env("PUBSUB_TOPIC")

    checks.append(run_cmd("gcloud instalado", "gcloud --version"))
    checks.append(run_cmd("Proyecto activo en gcloud", "gcloud config get-value project"))
    checks.append(run_cmd("Token ADC disponible", "gcloud auth application-default print-access-token"))

    try:
        import google.auth
        from google.auth.transport.requests import Request

        credentials, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        checks.append(result("PASS", "Credenciales ADC Python", f"project={adc_project}"))
    except Exception as e:
        checks.append(result("FAIL", "Credenciales ADC Python", str(e)))

    try:
        from google.cloud import storage

        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)

        blob_name = f"smoke/student_smoke_{safe_suffix()}_{int(time.time())}.txt"
        blob = bucket.blob(blob_name)
        blob.upload_from_string("AIOps Vertex AI smoke test OK", content_type="text/plain")

        exists = blob.exists()
        blob.delete()

        if exists:
            checks.append(result("PASS", "Cloud Storage lectura/escritura", f"gs://{bucket_name}/{blob_name}"))
        else:
            checks.append(result("FAIL", "Cloud Storage lectura/escritura", "El objeto no aparece tras subirlo"))
    except Exception as e:
        checks.append(result("FAIL", "Cloud Storage lectura/escritura", str(e)))

    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=project_id)
        table_name = f"student_smoke_{safe_suffix()}"
        query = f"""
        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_name}` AS
        SELECT CURRENT_TIMESTAMP() AS ts, 'ok' AS status
        """
        client.query(query).result()

        rows = list(client.query(
            f"SELECT status FROM `{project_id}.{dataset_id}.{table_name}` LIMIT 1"
        ).result())

        if rows and rows[0]["status"] == "ok":
            checks.append(result("PASS", "BigQuery lectura/escritura", f"{dataset_id}.{table_name}"))
        else:
            checks.append(result("FAIL", "BigQuery lectura/escritura", "No se pudo leer la tabla creada"))
    except Exception as e:
        checks.append(result("FAIL", "BigQuery lectura/escritura", str(e)))

    try:
        from google.cloud import pubsub_v1

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, pubsub_topic)

        future = publisher.publish(
            topic_path,
            b"AIOps Vertex AI smoke test",
            source="student-smoke-test"
        )
        message_id = future.result(timeout=30)

        checks.append(result("PASS", "Pub/Sub publish", f"message_id={message_id}"))
    except Exception as e:
        checks.append(result("FAIL", "Pub/Sub publish", str(e)))

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project_id, location=vertex_location)

        _ = aiplatform.Model.list(
            project=project_id,
            location=vertex_location,
            order_by="create_time desc"
        )

        checks.append(result("PASS", "Vertex AI SDK y acceso API", vertex_location))
    except Exception as e:
        checks.append(result("FAIL", "Vertex AI SDK y acceso API", str(e)))

    try:
        from google.cloud import logging

        client = logging.Client(project=project_id)
        entries = client.list_entries(page_size=1)
        list(entries)
        checks.append(result("PASS", "Cloud Logging lectura"))
    except Exception as e:
        checks.append(result("WARN", "Cloud Logging lectura", str(e)))

    functions_region = os.getenv("FUNCTIONS_REGION", "us-south1")

    checks.append(run_cmd(
        "Cloud Functions listado",
        f"gcloud functions list --regions={functions_region} --format=json",
        warn=True
    ))

    checks.append(run_cmd(
        "Workflows listado",
        f"gcloud workflows list --location={region} --format=json",
        warn=True
    ))

    checks.append(run_cmd(
        "Dataflow listado",
        f"gcloud dataflow jobs list --region={region} --format=json",
        warn=True
    ))

    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")

    print("\nResumen")
    print("-------")
    print(f"PASS = {pass_count}")
    print(f"WARN = {warn_count}")
    print(f"FAIL = {fail_count}")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "project_id": project_id,
        "region": region,
        "summary": {
            "PASS": pass_count,
            "WARN": warn_count,
            "FAIL": fail_count
        },
        "checks": checks
    }

    report_name = f"student_smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nInforme generado: {report_name}")

    if fail_count > 0:
        print("\nResultado: HAY ERRORES BLOQUEANTES. Envía el JSON al coordinador.")
        sys.exit(1)

    print("\nResultado: ENTORNO OK PARA EL CURSO.")

if __name__ == "__main__":
    main()
