import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from google.api_core.exceptions import Forbidden, NotFound, GoogleAPIError
from google.cloud import bigquery
from google.cloud import pubsub_v1
from google.cloud import storage
from google.cloud import aiplatform


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    recommendation: str = ""


def env(name: str, required: bool = True) -> Optional[str]:
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Falta variable de entorno: {name}")
    return value


def run_gcloud(args: List[str]) -> tuple[int, str, str]:
    gcloud = shutil.which("gcloud")
    if not gcloud:
        return 1, "", "gcloud no está disponible en PATH"

    completed = subprocess.run(
        [gcloud] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def check_project_permissions(project_id: str) -> CheckResult:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    permissions = [
        "resourcemanager.projects.get",
        "logging.logEntries.list",
        "monitoring.timeSeries.list",
        "aiplatform.models.list",
        "bigquery.datasets.get",
    ]

    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        session = AuthorizedSession(credentials)

        url = (
            "https://cloudresourcemanager.googleapis.com/v1/"
            f"projects/{project_id}:testIamPermissions"
        )

        response = session.post(
            url,
            json={"permissions": permissions},
            timeout=30,
        )

        if response.status_code != 200:
            return CheckResult(
                name="Project IAM permissions",
                status="WARN",
                detail=f"HTTP {response.status_code}: {response.text}",
                recommendation=(
                    "No se pudo ejecutar testIamPermissions vía API. "
                    "Puede faltar resourcemanager.projects.get o acceso al proyecto."
                ),
            )

        payload = response.json()
        granted = payload.get("permissions", [])
        missing = sorted(set(permissions) - set(granted))

        if missing:
            return CheckResult(
                name="Project IAM permissions",
                status="WARN",
                detail=f"Concedidos: {granted}. Faltan: {missing}",
                recommendation=(
                    "No añadas Owner/Editor. Identifica el rol mínimo para los permisos "
                    "necesarios en las prácticas."
                ),
            )

        return CheckResult(
            name="Project IAM permissions",
            status="OK",
            detail=f"Permisos concedidos: {granted}",
        )

    except Exception as e:
        return CheckResult(
            name="Project IAM permissions",
            status="WARN",
            detail=str(e),
            recommendation=(
                "No se pudo validar con testIamPermissions. "
                "Continúa con las comprobaciones específicas de Storage, BigQuery, Pub/Sub y Vertex AI."
            ),
        )

def check_storage(project_id: str, bucket_name: str) -> CheckResult:
    client = storage.Client(project=project_id)

    try:
        bucket = client.bucket(bucket_name)
        bucket.reload()
        blobs = list(client.list_blobs(bucket_name, max_results=1))

        return CheckResult(
            name="Cloud Storage bucket",
            status="OK",
            detail=f"Bucket accesible: gs://{bucket_name}. Objetos visibles en prueba: {len(blobs)}",
        )

    except NotFound:
        return CheckResult(
            name="Cloud Storage bucket",
            status="FAIL",
            detail=f"No existe el bucket gs://{bucket_name} o no es visible.",
            recommendation="Verifica nombre y proyecto. Si existe, necesitas permisos de lectura sobre el bucket.",
        )

    except Forbidden as e:
        return CheckResult(
            name="Cloud Storage bucket",
            status="FAIL",
            detail=str(e),
            recommendation="Para lectura granular, pide acceso sobre el bucket concreto, no sobre todo el proyecto.",
        )

    except GoogleAPIError as e:
        return CheckResult(
            name="Cloud Storage bucket",
            status="WARN",
            detail=str(e),
        )


def check_bigquery(project_id: str, dataset_id: str, table_id: str) -> CheckResult:
    client = bigquery.Client(project=project_id)
    full_dataset = f"{project_id}.{dataset_id}"
    full_table = f"{project_id}.{dataset_id}.{table_id}"

    try:
        client.get_dataset(full_dataset)
        client.get_table(full_table)

        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query = f"SELECT * FROM `{full_table}` LIMIT 1"
        dry_job = client.query(query, job_config=job_config)

        return CheckResult(
            name="BigQuery dataset/table",
            status="OK",
            detail=(
                f"Dataset y tabla accesibles: {full_table}. "
                f"Bytes estimados en dry-run: {dry_job.total_bytes_processed}"
            ),
        )

    except NotFound:
        return CheckResult(
            name="BigQuery dataset/table",
            status="FAIL",
            detail=f"No existe o no es visible: {full_table}",
            recommendation="Comprueba dataset/tabla. Si existe, pide permisos sobre dataset, tabla o vista autorizada.",
        )

    except Forbidden as e:
        return CheckResult(
            name="BigQuery dataset/table",
            status="FAIL",
            detail=str(e),
            recommendation=(
                "Para consulta necesitas permiso para crear jobs y leer los datos. "
                "Evita BigQuery Admin si solo necesitas consultar."
            ),
        )

    except GoogleAPIError as e:
        return CheckResult(
            name="BigQuery dataset/table",
            status="WARN",
            detail=str(e),
        )


def check_pubsub(project_id: str, topic_name: str) -> CheckResult:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)

    try:
        topic = publisher.get_topic(request={"topic": topic_path})

        permissions_to_test = [
            "pubsub.topics.get",
            "pubsub.topics.publish",
            "pubsub.topics.setIamPolicy",
        ]

        iam_result = publisher.test_iam_permissions(
            request={
                "resource": topic_path,
                "permissions": permissions_to_test,
            }
        )

        granted = sorted(iam_result.permissions)
        risky = "pubsub.topics.setIamPolicy" in granted

        if risky:
            return CheckResult(
                name="Pub/Sub topic",
                status="WARN",
                detail=f"Topic accesible: {topic.name}. Permisos concedidos: {granted}",
                recommendation=(
                    "Para una identidad de publicación normalmente basta con publish. "
                    "Revisa por qué puede modificar IAM del topic."
                ),
            )

        return CheckResult(
            name="Pub/Sub topic",
            status="OK",
            detail=f"Topic accesible: {topic.name}. Permisos concedidos: {granted}",
        )

    except NotFound:
        return CheckResult(
            name="Pub/Sub topic",
            status="FAIL",
            detail=f"No existe o no es visible el topic: {topic_path}",
            recommendation="Verifica nombre. Si existe, pide permisos sobre el topic concreto.",
        )

    except Forbidden as e:
        return CheckResult(
            name="Pub/Sub topic",
            status="FAIL",
            detail=str(e),
            recommendation="Pide un rol granular sobre el topic, por ejemplo publicación o lectura según el caso.",
        )

    except GoogleAPIError as e:
        return CheckResult(
            name="Pub/Sub topic",
            status="WARN",
            detail=str(e),
        )


def check_vertex_ai(project_id: str, region: str) -> CheckResult:
    try:
        aiplatform.init(project=project_id, location=region)
        models = aiplatform.Model.list(project=project_id, location=region)

        return CheckResult(
            name="Vertex AI",
            status="OK",
            detail=f"Vertex AI accesible en {region}. Modelos visibles: {len(models)}",
        )

    except Forbidden as e:
        return CheckResult(
            name="Vertex AI",
            status="FAIL",
            detail=str(e),
            recommendation=(
                "Pide permisos de uso o lectura en Vertex AI. "
                "No uses roles de administración si solo necesitas listar o ejecutar prácticas."
            ),
        )

    except GoogleAPIError as e:
        return CheckResult(
            name="Vertex AI",
            status="WARN",
            detail=str(e),
        )

    except Exception as e:
        return CheckResult(
            name="Vertex AI",
            status="WARN",
            detail=str(e),
            recommendation="Verifica región, APIs habilitadas y permisos.",
        )


def print_report(results: List[CheckResult]) -> None:
    print("\nResultado de control de acceso granular\n")

    for r in results:
        print(f"[{r.status}] {r.name}")
        print(f"  Detalle: {r.detail}")
        if r.recommendation:
            print(f"  Recomendación: {r.recommendation}")
        print()

    fail_count = sum(1 for r in results if r.status == "FAIL")
    warn_count = sum(1 for r in results if r.status == "WARN")

    print("Resumen")
    print(f"  OK:   {sum(1 for r in results if r.status == 'OK')}")
    print(f"  WARN: {warn_count}")
    print(f"  FAIL: {fail_count}")

    if fail_count:
        print("\nHay bloqueos de acceso. No los soluciones con Owner/Editor: identifica el recurso y permiso mínimo.")


def main() -> None:
    load_dotenv()

    project_id = env("PROJECT_ID")
    region = env("VERTEX_LOCATION")
    bucket = env("BUCKET_NAME")
    dataset = env("DATASET_ID")
    table = env("BQ_TABLE")
    topic = env("PUBSUB_TOPIC")

    results = [
        check_project_permissions(project_id),
        check_storage(project_id, bucket),
        check_bigquery(project_id, dataset, table),
        check_pubsub(project_id, topic),
        check_vertex_ai(project_id, region),
    ]

    print_report(results)


if __name__ == "__main__":
    main()