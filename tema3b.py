import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from google.api_core.exceptions import AlreadyExists, GoogleAPICallError, PermissionDenied
from google.cloud import monitoring_v3
from google.protobuf.duration_pb2 import Duration


METRIC_TYPE = "custom.googleapis.com/aiops/incident_risk_score"


def require_project_id() -> str:
    value = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if not value:
        raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT o PROJECT_ID en el .env")
    return value


def project_name(project_id: str) -> str:
    return f"projects/{project_id}"


def build_duration(seconds: int) -> Duration:
    duration = Duration()
    duration.seconds = seconds
    return duration


def build_metric_filter(environment: str, service: Optional[str]) -> str:
    parts = [
        f'metric.type = "{METRIC_TYPE}"',
        'resource.type = "global"',
        f'metric.labels.environment = "{environment}"',
    ]

    if service:
        parts.append(f'metric.labels.service = "{service}"')

    return " AND ".join(parts)


def build_display_name(environment: str, service: Optional[str]) -> str:
    if service:
        return f"[AIOps][{environment}][{service}] Incident risk alto"

    return f"[AIOps][{environment}] Incident risk alto"


def find_policy_by_display_name(
    client: monitoring_v3.AlertPolicyServiceClient,
    project: str,
    display_name: str,
) -> Optional[monitoring_v3.AlertPolicy]:
    for policy in client.list_alert_policies(name=project):
        if policy.display_name == display_name:
            return policy
    return None


def create_incident_risk_policy(
    client: monitoring_v3.AlertPolicyServiceClient,
    project: str,
    environment: str,
    service: Optional[str],
    threshold_score: float,
    retest_seconds: int,
) -> monitoring_v3.AlertPolicy:
    display_name = build_display_name(environment, service)

    existing = find_policy_by_display_name(client, project, display_name)
    if existing:
        print(f"La política ya existe: {existing.display_name}")
        print(f"Nombre técnico: {existing.name}")
        return existing

    metric_filter = build_metric_filter(environment, service)

    service_text = service if service else "cualquier servicio"

    documentation = f"""
### Qué significa

El score predictivo de riesgo de incidente es alto para `{service_text}` en el entorno `{environment}`.

### Condición

`{METRIC_TYPE} > {threshold_score}` durante {retest_seconds} segundos.

### Interpretación

La métrica `incident_risk_score` representa un valor entre 0 y 1.

- 0.00 indica riesgo muy bajo.
- 0.50 indica riesgo medio.
- 0.80 o superior indica riesgo operativo elevado.
- 1.00 indica riesgo máximo.

### Impacto probable

Puede existir una degradación, patrón anómalo o situación operativa que conviene revisar antes de que derive en incidente real.

### Primeras comprobaciones

1. Revisar en Metrics Explorer la serie por `service` y `environment`.
2. Validar si el riesgo alto afecta a un único servicio o a varios.
3. Revisar logs recientes del servicio afectado.
4. Comprobar si ha habido despliegues, cambios de configuración o aumento de tráfico.
5. Revisar errores, latencia, consumo de CPU/memoria y métricas de disponibilidad.
6. Si el score procede de un modelo predictivo, revisar las features que han contribuido al aumento.

### Criterio de escalado

Escalar si el score se mantiene alto durante más de 15 minutos, afecta a servicios críticos o coincide con errores reales de usuario.
""".strip()

    condition = monitoring_v3.AlertPolicy.Condition(
        display_name=f"incident_risk_score mayor que {threshold_score}",
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter=metric_filter,
            comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
            threshold_value=threshold_score,
            duration=build_duration(retest_seconds),
            trigger=monitoring_v3.AlertPolicy.Condition.Trigger(count=1),
            aggregations=[
                monitoring_v3.Aggregation(
                    alignment_period=build_duration(60),
                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                    cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_NONE,
                )
            ],
        ),
    )

    user_labels = {
        "course": "aiops",
        "environment": environment,
        "type": "incident-risk",
    }

    if service:
        user_labels["service"] = service

    policy = monitoring_v3.AlertPolicy(
        display_name=display_name,
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        enabled=True,
        conditions=[condition],
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content=documentation,
            mime_type="text/markdown",
        ),
        user_labels=user_labels,
    )

    created = client.create_alert_policy(
        name=project,
        alert_policy=policy,
    )

    print(f"Política creada: {created.display_name}")
    print(f"Nombre técnico: {created.name}")
    return created


def list_aiops_policies(
    client: monitoring_v3.AlertPolicyServiceClient,
    project: str,
) -> None:
    print("\nPolíticas AIOps encontradas:\n")

    found = False

    for policy in client.list_alert_policies(name=project):
        labels = dict(policy.user_labels)

        if labels.get("course") != "aiops":
            continue

        found = True

        state = "ENABLED" if policy.enabled else "DISABLED"

        print(f"- {policy.display_name}")
        print(f"  name: {policy.name}")
        print(f"  state: {state}")
        print(f"  labels: {labels}")
        print()

    if not found:
        print("No se han encontrado políticas AIOps.")


def delete_policy_by_display_name(
    client: monitoring_v3.AlertPolicyServiceClient,
    project: str,
    display_name: str,
) -> None:
    policy = find_policy_by_display_name(client, project, display_name)

    if not policy:
        print(f"No existe la política: {display_name}")
        return

    client.delete_alert_policy(name=policy.name)
    print(f"Política eliminada: {display_name}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Demo de alertas inteligentes sobre incident_risk_score en Cloud Monitoring"
    )
    parser.add_argument("--create", action="store_true", help="Crea la política de alerta")
    parser.add_argument("--list", action="store_true", help="Lista políticas AIOps")
    parser.add_argument("--delete", action="store_true", help="Elimina la política creada")
    parser.add_argument("--threshold", type=float, default=0.80, help="Umbral del score, entre 0 y 1")
    parser.add_argument("--retest", type=int, default=300, help="Ventana de retest en segundos")
    parser.add_argument(
        "--service",
        type=str,
        default=None,
        help="Servicio concreto. Ejemplo: checkout. Si no se indica, aplica a cualquier servicio.",
    )

    args = parser.parse_args()

    if args.threshold < 0 or args.threshold > 1:
        raise RuntimeError("--threshold debe estar entre 0 y 1")

    project_id = require_project_id()
    environment = os.getenv("AIOPS_ENVIRONMENT", "lab")

    project = project_name(project_id)
    client = monitoring_v3.AlertPolicyServiceClient()

    display_name = build_display_name(environment, args.service)

    try:
        if args.create:
            create_incident_risk_policy(
                client=client,
                project=project,
                environment=environment,
                service=args.service,
                threshold_score=args.threshold,
                retest_seconds=args.retest,
            )

        if args.list:
            list_aiops_policies(client, project)

        if args.delete:
            delete_policy_by_display_name(client, project, display_name)

        if not args.create and not args.list and not args.delete:
            print("Usa --create, --list o --delete.")

    except PermissionDenied as exc:
        print("Error de permisos en Cloud Monitoring.")
        print("Revisa que tienes permisos para crear, listar o borrar políticas de alerta.")
        print(f"Detalle técnico: {exc}")

    except AlreadyExists as exc:
        print("La política ya existe.")
        print(f"Detalle técnico: {exc}")

    except GoogleAPICallError as exc:
        print("Error llamando a Cloud Monitoring.")
        print("Revisa que monitoring.googleapis.com esté habilitada y que ADC esté configurado.")
        print(f"Detalle técnico: {exc}")


if __name__ == "__main__":
    main()