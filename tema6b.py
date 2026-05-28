import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import apache_beam as beam
from apache_beam import pvalue
from apache_beam.options.pipeline_options import PipelineOptions


class ParseAndNormalizeLogEntry(beam.DoFn):
    """
    Convierte una LogEntry exportada desde Cloud Logging a un evento operativo normalizado.

    Entrada:
        bytes recibidos desde Pub/Sub.

    Salida principal:
        dict normalizado para BigQuery.

    Salida lateral deadletter:
        eventos que no se pueden parsear o validar.
    """

    DEADLETTER_TAG = "deadletter"

    def process(self, message: bytes) -> Iterable[Dict[str, Any]]:
        try:
            raw_text = message.decode("utf-8")
            log_entry = json.loads(raw_text)

            json_payload = log_entry.get("jsonPayload") or {}
            resource = log_entry.get("resource") or {}
            resource_labels = resource.get("labels") or {}

            service_name = json_payload.get("service_name")
            event_type = json_payload.get("event_type")
            environment = json_payload.get("environment")
            latency_ms = json_payload.get("latency_ms")
            error_rate = json_payload.get("error_rate")

            if not service_name or not event_type:
                raise ValueError("Faltan campos obligatorios: service_name o event_type")

            normalized = {
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "event_timestamp": log_entry.get("timestamp"),
                "severity": log_entry.get("severity", "DEFAULT"),
                "log_name": log_entry.get("logName"),
                "resource_type": resource.get("type"),
                "project_id": resource_labels.get("project_id"),
                "service_name": str(service_name),
                "environment": str(environment or "unknown"),
                "event_type": str(event_type),
                "latency_ms": int(latency_ms or 0),
                "error_rate": float(error_rate or 0.0),
                "region": str(json_payload.get("region") or "unknown"),
                "correlation_id": str(json_payload.get("correlation_id") or ""),
                "aiops_candidate": bool(json_payload.get("aiops_candidate", False)),
            }

            yield normalized

        except Exception as exc:
            yield pvalue.TaggedOutput(
                self.DEADLETTER_TAG,
                {
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": str(exc),
                    "raw_message": message.decode("utf-8", errors="replace"),
                },
            )


class ClassifyOperationalEvent(beam.DoFn):
    """
    Añade una clasificación operativa simple.

    En producción, esta lógica podría sustituirse por:
    - reglas de negocio más completas,
    - un modelo de BigQuery ML,
    - un endpoint de Vertex AI,
    - o una combinación de reglas + inferencia.
    """

    def process(self, event: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        latency_ms = event.get("latency_ms", 0)
        error_rate = event.get("error_rate", 0.0)

        high_latency = latency_ms > 4000
        high_error_rate = error_rate > 0.20

        if high_latency and high_error_rate:
            incident_type = "major_incident"
        elif high_latency:
            incident_type = "performance_incident"
        elif high_error_rate:
            incident_type = "reliability_incident"
        elif event.get("severity") in ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"]:
            incident_type = "technical_incident"
        else:
            incident_type = "operational_event"

        event["incident_type"] = incident_type
        event["requires_attention"] = incident_type != "operational_event"

        return [event]


def build_event_schema() -> str:
    return ",".join(
        [
            "processed_at:TIMESTAMP",
            "event_timestamp:TIMESTAMP",
            "severity:STRING",
            "log_name:STRING",
            "resource_type:STRING",
            "project_id:STRING",
            "service_name:STRING",
            "environment:STRING",
            "event_type:STRING",
            "latency_ms:INTEGER",
            "error_rate:FLOAT",
            "region:STRING",
            "correlation_id:STRING",
            "aiops_candidate:BOOLEAN",
            "incident_type:STRING",
            "requires_attention:BOOLEAN",
        ]
    )


def build_deadletter_schema() -> str:
    return ",".join(
        [
            "processed_at:TIMESTAMP",
            "error_message:STRING",
            "raw_message:STRING",
        ]
    )


def run() -> None:
    parser = argparse.ArgumentParser()

    DATAFLOW_SERVICE_ACCOUNT = (
        "capacitacion-ejecucion-ia-4668@asteci-capacitacion-ia.iam.gserviceaccount.com"
    )

    parser.add_argument(
        "--subscription",
        required=True,
        help="Subscription completa: projects/<project>/subscriptions/<subscription>",
    )
    parser.add_argument(
        "--output_table",
        required=True,
        help="Tabla BigQuery destino: <project>:<dataset>.<table>",
    )
    parser.add_argument(
        "--deadletter_table",
        required=True,
        help="Tabla BigQuery dead letter: <project>:<dataset>.<table>",
    )
    parser.add_argument(
        "--service_account_email",
        default=DATAFLOW_SERVICE_ACCOUNT,
        help="Service Account usada por los workers de Dataflow",
    )

    known_args, pipeline_args = parser.parse_known_args()

    # Pasamos la service account a Apache Beam / Dataflow.
    # Si no se pasa explícitamente, Dataflow usaría la Compute Default SA.
    if known_args.service_account_email:
        pipeline_args.extend(
            [
                "--service_account_email",
                known_args.service_account_email,
            ]
        )

    options = PipelineOptions(
        pipeline_args,
        save_main_session=True,
        streaming=True,
    )

    with beam.Pipeline(options=options) as pipeline:
        parsed = (
            pipeline
            | "Read from PubSub"
            >> beam.io.ReadFromPubSub(subscription=known_args.subscription)
            | "Parse and normalize"
            >> beam.ParDo(ParseAndNormalizeLogEntry()).with_outputs(
                ParseAndNormalizeLogEntry.DEADLETTER_TAG,
                main="events",
            )
        )

        events = parsed.events
        deadletter = parsed.deadletter

        classified_events = (
            events
            | "Classify operational event"
            >> beam.ParDo(ClassifyOperationalEvent())
        )

        classified_events | "Write events to BigQuery" >> beam.io.WriteToBigQuery(
            known_args.output_table,
            schema=build_event_schema(),
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        )
        print(known_args)
        deadletter | "Write deadletter to BigQuery" >> beam.io.WriteToBigQuery(
            known_args.deadletter_table,
            schema=build_deadletter_schema(),
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        )


if __name__ == "__main__":
    run()