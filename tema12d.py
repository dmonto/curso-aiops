import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig, HttpOptions
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

GENERATIVE_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

BQ_DATASET = os.getenv("BQ_DATASET", "aiops_lab")
BQ_RAG_TABLE = os.getenv("BQ_RAG_TABLE", "technical_docs_chunks")
BQ_RAG_AUDIT_TABLE = os.getenv("BQ_RAG_AUDIT_TABLE", "rag_query_audit")

PROMPT_VERSION = "rag-technical-docs-v1"


if not PROJECT_ID:
    raise RuntimeError("Falta PROJECT_ID")

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "used_sources": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "doc_id": {"type": "STRING"},
                    "title": {"type": "STRING"},
                    "chunk_id": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["doc_id", "title", "chunk_id", "reason"],
            },
        },
        "confidence": {"type": "NUMBER"},
        "missing_information": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "needs_human_review": {"type": "BOOLEAN"},
        "suggested_next_steps": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "answer",
        "used_sources",
        "confidence",
        "missing_information",
        "needs_human_review",
        "suggested_next_steps",
    ],
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def get_genai_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def redact_sensitive_text(text: str) -> str:
    redacted = text
    redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED_TOKEN]", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key=)[A-Za-z0-9._\-]+", r"\1[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"(?i)(password=)[^&\s]+", r"\1[REDACTED_PASSWORD]", redacted)
    redacted = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]", redacted)
    return redacted


def ensure_tables() -> None:
    bq = get_bq_client()
    dataset_id = f"{PROJECT_ID}.{BQ_DATASET}"

    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "EU"
    bq.create_dataset(dataset, exists_ok=True)

    chunks_table = bigquery.Table(
        f"{dataset_id}.{BQ_RAG_TABLE}",
        schema=[
            bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source_path", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("chunk_index", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("tags_json", "STRING"),
            bigquery.SchemaField("sensitivity", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ],
    )
    bq.create_table(chunks_table, exists_ok=True)

    audit_table = bigquery.Table(
        f"{dataset_id}.{BQ_RAG_AUDIT_TABLE}",
        schema=[
            bigquery.SchemaField("query_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("question", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("top_k", "INTEGER"),
            bigquery.SchemaField("model_id", "STRING"),
            bigquery.SchemaField("embedding_model_id", "STRING"),
            bigquery.SchemaField("prompt_version", "STRING"),
            bigquery.SchemaField("confidence", "FLOAT"),
            bigquery.SchemaField("needs_human_review", "BOOLEAN"),
            bigquery.SchemaField("retrieved_sources_json", "STRING"),
            bigquery.SchemaField("answer_json", "STRING"),
        ],
    )
    bq.create_table(audit_table, exists_ok=True)


def seed_sample_docs(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "runbook_checkout_500.md": """### Runbook: errores 500 en checkout-api

Servicio principal: checkout-api.
Dependencias frecuentes: cloud-sql-orders, orders-events, payment-gateway.

Síntomas:
- Aumento de HTTP 500 en POST /checkout.
- Latencia elevada en operaciones de pago.
- Errores de conexión contra cloud-sql-orders.
- Backlog creciente en orders-events.

Diagnóstico inicial:
1. Revisar error rate 5xx de checkout-api en Cloud Monitoring.
2. Revisar logs ERROR de checkout-api en los últimos 30 minutos.
3. Validar conexiones activas en cloud-sql-orders.
4. Comprobar si hubo despliegue reciente de checkout-api.
5. Revisar backlog de Pub/Sub en orders-events.

Criterio de severidad:
- P1 si el pago está caído para la mayoría de usuarios.
- P2 si hay degradación relevante pero parcial.
- P3 si afecta a un subconjunto reducido.

Acciones seguras:
- Generar resumen técnico.
- Escalar a sre-platform si se confirma impacto P1/P2.
- Preparar rollback solo si hay evidencia de regresión tras despliegue.

Restricciones:
- No ejecutar rollback sin aprobación humana.
- No modificar pool de conexiones en producción sin revisión SRE.
""",
        "runbook_pubsub_backlog.md": """### Runbook: backlog elevado en Pub/Sub

Servicio principal: orders-consumer.
Dependencias: orders-events, checkout-api, billing-worker.

Síntomas:
- Aumento rápido de mensajes pendientes.
- El consumidor procesa lentamente.
- Errores repetidos de timeout.
- Retries acumulados en la suscripción.

Diagnóstico inicial:
1. Revisar backlog de la suscripción de orders-events.
2. Revisar errores recientes de orders-consumer.
3. Comprobar latencia de procesamiento por mensaje.
4. Validar si hay errores en dependencias downstream.
5. Revisar cambios recientes de configuración o despliegue.

Criterio de escalado:
- Escalar a sre-messaging si el backlog crece durante más de 15 minutos.
- Escalar a billing-ops si el bloqueo afecta a facturación.
- Escalar a sre-platform si hay caída transversal.

Acciones seguras:
- Informar de backlog y tendencia.
- Preparar informe técnico.
- No cambiar concurrencia ni ack deadline sin revisión.
""",
        "runbook_cloudsql_connections.md": """### Runbook: saturación de conexiones Cloud SQL

Servicio principal: cloud-sql-orders.
Servicios consumidores habituales: checkout-api, billing-worker, admin-api.

Síntomas:
- Mensajes "too many connections".
- Latencia elevada en consultas.
- Timeouts de aplicaciones consumidoras.
- Retry storm desde servicios upstream.

Diagnóstico inicial:
1. Revisar conexiones activas y límite configurado.
2. Identificar servicios con mayor número de conexiones.
3. Revisar cambios recientes en pool de conexiones.
4. Comprobar si algún despliegue aumentó concurrencia.
5. Revisar queries lentas y bloqueos.

Hipótesis habituales:
- Pool de conexiones mal dimensionado.
- Retry sin backoff suficiente.
- Despliegue con aumento de concurrencia.
- Consultas lentas que retienen conexiones.

Acciones seguras:
- Reducir ruido de retries desde el servicio consumidor si existe procedimiento aprobado.
- Escalar a DBA o SRE si conexiones superan el 90%.
- No aumentar límites sin revisar impacto y coste.
""",
        "procedimiento_storage_access_denied.md": """### Procedimiento: errores PermissionDenied en Cloud Storage

Servicios habituales: procesos batch, indexadores de documentos, asistentes RAG.

Síntomas:
- Error PermissionDenied al leer objetos.
- Fallo al cargar documentación técnica.
- El proceso puede listar el bucket pero no leer objetos.
- El usuario puede ver el bucket en consola pero el script falla.

Diagnóstico inicial:
1. Confirmar identidad efectiva usada por el script.
2. Revisar si se usa Application Default Credentials.
3. Validar permisos Storage Object Viewer o Storage Object Admin según necesidad.
4. Revisar si el bucket tiene restricciones adicionales.
5. Confirmar que el objeto existe y la ruta es correcta.

Buenas prácticas:
- Usar roles mínimos.
- No usar Owner para resolver problemas de lectura.
- Separar permisos de lectura y escritura.
- Revisar logs de auditoría si el error persiste.
""",
    }

    for filename, content in samples.items():
        path = docs_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def read_documents(docs_dir: Path) -> List[Dict[str, str]]:
    docs = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in [".md", ".txt"]:
            continue

        content = redact_sensitive_text(path.read_text(encoding="utf-8"))
        title = extract_title(content) or path.stem.replace("_", " ").title()
        doc_id = stable_id(str(path), length=12)

        docs.append(
            {
                "doc_id": doc_id,
                "title": title,
                "source_path": str(path),
                "content": content,
            }
        )

    return docs


def extract_title(content: str) -> Optional[str]:
    for line in content.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            return clean.lstrip("#").strip()
    return None


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            if len(paragraph) > max_chars:
                for start in range(0, len(paragraph), max_chars - overlap):
                    chunks.append(paragraph[start:start + max_chars])
                current = ""
            else:
                current = paragraph

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            overlapped.append(chunk)
        else:
            prefix = chunks[idx - 1][-overlap:]
            overlapped.append(f"{prefix}\n\n{chunk}")

    return overlapped


def embed_text(text: str, task_type: str, title: Optional[str] = None) -> List[float]:
    client = get_genai_client()

    config = EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM,
        title=title,
    )

    response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=[text],
        config=config,
    )

    if not response.embeddings:
        raise RuntimeError("No se recibió embedding.")

    return list(response.embeddings[0].values)


def reset_chunks_table() -> None:
    bq = get_bq_client()
    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_RAG_TABLE}`"
    bq.query(f"DELETE FROM {table_id} WHERE TRUE").result()


def ingest_documents(docs_dir: Path, reset: bool) -> None:
    ensure_tables()
    seed_sample_docs(docs_dir)

    if reset:
        reset_chunks_table()

    bq = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_RAG_TABLE}"

    documents = read_documents(docs_dir)

    if not documents:
        print(f"No se encontraron documentos .md o .txt en {docs_dir}")
        return

    rows = []

    for doc in documents:
        chunks = chunk_text(doc["content"])

        for idx, chunk in enumerate(chunks):
            chunk_id = stable_id(f"{doc['doc_id']}-{idx}-{chunk}", length=20)
            embedding = embed_text(chunk, task_type="RETRIEVAL_DOCUMENT", title=doc["title"])

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source_path": doc["source_path"],
                    "chunk_index": idx,
                    "content": chunk,
                    "embedding": embedding,
                    "tags_json": json.dumps(infer_tags(doc["title"], chunk), ensure_ascii=False),
                    "sensitivity": "internal",
                    "created_at": utc_now().isoformat(),
                }
            )

            print(f"Ingestado chunk {idx} de {doc['title']}")

            if len(rows) >= 100:
                insert_rows(table_id, rows)
                rows = []

    if rows:
        insert_rows(table_id, rows)

    print("Ingesta completada.")


def infer_tags(title: str, content: str) -> List[str]:
    text = f"{title} {content}".lower()
    tags = []

    for tag in [
        "checkout-api",
        "cloud-sql-orders",
        "orders-events",
        "orders-consumer",
        "billing-worker",
        "storage",
        "pubsub",
        "iam",
        "runbook",
    ]:
        if tag.lower() in text:
            tags.append(tag)

    return sorted(set(tags))


def insert_rows(table_id: str, rows: List[Dict[str, Any]]) -> None:
    bq = get_bq_client()
    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"Errores insertando chunks en BigQuery: {errors}")


def retrieve_with_bigquery_vector_search(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    bq = get_bq_client()
    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_RAG_TABLE}`"

    query = f"""
    SELECT
      base.chunk_id,
      base.doc_id,
      base.title,
      base.source_path,
      base.chunk_index,
      base.content,
      base.tags_json,
      base.sensitivity,
      distance
    FROM VECTOR_SEARCH(
      TABLE {table_id},
      'embedding',
      (SELECT @query_embedding AS embedding),
      top_k => @top_k,
      distance_type => 'COSINE'
    )
    ORDER BY distance ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
            bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
        ]
    )

    rows = list(bq.query(query, job_config=job_config).result())
    return [dict(row) for row in rows]


def retrieve_with_python_fallback(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    bq = get_bq_client()
    table_id = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_RAG_TABLE}`"

    rows = list(
        bq.query(
            f"""
            SELECT
              chunk_id,
              doc_id,
              title,
              source_path,
              chunk_index,
              content,
              tags_json,
              sensitivity,
              embedding
            FROM {table_id}
            """
        ).result()
    )

    q = np.array(query_embedding, dtype=float)
    q_norm = np.linalg.norm(q)

    scored = []
    for row in rows:
        emb = np.array(row["embedding"], dtype=float)
        denom = q_norm * np.linalg.norm(emb)
        similarity = float(np.dot(q, emb) / denom) if denom else 0.0
        distance = 1.0 - similarity

        item = dict(row)
        item["distance"] = distance
        scored.append(item)

    scored.sort(key=lambda x: x["distance"])
    return scored[:top_k]


def retrieve_context(question: str, top_k: int) -> List[Dict[str, Any]]:
    query_embedding = embed_text(question, task_type="RETRIEVAL_QUERY")

    try:
        return retrieve_with_bigquery_vector_search(query_embedding, top_k=top_k)
    except Exception as exc:
        print(f"VECTOR_SEARCH no disponible o falló. Usando fallback local. Detalle: {exc}")
        return retrieve_with_python_fallback(query_embedding, top_k=top_k)


def build_rag_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for idx, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"""
[FUENTE {idx}]
doc_id: {item.get("doc_id")}
chunk_id: {item.get("chunk_id")}
title: {item.get("title")}
source_path: {item.get("source_path")}
distance: {item.get("distance")}
content:
{item.get("content")}
""".strip()
        )

    return f"""
Actúa como asistente interno de soporte AIOps.

Debes responder a la pregunta usando únicamente las fuentes recuperadas.
No uses conocimiento externo si no está apoyado en las fuentes.
No inventes procedimientos, servicios ni comandos.
Si las fuentes no son suficientes, dilo claramente.
Incluye qué fuentes has usado.
Si hay una acción de rollback, escalado, cambio de configuración o IAM, marca revisión humana.
Devuelve solo JSON válido.

Pregunta:
{redact_sensitive_text(question)}

Fuentes recuperadas:
{chr(10).join(context_blocks)}
""".strip()


def generate_grounded_answer(question: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = get_genai_client()
    prompt = build_rag_prompt(question, retrieved)

    response = client.models.generate_content(
        model=GENERATIVE_MODEL_ID,
        contents=prompt,
        config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": ANSWER_SCHEMA,
        },
    )

    if not response.text:
        raise RuntimeError("Gemini no devolvió respuesta.")

    result = json.loads(response.text)

    if float(result.get("confidence", 0)) < 0.65:
        result["needs_human_review"] = True

    return result


def save_query_audit(
    question: str,
    top_k: int,
    retrieved: List[Dict[str, Any]],
    answer: Dict[str, Any],
) -> None:
    bq = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_RAG_AUDIT_TABLE}"

    query_id = stable_id(f"{question}-{utc_now().isoformat()}", length=20)

    retrieved_sources = [
        {
            "doc_id": item.get("doc_id"),
            "title": item.get("title"),
            "chunk_id": item.get("chunk_id"),
            "distance": item.get("distance"),
        }
        for item in retrieved
    ]

    row = {
        "query_id": query_id,
        "created_at": utc_now().isoformat(),
        "question": redact_sensitive_text(question),
        "top_k": top_k,
        "model_id": GENERATIVE_MODEL_ID,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "confidence": float(answer.get("confidence", 0)),
        "needs_human_review": bool(answer.get("needs_human_review", True)),
        "retrieved_sources_json": json.dumps(retrieved_sources, ensure_ascii=False),
        "answer_json": json.dumps(answer, ensure_ascii=False),
    }

    errors = bq.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Error guardando auditoría RAG: {errors}")


def ask_question(question: str, top_k: int, dry_run: bool) -> None:
    ensure_tables()

    retrieved = retrieve_context(question, top_k=top_k)

    if not retrieved:
        print("No se recuperó contexto documental.")
        return

    answer = generate_grounded_answer(question, retrieved)

    if not dry_run:
        save_query_audit(question, top_k, retrieved, answer)

    print("\n=== Respuesta RAG ===")
    print(answer.get("answer", ""))

    print("\n=== Fuentes usadas ===")
    for src in answer.get("used_sources", []):
        print(f"- {src.get('title')} | doc_id={src.get('doc_id')} | chunk_id={src.get('chunk_id')}")

    print("\n=== Próximos pasos sugeridos ===")
    for step in answer.get("suggested_next_steps", []):
        print(f"- {step}")

    print("\n=== Control ===")
    print(f"Confianza: {answer.get('confidence')}")
    print(f"Revisión humana: {answer.get('needs_human_review')}")

    missing = answer.get("missing_information", [])
    if missing:
        print("\n=== Información faltante ===")
        for item in missing:
            print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG aplicado a documentación técnica AIOps")
    parser.add_argument("--docs-dir", default="rag_docs", help="Carpeta con documentos .md o .txt")
    parser.add_argument("--ingest", action="store_true", help="Ingestar documentación")
    parser.add_argument("--reset", action="store_true", help="Vaciar tabla de chunks antes de ingestar")
    parser.add_argument("--ask", default=None, help="Pregunta técnica")
    parser.add_argument("--top-k", type=int, default=4, help="Número de chunks a recuperar")
    parser.add_argument("--dry-run", action="store_true", help="No guardar auditoría")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)

    if args.ingest:
        ingest_documents(docs_dir, reset=args.reset)

    if args.ask:
        ask_question(args.ask, top_k=args.top_k, dry_run=args.dry_run)

    if not args.ingest and not args.ask:
        parser.print_help()


if __name__ == "__main__":
    main()