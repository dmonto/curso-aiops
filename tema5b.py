import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "incident_classifier.joblib")


INCIDENT_PATTERNS = {
    "performance": [
        "latencia elevada en el servicio",
        "p95 por encima del umbral de SLO",
        "respuesta lenta detectada por usuarios",
        "degradacion progresiva del tiempo de respuesta",
        "timeout intermitente por saturacion de peticiones",
    ],
    "application_error": [
        "errores 500 en la API",
        "excepcion no controlada en backend",
        "fallo de validacion en peticiones",
        "error interno despues de procesar request",
        "aumento de errores HTTP 5xx",
    ],
    "capacity": [
        "CPU alta en nodos de aplicacion",
        "memoria cerca del limite",
        "disco casi lleno",
        "cola de procesamiento acumulada",
        "saturacion de workers",
    ],
    "dependency": [
        "fallo conectando con base de datos",
        "timeout contra servicio externo",
        "DNS no resuelve dependencia",
        "error de conexion con cola de mensajes",
        "API externa devuelve errores",
    ],
    "deployment": [
        "fallo despues de despliegue",
        "rollback requerido por nueva version",
        "cambio de configuracion rompe servicio",
        "release reciente genera errores",
        "nueva version aumenta latencia",
    ],
    "security": [
        "multiples intentos de login fallidos",
        "trafico sospechoso desde varias regiones",
        "posible abuso de credenciales",
        "incremento anomalo de peticiones no autorizadas",
        "patron de acceso no habitual",
    ],
    "noise": [
        "alerta duplicada ya cubierta",
        "evento temporal sin impacto",
        "falsa alarma de monitorizacion",
        "alerta resuelta automaticamente",
        "notificacion sin accion requerida",
    ],
}


def build_synthetic_incident_dataset(n_rows: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    labels = list(INCIDENT_PATTERNS.keys())
    services = [
        "api-login",
        "api-orders",
        "api-payments",
        "api-checkout",
        "worker-billing",
        "frontend-web",
        "db-main",
    ]
    environments = ["dev", "test", "prod"]
    regions = ["europe-west1", "europe-southwest1", "us-central1"]

    rows = []

    for i in range(n_rows):
        label = rng.choice(
            labels,
            p=[0.18, 0.20, 0.16, 0.16, 0.12, 0.08, 0.10],
        )

        service = rng.choice(services)
        environment = rng.choice(environments, p=[0.15, 0.20, 0.65])
        region = rng.choice(regions)

        summary = rng.choice(INCIDENT_PATTERNS[label])

        # Valores base
        latency_p95_ms = rng.normal(350, 120)
        error_rate = rng.normal(0.02, 0.01)
        cpu_avg = rng.normal(50, 15)
        memory_avg = rng.normal(55, 12)
        deploy_recent = rng.choice([0, 1], p=[0.8, 0.2])
        open_alerts_30m = rng.poisson(1)

        # Ajustes por tipo de incidente
        if label == "performance":
            latency_p95_ms += rng.normal(1200, 300)
            open_alerts_30m += rng.integers(1, 4)

        elif label == "application_error":
            error_rate += rng.normal(0.18, 0.05)
            latency_p95_ms += rng.normal(200, 80)

        elif label == "capacity":
            cpu_avg += rng.normal(40, 8)
            memory_avg += rng.normal(30, 8)
            open_alerts_30m += rng.integers(1, 5)

        elif label == "dependency":
            latency_p95_ms += rng.normal(700, 200)
            error_rate += rng.normal(0.08, 0.03)

        elif label == "deployment":
            deploy_recent = 1
            error_rate += rng.normal(0.10, 0.04)
            latency_p95_ms += rng.normal(300, 120)

        elif label == "security":
            error_rate += rng.normal(0.04, 0.02)
            open_alerts_30m += rng.integers(2, 7)

        elif label == "noise":
            latency_p95_ms = rng.normal(180, 50)
            error_rate = rng.normal(0.005, 0.003)
            cpu_avg = rng.normal(35, 10)
            memory_avg = rng.normal(40, 10)
            open_alerts_30m = rng.choice([0, 1])

        rows.append(
            {
                "incident_id": f"INC{i:05d}",
                "summary": summary,
                "service": service,
                "environment": environment,
                "region": region,
                "latency_p95_ms": max(10, round(float(latency_p95_ms), 2)),
                "error_rate": max(0, round(float(error_rate), 4)),
                "cpu_avg": min(100, max(1, round(float(cpu_avg), 2))),
                "memory_avg": min(100, max(1, round(float(memory_avg), 2))),
                "deploy_recent": int(deploy_recent),
                "open_alerts_30m": int(open_alerts_30m),
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def build_model() -> Pipeline:
    text_feature = "summary"
    categorical_features = ["service", "environment", "region"]
    numeric_features = [
        "latency_p95_ms",
        "error_rate",
        "cpu_avg",
        "memory_avg",
        "deploy_recent",
        "open_alerts_30m",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
                text_feature,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    feature_columns = [
        "summary",
        "service",
        "environment",
        "region",
        "latency_p95_ms",
        "error_rate",
        "cpu_avg",
        "memory_avg",
        "deploy_recent",
        "open_alerts_30m",
    ]

    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nDistribución de clases:")
    print(y.value_counts(normalize=True).round(3))

    print("\nMatriz de confusión:")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred, digits=3))

    return model


def classify_new_incidents(model: Pipeline) -> None:
    new_incidents = pd.DataFrame(
        [
            {
                "summary": "p95 por encima del umbral de SLO en api-payments",
                "service": "api-payments",
                "environment": "prod",
                "region": "europe-west1",
                "latency_p95_ms": 1850,
                "error_rate": 0.04,
                "cpu_avg": 67,
                "memory_avg": 71,
                "deploy_recent": 0,
                "open_alerts_30m": 3,
            },
            {
                "summary": "fallo despues de despliegue con errores 500",
                "service": "api-checkout",
                "environment": "prod",
                "region": "europe-southwest1",
                "latency_p95_ms": 900,
                "error_rate": 0.21,
                "cpu_avg": 62,
                "memory_avg": 66,
                "deploy_recent": 1,
                "open_alerts_30m": 5,
            },
            {
                "summary": "alerta duplicada ya cubierta por incidente abierto",
                "service": "frontend-web",
                "environment": "prod",
                "region": "europe-west1",
                "latency_p95_ms": 160,
                "error_rate": 0.004,
                "cpu_avg": 31,
                "memory_avg": 37,
                "deploy_recent": 0,
                "open_alerts_30m": 1,
            },
        ]
    )

    predictions = model.predict(new_incidents)
    probabilities = model.predict_proba(new_incidents)
    classes = model.named_steps["classifier"].classes_

    result = new_incidents.copy()
    result["predicted_type"] = predictions
    result["confidence"] = probabilities.max(axis=1).round(3)

    # Añadimos una recomendación operativa sencilla
    result["routing_team"] = result["predicted_type"].map(
        {
            "performance": "sre-platform",
            "application_error": "backend-team",
            "capacity": "cloud-ops",
            "dependency": "platform-integrations",
            "deployment": "release-engineering",
            "security": "security-ops",
            "noise": "no-action-review",
        }
    )

    result["recommended_action"] = result["predicted_type"].map(
        {
            "performance": "revisar SLO, latencia p95 y saturacion",
            "application_error": "revisar logs de aplicacion y errores 5xx",
            "capacity": "revisar capacidad, autoscaling y cuotas",
            "dependency": "validar dependencia externa o base de datos",
            "deployment": "comparar con ultimo release y valorar rollback",
            "security": "revisar autenticacion, origenes y actividad sospechosa",
            "noise": "deduplicar o cerrar si no hay impacto",
        }
    )

    print("\nPredicciones sobre nuevos incidentes:")
    print(
        result[
            [
                "summary",
                "service",
                "environment",
                "predicted_type",
                "confidence",
                "routing_team",
                "recommended_action",
            ]
        ].to_string(index=False)
    )

    print("\nProbabilidades por clase:")
    proba_df = pd.DataFrame(probabilities, columns=classes)
    print(proba_df.round(3).to_string(index=False))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = build_synthetic_incident_dataset()
    print("Muestra del dataset:")
    print(df.head().to_string(index=False))

    model = train_and_evaluate(df)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")

    classify_new_incidents(model)


if __name__ == "__main__":
    main()