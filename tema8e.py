import os
from pathlib import Path

from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


PIPELINE_PACKAGE_PATH = "aiops_latency_mlops_pipeline.json"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
    ],
)
def build_dataset(output_dataset: Output[Dataset]) -> None:
    import pandas as pd

    rows = [
        ["checkout-api", "europe-west1", "prod", 1200, 0.3, 42, 58, 3, 0, 210],
        ["checkout-api", "europe-west1", "prod", 2500, 0.8, 55, 62, 3, 0, 285],
        ["checkout-api", "europe-west1", "prod", 4300, 1.4, 68, 71, 4, 0, 390],
        ["checkout-api", "europe-west1", "prod", 6200, 2.5, 79, 78, 4, 1, 610],
        ["checkout-api", "europe-west1", "prod", 8500, 4.2, 88, 84, 5, 1, 890],
        ["checkout-api", "europe-west1", "prod", 9600, 5.1, 93, 89, 5, 1, 1120],
        ["payment-api", "europe-west1", "prod", 900, 0.2, 38, 51, 2, 0, 180],
        ["payment-api", "europe-west1", "prod", 1600, 0.7, 48, 57, 2, 0, 245],
        ["payment-api", "europe-west1", "prod", 2900, 1.2, 63, 66, 3, 0, 330],
        ["payment-api", "europe-west1", "prod", 4800, 2.8, 76, 75, 3, 1, 570],
        ["payment-api", "europe-west1", "prod", 6100, 3.9, 85, 82, 4, 1, 760],
        ["payment-api", "europe-west1", "prod", 7400, 5.4, 91, 87, 4, 1, 980],
        ["catalog-api", "europe-west1", "prod", 1800, 0.1, 35, 44, 2, 0, 140],
        ["catalog-api", "europe-west1", "prod", 3600, 0.4, 49, 53, 2, 0, 190],
        ["catalog-api", "europe-west1", "prod", 5200, 0.9, 61, 64, 3, 0, 265],
        ["catalog-api", "europe-west1", "prod", 7900, 1.8, 73, 72, 3, 0, 410],
        ["catalog-api", "europe-west1", "prod", 10300, 2.7, 83, 80, 4, 1, 650],
        ["catalog-api", "europe-west1", "prod", 12100, 3.5, 90, 86, 4, 1, 830],
        ["orders-api", "europe-west1", "prod", 1500, 0.4, 44, 49, 2, 0, 220],
        ["orders-api", "europe-west1", "prod", 2700, 0.9, 57, 61, 2, 0, 310],
        ["orders-api", "europe-west1", "prod", 4100, 1.6, 70, 70, 3, 0, 455],
        ["orders-api", "europe-west1", "prod", 6900, 3.1, 82, 79, 3, 1, 720],
        ["orders-api", "europe-west1", "prod", 8300, 4.3, 89, 85, 4, 1, 940],
        ["orders-api", "europe-west1", "prod", 9700, 5.8, 95, 90, 4, 1, 1230],
        ["checkout-api", "us-central1", "prod", 2200, 0.6, 52, 60, 3, 0, 300],
        ["checkout-api", "us-central1", "prod", 5700, 2.1, 77, 76, 4, 1, 640],
        ["payment-api", "us-central1", "prod", 2500, 0.8, 56, 63, 3, 0, 310],
        ["payment-api", "us-central1", "prod", 6600, 3.7, 87, 84, 4, 1, 850],
        ["catalog-api", "us-central1", "prod", 4400, 0.7, 58, 62, 3, 0, 250],
        ["catalog-api", "us-central1", "prod", 9900, 2.4, 81, 79, 4, 1, 620],
        ["orders-api", "us-central1", "prod", 3900, 1.4, 69, 68, 3, 0, 430],
        ["orders-api", "us-central1", "prod", 8800, 4.9, 92, 88, 4, 1, 1080],
    ]

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

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_dataset.path, index=False)

    output_dataset.metadata["rows"] = len(df)
    output_dataset.metadata["target"] = "latency_p95_ms"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
    ],
)
def train_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    train_metrics: Output[Metrics],
) -> None:
    import os

    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(input_dataset.path)

    feature_columns = [
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

    target_column = "latency_p95_ms"

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    categorical_features = ["service", "region", "environment"]
    numeric_features = [
        "request_count",
        "error_rate",
        "cpu_percent",
        "memory_percent",
        "instances",
        "is_after_deployment",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
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

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    os.makedirs(output_model.path, exist_ok=True)

    # Nombre esperado por los contenedores prebuilt de sklearn en Vertex AI.
    model_file = os.path.join(output_model.path, "model.joblib")
    joblib.dump(pipeline, model_file)

    output_model.metadata["framework"] = "scikit-learn"
    output_model.metadata["target"] = target_column
    output_model.metadata["mae"] = float(mae)
    output_model.metadata["rmse"] = float(rmse)
    output_model.metadata["r2"] = float(r2)

    train_metrics.log_metric("mae", float(mae))
    train_metrics.log_metric("rmse", float(rmse))
    train_metrics.log_metric("r2", float(r2))


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
    ],
)
def evaluate_model(
    input_dataset: Input[Dataset],
    trained_model: Input[Model],
    mae_threshold: float,
) -> str:
    import os

    import joblib
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_dataset.path)

    feature_columns = [
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

    target_column = "latency_p95_ms"

    X = df[feature_columns]
    y = df[target_column]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model_file = os.path.join(trained_model.path, "model.joblib")
    model = joblib.load(model_file)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"MAE calculado: {mae}")
    print(f"Umbral MAE: {mae_threshold}")

    if mae <= mae_threshold:
        print("Modelo aprobado para registro.")
        return "approved"

    print("Modelo rechazado. No se registra.")
    return "rejected"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform==1.74.0",
    ],
)
def register_model(
    project_id: str,
    region: str,
    model_display_name: str,
    trained_model: Input[Model],
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=region,
    )

    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=trained_model.uri,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
        ),
        sync=True,
    )

    print(f"Modelo registrado: {uploaded_model.resource_name}")

    return uploaded_model.resource_name


@dsl.pipeline(
    name="aiops-latency-mlops-pipeline",
)
def aiops_latency_mlops_pipeline(
    project_id: str,
    region: str,
    model_display_name: str,
    mae_threshold: float = 140.0,
) -> None:
    dataset_task = build_dataset()

    train_task = train_model(
        input_dataset=dataset_task.outputs["output_dataset"],
    )

    eval_task = evaluate_model(
        input_dataset=dataset_task.outputs["output_dataset"],
        trained_model=train_task.outputs["output_model"],
        mae_threshold=mae_threshold,
    )

    with dsl.If(eval_task.output == "approved"):
        register_model(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            trained_model=train_task.outputs["output_model"],
        )


def compile_and_run_pipeline() -> None:
    project_id = require_env("PROJECT_ID")
    region = os.getenv("VERTEX_LOCATION", "europe-west1")
    bucket_name = require_env("BUCKET_NAME")

    if bucket_name.startswith("gs://"):
        staging_bucket = bucket_name.rstrip("/")
    else:
        staging_bucket = f"gs://{bucket_name.strip('/')}"

    pipeline_root = f"{staging_bucket}/vertex-pipelines/aiops-latency-mlops"
    service_account = os.getenv("VERTEX_SERVICE_ACCOUNT")

    compiler.Compiler().compile(
        pipeline_func=aiops_latency_mlops_pipeline,
        package_path=PIPELINE_PACKAGE_PATH,
    )

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
    )

    job = aiplatform.PipelineJob(
        display_name="aiops-latency-mlops-pipeline-run",
        template_path=PIPELINE_PACKAGE_PATH,
        pipeline_root=pipeline_root,
        parameter_values={
            "project_id": project_id,
            "region": region,
            "model_display_name": "aiops-latency-regressor",
            "mae_threshold": 140.0,
        },
        enable_caching=False,
    )

    submit_kwargs = {}

    if service_account:
        submit_kwargs["service_account"] = service_account

    job.submit(**submit_kwargs)

    print("Pipeline enviado a Vertex AI.")
    print(f"Project : {project_id}")
    print(f"Region  : {region}")
    print(f"Root    : {pipeline_root}")
    print(f"Package : {Path(PIPELINE_PACKAGE_PATH).resolve()}")


if __name__ == "__main__":
    compile_and_run_pipeline()