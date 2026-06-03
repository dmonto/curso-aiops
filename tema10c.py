import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google.cloud import bigquery
from google.api_core.exceptions import NotFound


load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET", "aiops_cost")
BQ_LOCATION = os.getenv("BQ_LOCATION", "EU")

FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "14"))
FORECAST_CONFIDENCE_LEVEL = float(os.getenv("FORECAST_CONFIDENCE_LEVEL", "0.90"))
MONTHLY_BUDGET = float(os.getenv("MONTHLY_BUDGET", "1000"))

if not PROJECT_ID:
    raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT o PROJECT_ID en el .env")

client = bigquery.Client(project=PROJECT_ID)

SOURCE_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.daily_service_cost"
TRAINING_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.cost_forecast_training"
MODEL_ID = f"{PROJECT_ID}.{BQ_DATASET}.cost_forecast_arima"
FORECAST_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.cost_forecast_14d"

Path("outputs").mkdir(exist_ok=True)


def run_query(sql: str):
    job = client.query(sql, location=BQ_LOCATION)
    return job.result()


def table_exists(table_id: str) -> bool:
    try:
        client.get_table(table_id)
        return True
    except NotFound:
        return False


def model_exists(model_id: str) -> bool:
    try:
        client.get_model(model_id)
        return True
    except NotFound:
        return False


def require_source_table() -> None:
    if not table_exists(SOURCE_TABLE):
        raise RuntimeError(
            f"No existe la tabla {SOURCE_TABLE}. "
            "Ejecuta antes el ejercicio de análisis histórico de consumo."
        )


def create_training_table() -> None:
    sql = f"""
    CREATE OR REPLACE TABLE `{TRAINING_TABLE}` AS
    SELECT
      TIMESTAMP(usage_date) AS usage_ts,
      DATE(usage_date) AS usage_date,
      service,
      SUM(net_cost) AS net_cost
    FROM `{SOURCE_TABLE}`
    WHERE usage_date IS NOT NULL
      AND service IS NOT NULL
      AND net_cost IS NOT NULL
    GROUP BY
      usage_ts,
      usage_date,
      service
    HAVING net_cost >= 0
    ORDER BY
      service,
      usage_ts
    """
    run_query(sql)
    print(f"Tabla de entrenamiento creada: {TRAINING_TABLE}")


def show_training_summary() -> None:
    sql = f"""
    SELECT
      service,
      COUNT(*) AS n_days,
      MIN(usage_date) AS first_day,
      MAX(usage_date) AS last_day,
      ROUND(SUM(net_cost), 2) AS total_cost,
      ROUND(AVG(net_cost), 2) AS avg_daily_cost
    FROM `{TRAINING_TABLE}`
    GROUP BY service
    ORDER BY total_cost DESC
    """

    df = client.query(sql, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=False)

    print("\nResumen de series temporales")
    print(df.to_string(index=False))

    too_short = df[df["n_days"] < 30]
    if not too_short.empty:
        print(
            "\nAviso: hay series con menos de 30 días. "
            "El forecast puede ser poco estable para esos servicios."
        )


def train_arima_model() -> None:
    sql = f"""
    CREATE OR REPLACE MODEL `{MODEL_ID}`
    OPTIONS(
      MODEL_TYPE = 'ARIMA_PLUS',
      TIME_SERIES_TIMESTAMP_COL = 'usage_ts',
      TIME_SERIES_DATA_COL = 'net_cost',
      TIME_SERIES_ID_COL = 'service',
      DATA_FREQUENCY = 'DAILY',
      HORIZON = {FORECAST_HORIZON_DAYS},
      AUTO_ARIMA = TRUE,
      CLEAN_SPIKES_AND_DIPS = TRUE,
      ADJUST_STEP_CHANGES = TRUE
    ) AS
    SELECT
      usage_ts,
      service,
      net_cost
    FROM `{TRAINING_TABLE}`
    """

    print("\nEntrenando modelo ARIMA_PLUS en BigQuery ML...")
    print("Este paso puede tardar varios minutos según el histórico y número de series.")

    run_query(sql)

    if not model_exists(MODEL_ID):
        raise RuntimeError(f"No se ha encontrado el modelo tras el entrenamiento: {MODEL_ID}")

    print(f"Modelo creado: {MODEL_ID}")


def create_forecast_table() -> None:
    sql = f"""
    CREATE OR REPLACE TABLE `{FORECAST_TABLE}` AS
    SELECT
      DATE(forecast_timestamp) AS forecast_date,
      service,
      forecast_value AS raw_forecast_value,
      GREATEST(0, forecast_value) AS forecast_cost,
      GREATEST(0, prediction_interval_lower_bound) AS lower_bound_cost,
      GREATEST(0, prediction_interval_upper_bound) AS upper_bound_cost,
      standard_error
    FROM ML.FORECAST(
      MODEL `{MODEL_ID}`,
      STRUCT(
        {FORECAST_HORIZON_DAYS} AS horizon,
        {FORECAST_CONFIDENCE_LEVEL} AS confidence_level
      )
    )
    ORDER BY
      service,
      forecast_date
    """

    run_query(sql)
    print(f"Tabla de forecast creada: {FORECAST_TABLE}")


def get_arima_evaluation() -> pd.DataFrame:
    sql = f"""
    SELECT *
    FROM ML.ARIMA_EVALUATE(MODEL `{MODEL_ID}`)
    """
    df = client.query(sql, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=False)
    df.to_csv("outputs/bqml_arima_evaluation.csv", index=False)
    return df


def export_forecast() -> pd.DataFrame:
    sql = f"""
    SELECT
      forecast_date,
      service,
      ROUND(forecast_cost, 4) AS forecast_cost,
      ROUND(lower_bound_cost, 4) AS lower_bound_cost,
      ROUND(upper_bound_cost, 4) AS upper_bound_cost,
      ROUND(standard_error, 4) AS standard_error
    FROM `{FORECAST_TABLE}`
    ORDER BY
      forecast_date,
      service
    """

    df = client.query(sql, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=False)
    df.to_csv("outputs/bqml_cost_forecast_by_service.csv", index=False)
    return df


def calculate_month_projection(forecast_df: pd.DataFrame) -> dict:
    sql = f"""
    SELECT
      MAX(usage_date) AS last_actual_day,
      SUM(
        CASE
          WHEN usage_date >= DATE_TRUNC((SELECT MAX(usage_date) FROM `{TRAINING_TABLE}`), MONTH)
          THEN net_cost
          ELSE 0
        END
      ) AS actual_month_cost
    FROM `{TRAINING_TABLE}`
    """

    actual = client.query(sql, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=False).iloc[0]

    last_actual_day = pd.to_datetime(actual["last_actual_day"]).date()
    actual_month_cost = float(actual["actual_month_cost"])

    month_end = pd.Timestamp(last_actual_day) + pd.offsets.MonthEnd(0)
    remaining_days = (month_end.date() - last_actual_day).days

    daily_forecast = (
        forecast_df.groupby("forecast_date", as_index=False)["forecast_cost"]
        .sum()
        .sort_values("forecast_date")
    )

    daily_upper = (
        forecast_df.groupby("forecast_date", as_index=False)["upper_bound_cost"]
        .sum()
        .sort_values("forecast_date")
    )

    if remaining_days <= 0:
        forecast_remaining = 0.0
        upper_remaining = 0.0
    elif remaining_days <= FORECAST_HORIZON_DAYS:
        max_date = pd.Timestamp(last_actual_day) + pd.Timedelta(days=remaining_days)
        forecast_remaining = daily_forecast[
            pd.to_datetime(daily_forecast["forecast_date"]) <= max_date
        ]["forecast_cost"].sum()

        upper_remaining = daily_upper[
            pd.to_datetime(daily_upper["forecast_date"]) <= max_date
        ]["upper_bound_cost"].sum()
    else:
        avg_daily_forecast = daily_forecast["forecast_cost"].mean()
        avg_daily_upper = daily_upper["upper_bound_cost"].mean()
        forecast_remaining = avg_daily_forecast * remaining_days
        upper_remaining = avg_daily_upper * remaining_days

    projected_month_end = actual_month_cost + forecast_remaining
    projected_month_end_upper = actual_month_cost + upper_remaining

    if projected_month_end_upper >= MONTHLY_BUDGET * 1.10:
        risk = "ALTO"
    elif projected_month_end >= MONTHLY_BUDGET:
        risk = "MEDIO"
    else:
        risk = "BAJO"

    return {
        "last_actual_day": str(last_actual_day),
        "remaining_days": remaining_days,
        "actual_month_cost": actual_month_cost,
        "forecast_remaining": float(forecast_remaining),
        "projected_month_end": float(projected_month_end),
        "projected_month_end_upper": float(projected_month_end_upper),
        "monthly_budget": MONTHLY_BUDGET,
        "risk": risk,
    }


def plot_forecast(forecast_df: pd.DataFrame) -> None:
    history_sql = f"""
    SELECT
      usage_date,
      SUM(net_cost) AS actual_cost
    FROM `{TRAINING_TABLE}`
    GROUP BY usage_date
    ORDER BY usage_date
    """

    history_df = client.query(history_sql, location=BQ_LOCATION).to_dataframe(create_bqstorage_client=False)
    history_df["usage_date"] = pd.to_datetime(history_df["usage_date"])

    daily_forecast = (
        forecast_df.groupby("forecast_date", as_index=False)
        .agg(
            forecast_cost=("forecast_cost", "sum"),
            lower_bound_cost=("lower_bound_cost", "sum"),
            upper_bound_cost=("upper_bound_cost", "sum"),
        )
        .sort_values("forecast_date")
    )
    daily_forecast["forecast_date"] = pd.to_datetime(daily_forecast["forecast_date"])

    plt.figure(figsize=(11, 5))

    recent_history = history_df.tail(60)

    plt.plot(
        recent_history["usage_date"],
        recent_history["actual_cost"],
        label="Coste histórico",
    )

    plt.plot(
        daily_forecast["forecast_date"],
        daily_forecast["forecast_cost"],
        marker="o",
        label="Forecast",
    )

    plt.fill_between(
        daily_forecast["forecast_date"],
        daily_forecast["lower_bound_cost"],
        daily_forecast["upper_bound_cost"],
        alpha=0.2,
        label="Intervalo de confianza",
    )

    plt.title("Forecast de gasto con BigQuery ML")
    plt.xlabel("Fecha")
    plt.ylabel("Coste neto diario")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/bqml_cost_forecast.png", dpi=140)


def print_service_summary(forecast_df: pd.DataFrame) -> None:
    summary = (
        forecast_df.groupby("service", as_index=False)
        .agg(
            forecast_14d=("forecast_cost", "sum"),
            upper_14d=("upper_bound_cost", "sum"),
            avg_daily_forecast=("forecast_cost", "mean"),
        )
        .sort_values("forecast_14d", ascending=False)
    )

    summary.to_csv("outputs/bqml_cost_forecast_service_summary.csv", index=False)

    print("\nForecast por servicio")
    print(summary.to_string(index=False))


def main() -> None:
    require_source_table()
    create_training_table()
    show_training_summary()
    train_arima_model()
    create_forecast_table()

    eval_df = get_arima_evaluation()
    forecast_df = export_forecast()

    print("\nEvaluación ARIMA exportada a:")
    print("- outputs/bqml_arima_evaluation.csv")

    print_service_summary(forecast_df)

    projection = calculate_month_projection(forecast_df)

    pd.DataFrame([projection]).to_csv(
        "outputs/bqml_month_projection.csv",
        index=False,
    )

    plot_forecast(forecast_df)

    print("\nEstimación de cierre mensual")
    print(f"Último día real disponible        : {projection['last_actual_day']}")
    print(f"Días restantes del mes            : {projection['remaining_days']}")
    print(f"Coste acumulado del mes           : {projection['actual_month_cost']:.2f}")
    print(f"Coste restante previsto           : {projection['forecast_remaining']:.2f}")
    print(f"Cierre mensual estimado           : {projection['projected_month_end']:.2f}")
    print(f"Cierre mensual límite superior    : {projection['projected_month_end_upper']:.2f}")
    print(f"Presupuesto mensual               : {projection['monthly_budget']:.2f}")
    print(f"Riesgo predictivo                 : {projection['risk']}")

    print("\nFicheros generados:")
    print("- outputs/bqml_cost_forecast_by_service.csv")
    print("- outputs/bqml_cost_forecast_service_summary.csv")
    print("- outputs/bqml_month_projection.csv")
    print("- outputs/bqml_arima_evaluation.csv")
    print("- outputs/bqml_cost_forecast.png")


if __name__ == "__main__":
    main()