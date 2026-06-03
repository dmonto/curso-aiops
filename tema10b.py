import os
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google.cloud import bigquery
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET", "aiops_cost")
MONTHLY_BUDGET = float(os.getenv("MONTHLY_BUDGET", "1000"))

if not PROJECT_ID:
    raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT o PROJECT_ID en el .env")

SOURCE_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.daily_service_cost"

client = bigquery.Client(project=PROJECT_ID)
Path("outputs").mkdir(exist_ok=True)


def load_cost_data() -> pd.DataFrame:
    sql = f"""
    SELECT
      usage_date,
      service,
      net_cost,
      total_cost,
      service_share
    FROM `{SOURCE_TABLE}`
    ORDER BY usage_date, service
    """

    df = client.query(sql).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        raise RuntimeError(f"No hay datos en {SOURCE_TABLE}")

    df["usage_date"] = pd.to_datetime(df["usage_date"])
    df["net_cost"] = df["net_cost"].astype(float)
    df["total_cost"] = df["total_cost"].astype(float)
    df["service_share"] = df["service_share"].astype(float)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["service", "usage_date"]).copy()

    df["day_of_week"] = df["usage_date"].dt.dayofweek
    df["day_of_month"] = df["usage_date"].dt.day
    df["month"] = df["usage_date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    min_date = df["usage_date"].min()
    df["trend_index"] = (df["usage_date"] - min_date).dt.days

    grouped = df.groupby("service", group_keys=False)

    df["cost_lag_1"] = grouped["net_cost"].shift(1)
    df["cost_lag_2"] = grouped["net_cost"].shift(2)
    df["cost_lag_7"] = grouped["net_cost"].shift(7)

    df["rolling_mean_7"] = grouped["net_cost"].apply(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).mean()
    )

    df["rolling_std_7"] = grouped["net_cost"].apply(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).std()
    )

    df["rolling_std_7"] = df["rolling_std_7"].fillna(0)

    # Eliminamos filas iniciales sin histórico suficiente.
    df = df.dropna(
        subset=[
            "cost_lag_1",
            "cost_lag_2",
            "cost_lag_7",
            "rolling_mean_7",
        ]
    ).copy()

    return df


def train_model(df: pd.DataFrame):
    max_date = df["usage_date"].max()
    split_date = max_date - pd.Timedelta(days=14)

    train_df = df[df["usage_date"] <= split_date].copy()
    test_df = df[df["usage_date"] > split_date].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError(
            "No hay suficiente histórico para separar entrenamiento y test."
        )

    target = "net_cost"

    numeric_features = [
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "trend_index",
        "cost_lag_1",
        "cost_lag_2",
        "cost_lag_7",
        "rolling_mean_7",
        "rolling_std_7",
        "service_share",
    ]

    categorical_features = ["service"]

    features = numeric_features + categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(train_df[features], train_df[target])

    predictions = pipeline.predict(test_df[features])

    mae = mean_absolute_error(test_df[target], predictions)
    rmse = mean_squared_error(test_df[target], predictions) ** 0.5

    test_result = test_df[
        ["usage_date", "service", "net_cost"]
    ].copy()
    test_result["predicted_cost"] = predictions
    test_result["abs_error"] = (
        test_result["net_cost"] - test_result["predicted_cost"]
    ).abs()

    return pipeline, features, test_result, mae, rmse


def build_future_rows(df_original: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Genera filas futuras por servicio usando como punto de partida
    el histórico real más reciente.

    Para mantener el ejemplo simple, las variables lag y rolling se calculan
    desde el histórico disponible. En un sistema productivo, se haría predicción
    recursiva actualizando lags con las predicciones anteriores.
    """
    latest_date = df_original["usage_date"].max()
    min_date = df_original["usage_date"].min()
    services = sorted(df_original["service"].unique())

    future_rows = []

    for service in services:
        service_hist = (
            df_original[df_original["service"] == service]
            .sort_values("usage_date")
            .copy()
        )

        if len(service_hist) < 14:
            continue

        last_values = service_hist["net_cost"].tail(14).tolist()

        for step in range(1, days + 1):
            future_date = latest_date + pd.Timedelta(days=step)

            recent_7 = last_values[-7:]
            row = {
                "usage_date": future_date,
                "service": service,
                "day_of_week": future_date.dayofweek,
                "day_of_month": future_date.day,
                "month": future_date.month,
                "is_weekend": int(future_date.dayofweek >= 5),
                "trend_index": (future_date - min_date).days,
                "cost_lag_1": last_values[-1],
                "cost_lag_2": last_values[-2],
                "cost_lag_7": last_values[-7],
                "rolling_mean_7": float(np.mean(recent_7)),
                "rolling_std_7": float(np.std(recent_7)),
                "service_share": float(service_hist["service_share"].tail(7).mean()),
            }

            future_rows.append(row)

            # Aproximación conservadora para poder avanzar el horizonte:
            # mantenemos el último patrón hasta tener predicción.
            last_values.append(float(np.mean(recent_7)))

    return pd.DataFrame(future_rows)


def estimate_month_end(df_original: pd.DataFrame, future_predictions: pd.DataFrame) -> dict:
    today = df_original["usage_date"].max()
    month_start = today.replace(day=1)

    actual_month_cost = df_original[
        df_original["usage_date"] >= month_start
    ].drop_duplicates(
        subset=["usage_date", "service"]
    )["net_cost"].sum()

    days_in_month = (today + pd.offsets.MonthEnd(0)).day
    remaining_days = days_in_month - today.day

    predicted_7d = future_predictions["predicted_cost"].sum()

    if remaining_days <= 0:
        projected_remaining = 0.0
    elif remaining_days <= 7:
        projected_remaining = future_predictions[
            future_predictions["usage_date"] <= today + pd.Timedelta(days=remaining_days)
        ]["predicted_cost"].sum()
    else:
        avg_predicted_daily = (
            future_predictions.groupby("usage_date")["predicted_cost"].sum().mean()
        )
        projected_remaining = avg_predicted_daily * remaining_days

    projected_month_end = actual_month_cost + projected_remaining

    if projected_month_end >= MONTHLY_BUDGET * 1.10:
        risk = "ALTO"
    elif projected_month_end >= MONTHLY_BUDGET:
        risk = "MEDIO"
    else:
        risk = "BAJO"

    return {
        "actual_month_cost": actual_month_cost,
        "projected_remaining": projected_remaining,
        "projected_month_end": projected_month_end,
        "monthly_budget": MONTHLY_BUDGET,
        "risk": risk,
    }


def plot_predictions(df_original: pd.DataFrame, future_predictions: pd.DataFrame) -> None:
    daily_actual = (
        df_original.groupby("usage_date", as_index=False)["net_cost"]
        .sum()
        .rename(columns={"net_cost": "actual_cost"})
    )

    daily_pred = (
        future_predictions.groupby("usage_date", as_index=False)["predicted_cost"]
        .sum()
    )

    plt.figure(figsize=(11, 5))
    plt.plot(
        daily_actual["usage_date"],
        daily_actual["actual_cost"],
        label="Coste histórico",
    )
    plt.plot(
        daily_pred["usage_date"],
        daily_pred["predicted_cost"],
        marker="o",
        label="Predicción 7 días",
    )

    plt.title("Modelo predictivo de gasto cloud")
    plt.xlabel("Fecha")
    plt.ylabel("Coste neto diario")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/cost_prediction_7d.png", dpi=140)


def main() -> None:
    df_raw = load_cost_data()
    df_model = add_features(df_raw)

    model, features, test_result, mae, rmse = train_model(df_model)

    future_df = build_future_rows(df_raw, days=7)
    future_df["predicted_cost"] = model.predict(future_df[features])

    month_estimate = estimate_month_end(df_raw, future_df)

    test_result.to_csv("outputs/cost_model_test_result.csv", index=False)
    future_df.to_csv("outputs/cost_prediction_7d_by_service.csv", index=False)

    service_forecast = (
        future_df.groupby("service", as_index=False)["predicted_cost"]
        .sum()
        .sort_values("predicted_cost", ascending=False)
    )
    service_forecast.to_csv("outputs/cost_prediction_7d_by_service_summary.csv", index=False)

    plot_predictions(df_raw, future_df)

    print("\nEvaluación del modelo")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    print("\nPredicción de los próximos 7 días por servicio")
    print(service_forecast.to_string(index=False))

    print("\nEstimación de cierre mensual")
    print(f"Coste acumulado del mes       : {month_estimate['actual_month_cost']:.2f}")
    print(f"Coste pendiente estimado      : {month_estimate['projected_remaining']:.2f}")
    print(f"Cierre mensual estimado       : {month_estimate['projected_month_end']:.2f}")
    print(f"Presupuesto mensual           : {month_estimate['monthly_budget']:.2f}")
    print(f"Riesgo de sobrecoste          : {month_estimate['risk']}")

    print("\nFicheros generados:")
    print("- outputs/cost_model_test_result.csv")
    print("- outputs/cost_prediction_7d_by_service.csv")
    print("- outputs/cost_prediction_7d_by_service_summary.csv")
    print("- outputs/cost_prediction_7d.png")


if __name__ == "__main__":
    main()