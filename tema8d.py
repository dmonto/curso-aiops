from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_PATH = Path("time_series_latency_model.joblib")
PREDICTIONS_PATH = Path("time_series_latency_predictions.csv")


def build_time_series() -> pd.DataFrame:
    """
    Crea una serie temporal sintética de latencia p95.
    En un caso real, estos datos vendrían de Cloud Monitoring,
    Cloud Logging exportado a BigQuery o una tabla agregada por ventanas.
    """

    timestamps = pd.date_range(
        start="2026-05-23 08:00:00",
        periods=180,
        freq="5min",
    )

    rows = []

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        minute = ts.minute

        # Patrón base
        base_latency = 180

        # Más tráfico en horario laboral
        business_hours_effect = 80 if 9 <= hour <= 18 else 20

        # Pico suave a media mañana
        morning_peak = 120 if 10 <= hour <= 11 else 0

        # Pico por despliegue simulado
        deployment_effect = 0
        if pd.Timestamp("2026-05-23 14:00:00") <= ts <= pd.Timestamp("2026-05-23 15:00:00"):
            deployment_effect = 180 + (i % 6) * 20

        # Tendencia creciente al final del día
        late_day_trend = max(0, (hour - 16) * 35)

        # Ruido determinista sencillo para que el ejemplo sea reproducible
        noise = ((i * 17) % 45) - 20

        latency = (
            base_latency
            + business_hours_effect
            + morning_peak
            + deployment_effect
            + late_day_trend
            + noise
        )

        request_count = (
            800
            + business_hours_effect * 18
            + morning_peak * 10
            + late_day_trend * 12
            + ((i * 23) % 300)
        )

        error_rate = max(
            0.1,
            (latency - 250) / 300 + ((i * 7) % 20) / 100,
        )

        cpu_percent = min(
            98,
            35 + request_count / 120 + deployment_effect / 18 + ((i * 5) % 12),
        )

        rows.append(
            {
                "timestamp": ts,
                "service": "checkout-api",
                "latency_p95_ms": round(latency, 2),
                "request_count": round(request_count, 2),
                "error_rate": round(error_rate, 3),
                "cpu_percent": round(cpu_percent, 2),
                "is_after_deployment": 1 if deployment_effect > 0 else 0,
            }
        )

    return pd.DataFrame(rows)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_business_hours"] = result["hour"].between(9, 18).astype(int)

    return result


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values("timestamp").copy()

    # Lags de la métrica objetivo
    result["latency_lag_1"] = result["latency_p95_ms"].shift(1)
    result["latency_lag_2"] = result["latency_p95_ms"].shift(2)
    result["latency_lag_3"] = result["latency_p95_ms"].shift(3)
    result["latency_lag_6"] = result["latency_p95_ms"].shift(6)

    # Rolling features sobre latencia
    result["latency_roll_mean_3"] = result["latency_p95_ms"].rolling(window=3).mean()
    result["latency_roll_mean_6"] = result["latency_p95_ms"].rolling(window=6).mean()
    result["latency_roll_max_6"] = result["latency_p95_ms"].rolling(window=6).max()

    # Rolling features sobre errores y CPU
    result["error_rate_roll_mean_3"] = result["error_rate"].rolling(window=3).mean()
    result["cpu_roll_mean_3"] = result["cpu_percent"].rolling(window=3).mean()

    # Target: latencia de la siguiente ventana
    result["target_latency_next_window"] = result["latency_p95_ms"].shift(-1)

    # Eliminamos filas incompletas por lags y target futuro
    result = result.dropna().reset_index(drop=True)

    return result


def train_test_split_by_time(df: pd.DataFrame, train_ratio: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def get_feature_columns() -> list[str]:
    return [
        "request_count",
        "error_rate",
        "cpu_percent",
        "is_after_deployment",
        "hour",
        "day_of_week",
        "is_business_hours",
        "latency_lag_1",
        "latency_lag_2",
        "latency_lag_3",
        "latency_lag_6",
        "latency_roll_mean_3",
        "latency_roll_mean_6",
        "latency_roll_max_6",
        "error_rate_roll_mean_3",
        "cpu_roll_mean_3",
    ]


def train_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    feature_columns = get_feature_columns()

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )

    model.fit(
        train_df[feature_columns],
        train_df["target_latency_next_window"],
    )

    return model


def evaluate_model(model: RandomForestRegressor, test_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = get_feature_columns()

    predictions = model.predict(test_df[feature_columns])

    mae = mean_absolute_error(test_df["target_latency_next_window"], predictions)
    rmse = math.sqrt(mean_squared_error(test_df["target_latency_next_window"], predictions))
    r2 = r2_score(test_df["target_latency_next_window"], predictions)

    print("\n=== Evaluación temporal ===")
    print(f"MAE  : {mae:.2f} ms")
    print(f"RMSE : {rmse:.2f} ms")
    print(f"R2   : {r2:.3f}")

    result = test_df[
        [
            "timestamp",
            "service",
            "latency_p95_ms",
            "target_latency_next_window",
            "request_count",
            "error_rate",
            "cpu_percent",
            "is_after_deployment",
        ]
    ].copy()

    result["predicted_latency_next_window"] = predictions.round(2)
    result["absolute_error_ms"] = (
        result["target_latency_next_window"] - result["predicted_latency_next_window"]
    ).abs().round(2)

    result["risk_level"] = pd.cut(
        result["predicted_latency_next_window"],
        bins=[0, 300, 700, float("inf")],
        labels=["low", "medium", "high"],
        right=False,
    )

    result["recommended_action"] = result["risk_level"].map(
        {
            "low": "Sin acción inmediata",
            "medium": "Revisar tendencia y SLO",
            "high": "Generar alerta predictiva y revisar escalado",
        }
    )

    return result


def forecast_next_window(model: RandomForestRegressor, prepared_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simula una predicción online usando la última fila disponible.
    En producción, esta fila vendría de la última ventana cerrada.
    """

    feature_columns = get_feature_columns()

    last_row = prepared_df.tail(1).copy()
    prediction = model.predict(last_row[feature_columns])[0]

    forecast = last_row[
        [
            "timestamp",
            "service",
            "latency_p95_ms",
            "request_count",
            "error_rate",
            "cpu_percent",
        ]
    ].copy()

    forecast["forecast_for_next_window"] = last_row["timestamp"] + pd.Timedelta(minutes=5)
    forecast["predicted_latency_p95_ms"] = round(prediction, 2)

    if prediction >= 700:
        forecast["risk_level"] = "high"
        forecast["recommended_action"] = "Generar alerta predictiva"
    elif prediction >= 300:
        forecast["risk_level"] = "medium"
        forecast["recommended_action"] = "Revisar tendencia"
    else:
        forecast["risk_level"] = "low"
        forecast["recommended_action"] = "Sin acción inmediata"

    return forecast


def main() -> None:
    raw_df = build_time_series()
    df = add_time_features(raw_df)
    prepared_df = add_lag_and_rolling_features(df)

    print("=== Muestra de la serie preparada ===")
    print(prepared_df.head(10))

    train_df, test_df = train_test_split_by_time(prepared_df, train_ratio=0.75)

    print("\nFilas train:", len(train_df))
    print("Filas test :", len(test_df))

    model = train_model(train_df)

    evaluation_df = evaluate_model(model, test_df)

    print("\n=== Predicciones sobre tramo de test ===")
    print(evaluation_df.tail(20))

    evaluation_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    joblib.dump(model, MODEL_PATH)

    print(f"\nPredicciones exportadas a: {PREDICTIONS_PATH}")
    print(f"Modelo guardado en: {MODEL_PATH}")

    next_forecast = forecast_next_window(model, prepared_df)

    print("\n=== Predicción de la próxima ventana ===")
    print(next_forecast)


if __name__ == "__main__":
    main()