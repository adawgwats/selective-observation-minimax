from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


PRICE_DYNAMICS_MODELS = ("ema", "autoregressive", "statsmodels_arima")


@dataclass(frozen=True)
class PriceDynamicsConfig:
    model: str = "ema"
    spot_weight: float = 0.65
    ema_alpha: float = 0.35

    def __post_init__(self) -> None:
        if self.model not in PRICE_DYNAMICS_MODELS:
            raise ValueError(f"model must be one of {PRICE_DYNAMICS_MODELS}.")
        if not 0.0 <= self.spot_weight <= 1.0:
            raise ValueError("spot_weight must be in [0, 1].")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")


def estimate_decision_price(
    *,
    history: Sequence[float],
    spot_price: float,
    config: PriceDynamicsConfig,
) -> float:
    clean_history = [max(float(price), 0.01) for price in history if math.isfinite(float(price))]
    clean_spot = max(float(spot_price), 0.01)
    if not clean_history:
        return clean_spot
    history_forecast = _forecast_from_history(clean_history, config=config)
    blended = config.spot_weight * clean_spot + (1.0 - config.spot_weight) * history_forecast
    return max(float(blended), 0.01)


def _forecast_from_history(history: Sequence[float], *, config: PriceDynamicsConfig) -> float:
    if config.model == "ema":
        return _ema_forecast(history, alpha=config.ema_alpha)
    if config.model == "autoregressive":
        return _ar1_forecast(history)
    statsmodels_forecast = _statsmodels_arima_forecast(history)
    if statsmodels_forecast is not None:
        return statsmodels_forecast
    return _ar1_forecast(history)


def _ema_forecast(history: Sequence[float], *, alpha: float) -> float:
    level = float(history[0])
    for price in history[1:]:
        level = alpha * float(price) + (1.0 - alpha) * level
    return max(level, 0.01)


def _ar1_forecast(history: Sequence[float]) -> float:
    if len(history) < 2:
        return max(float(history[-1]), 0.01)
    x = [float(value) for value in history[:-1]]
    y = [float(value) for value in history[1:]]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    covariance = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y))
    variance = sum((x_value - x_mean) ** 2 for x_value in x)
    slope = covariance / variance if variance > 1.0e-12 else 0.0
    intercept = y_mean - slope * x_mean
    forecast = intercept + slope * float(history[-1])
    return max(forecast, 0.01)


def _statsmodels_arima_forecast(history: Sequence[float]) -> float | None:
    if len(history) < 4:
        return None
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        model = ARIMA(list(history), order=(1, 1, 0), trend="n")
        fitted = model.fit()
        forecast = float(fitted.forecast(steps=1)[0])
    except Exception:
        return None
    if not math.isfinite(forecast):
        return None
    return max(forecast, 0.01)
