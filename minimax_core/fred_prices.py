from __future__ import annotations

import csv
import io
import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Sequence


FRED_SERIES_BY_CROP: dict[str, str] = {
    "corn": "WPU012202",
    "soy": "WPU01830131",
    "wheat": "WPU0121",
    "rice": "WPU0123",
    "peanut": "WPU01830111",
    "sunflower": "WPU01830161",
}


@dataclass(frozen=True)
class FREDCropHistory:
    crop: str
    series_id: str
    start_year: int
    end_year: int
    annual_observations: int
    lookback_years_requested: int
    lookback_years_used: int
    annual_index_values: tuple[float, ...]
    rebased_prices: tuple[float, ...]


@dataclass(frozen=True)
class FREDActionHistoryBundle:
    price_history_by_action: dict[tuple[str, str], list[float]]
    crop_histories: dict[str, FREDCropHistory]


def build_action_price_histories_from_fred(
    *,
    action_keys: Sequence[tuple[str, str]],
    base_price_by_action: dict[tuple[str, str], float],
    lookback_years: int = 100,
    end_year: int | None = None,
    cache_dir: str = "data/fred_cache",
) -> FREDActionHistoryBundle:
    unique_crops = sorted({str(crop) for crop, _ in action_keys})
    crop_histories: dict[str, FREDCropHistory] = {}
    for crop in unique_crops:
        series_id = FRED_SERIES_BY_CROP.get(crop)
        if series_id is None:
            continue
        history = fetch_crop_rebased_history(
            crop=crop,
            series_id=series_id,
            lookback_years=lookback_years,
            end_year=end_year,
            cache_dir=cache_dir,
        )
        if history is not None:
            crop_histories[crop] = history

    price_history_by_action: dict[tuple[str, str], list[float]] = {}
    for action_key in action_keys:
        crop = str(action_key[0])
        base_price = max(float(base_price_by_action.get(action_key, 1.0)), 0.01)
        crop_history = crop_histories.get(crop)
        if crop_history is None or not crop_history.annual_index_values:
            price_history_by_action[action_key] = [base_price for _ in range(max(lookback_years, 1))]
            continue
        # Convert rebased-to-1 index series into action-specific price levels.
        rebased_prices = [base_price * value for value in crop_history.rebased_prices]
        price_history_by_action[action_key] = rebased_prices

    return FREDActionHistoryBundle(
        price_history_by_action=price_history_by_action,
        crop_histories=crop_histories,
    )


def fetch_crop_rebased_history(
    *,
    crop: str,
    series_id: str,
    lookback_years: int,
    end_year: int | None,
    cache_dir: str,
) -> FREDCropHistory | None:
    if lookback_years <= 0:
        return None
    series_rows = _fetch_fred_series_rows(series_id, cache_dir=cache_dir)
    annual = _annual_means(series_rows)
    if not annual:
        return None

    effective_end_year = _effective_end_year(end_year)
    annual = [(year, value) for year, value in annual if year <= effective_end_year]
    if not annual:
        return None

    annual = annual[-lookback_years:]
    values = [value for _, value in annual]
    if not values:
        return None

    anchor_window = values[-min(10, len(values)) :]
    anchor = max(mean(anchor_window), 1.0e-6)
    rebased = [max(value / anchor, 0.01) for value in values]
    return FREDCropHistory(
        crop=crop,
        series_id=series_id,
        start_year=int(annual[0][0]),
        end_year=int(annual[-1][0]),
        annual_observations=len(annual),
        lookback_years_requested=int(lookback_years),
        lookback_years_used=len(annual),
        annual_index_values=tuple(float(value) for value in values),
        rebased_prices=tuple(float(value) for value in rebased),
    )


def _effective_end_year(explicit_end_year: int | None) -> int:
    if explicit_end_year is not None:
        return int(explicit_end_year)
    return int(date.today().year - 1)


def _annual_means(rows: Sequence[tuple[date, float]]) -> list[tuple[int, float]]:
    by_year: dict[int, list[float]] = {}
    for observed_date, value in rows:
        by_year.setdefault(int(observed_date.year), []).append(float(value))

    annual: list[tuple[int, float]] = []
    for year in sorted(by_year):
        year_values = by_year[year]
        if len(year_values) < 12:
            # Keep only complete years, avoiding partial-year lookahead.
            continue
        annual.append((year, sum(year_values) / len(year_values)))
    return annual


def _fetch_fred_series_rows(series_id: str, *, cache_dir: str) -> list[tuple[date, float]]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    data_path = cache_path / f"{series_id}.csv"
    meta_path = cache_path / f"{series_id}.meta.json"
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", errors="replace")
    data_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "series_id": series_id,
                "url": url,
                "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[tuple[date, float]] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        raw_date = (row.get("observation_date") or "").strip()
        raw_value = (row.get(series_id) or "").strip()
        if not raw_date or not raw_value or raw_value == ".":
            continue
        try:
            observed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            value = float(raw_value)
        except ValueError:
            continue
        rows.append((observed_date, value))
    return rows
