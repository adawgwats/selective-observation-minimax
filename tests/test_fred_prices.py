from __future__ import annotations

from datetime import date

import pytest

from minimax_core import fred_prices as fp


def test_annual_means_drops_partial_years() -> None:
    rows = []
    for month in range(1, 13):
        rows.append((date(2020, month, 1), 100.0 + month))
    rows.append((date(2021, 1, 1), 120.0))
    rows.append((date(2021, 2, 1), 121.0))

    annual = fp._annual_means(rows)

    assert annual == [(2020, pytest.approx(106.5))]


def test_build_action_price_histories_scales_rebased_series(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch_crop_rebased_history(**_kwargs):
        return fp.FREDCropHistory(
            crop="corn",
            series_id="WPU012202",
            start_year=2000,
            end_year=2001,
            annual_observations=2,
            lookback_years_requested=2,
            lookback_years_used=2,
            annual_index_values=(120.0, 180.0),
            rebased_prices=(0.8, 1.2),
        )

    monkeypatch.setattr(fp, "fetch_crop_rebased_history", fake_fetch_crop_rebased_history)

    bundle = fp.build_action_price_histories_from_fred(
        action_keys=(("corn", "low"), ("corn", "medium"), ("unknown", "low")),
        base_price_by_action={
            ("corn", "low"): 5.0,
            ("corn", "medium"): 6.0,
            ("unknown", "low"): 3.0,
        },
        lookback_years=2,
        end_year=2001,
        cache_dir="data/fred_cache_test",
    )

    assert bundle.price_history_by_action[("corn", "low")] == pytest.approx([4.0, 6.0])
    assert bundle.price_history_by_action[("corn", "medium")] == pytest.approx([4.8, 7.2])
    assert bundle.price_history_by_action[("unknown", "low")] == pytest.approx([3.0, 3.0])
