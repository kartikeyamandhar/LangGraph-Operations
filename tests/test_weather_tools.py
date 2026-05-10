"""
Weather risk tests — boundary conditions matter because the integer score
drives the buffer policy and the escalation flag downstream.
"""
from __future__ import annotations

from tools.weather_tools import derive_dispatch_weather_risk


def _forecast(precip: list, gusts: list, tmin: list) -> dict:
    return {"daily": {
        "precipitation_sum": precip,
        "wind_gusts_10m_max": gusts,
        "temperature_2m_min": tmin,
    }}


# ---------------------------------------------------------------------------
# Score 0: nothing triggered
# ---------------------------------------------------------------------------
def test_score_zero_clear_weather():
    risk = derive_dispatch_weather_risk(_forecast([0.0, 1.0], [10.0, 12.0], [12.0, 14.0]))
    assert risk["risk_score_0_3"] == 0
    assert risk["risk_flags"] == {
        "heavy_rain_risk": False,
        "high_wind_risk": False,
        "freezing_risk": False,
    }


# ---------------------------------------------------------------------------
# Each individual flag at the threshold
# ---------------------------------------------------------------------------
def test_rain_threshold_inclusive_at_15mm():
    """Playbook says >= 15 mm/day triggers — exactly 15.0 must trigger."""
    risk = derive_dispatch_weather_risk(_forecast([15.0, 0.0], [10.0], [10.0]))
    assert risk["risk_flags"]["heavy_rain_risk"] is True
    assert risk["risk_score_0_3"] == 1


def test_rain_just_below_threshold_does_not_trigger():
    risk = derive_dispatch_weather_risk(_forecast([14.9, 0.0], [10.0], [10.0]))
    assert risk["risk_flags"]["heavy_rain_risk"] is False
    assert risk["risk_score_0_3"] == 0


def test_wind_threshold_inclusive_at_45kmh():
    risk = derive_dispatch_weather_risk(_forecast([0.0], [45.0, 30.0], [10.0]))
    assert risk["risk_flags"]["high_wind_risk"] is True
    assert risk["risk_score_0_3"] == 1


def test_wind_just_below_threshold_does_not_trigger():
    risk = derive_dispatch_weather_risk(_forecast([0.0], [44.9], [10.0]))
    assert risk["risk_flags"]["high_wind_risk"] is False


def test_freezing_threshold_inclusive_at_zero_celsius():
    risk = derive_dispatch_weather_risk(_forecast([0.0], [10.0], [0.0, 5.0]))
    assert risk["risk_flags"]["freezing_risk"] is True
    assert risk["risk_score_0_3"] == 1


def test_freezing_just_above_threshold_does_not_trigger():
    risk = derive_dispatch_weather_risk(_forecast([0.0], [10.0], [0.1]))
    assert risk["risk_flags"]["freezing_risk"] is False


# ---------------------------------------------------------------------------
# Score accumulation
# ---------------------------------------------------------------------------
def test_two_flags_score_two():
    risk = derive_dispatch_weather_risk(_forecast([20.0], [60.0], [5.0]))
    assert risk["risk_score_0_3"] == 2


def test_all_flags_score_three():
    risk = derive_dispatch_weather_risk(_forecast([25.0], [50.0], [-2.0]))
    assert risk["risk_score_0_3"] == 3
    assert all(risk["risk_flags"].values())


# ---------------------------------------------------------------------------
# Empty / degenerate inputs must not crash
# ---------------------------------------------------------------------------
def test_empty_forecast_returns_score_zero():
    risk = derive_dispatch_weather_risk({"daily": {}})
    assert risk["risk_score_0_3"] == 0
    assert risk["min_temp_c"] is None


def test_missing_daily_block():
    risk = derive_dispatch_weather_risk({})
    assert risk["risk_score_0_3"] == 0


# ---------------------------------------------------------------------------
# Output schema invariants — used by audit + report nodes
# ---------------------------------------------------------------------------
def test_output_has_all_required_fields():
    risk = derive_dispatch_weather_risk(_forecast([5.0], [20.0], [10.0]))
    for key in ("max_precip_mm_day", "max_wind_gust_kmh", "min_temp_c",
                "risk_flags", "risk_score_0_3"):
        assert key in risk
    assert isinstance(risk["risk_score_0_3"], int)
    assert 0 <= risk["risk_score_0_3"] <= 3
