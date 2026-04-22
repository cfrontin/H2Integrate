from types import SimpleNamespace

from attrs import evolve
import numpy as np
import pandas as pd
import pyomo.environ as pyomo
import pytest

from h2integrate.control.control_strategies.storage.plm_optimized_storage_controller import (
    PLMOptimizedControllerConfig,
    PLMOptimizedStorageController,
)
from h2integrate.storage.storage_baseclass import StoragePerformanceBase


def _make_controller():
    return object.__new__(PLMOptimizedStorageController)


def _make_controller_with_config(config, n_timesteps=24):
    """Build a controller with pre-computed masks, bypassing OpenMDAO setup."""
    controller = _make_controller()
    controller.config = config
    controller.updated_initial_soc = config.init_soc_fraction
    controller.time_index = pd.date_range("2024-01-01", periods=n_timesteps, freq="h")
    controller.in_peak_window = controller._compute_peak_window_mask()
    controller.month_ids = controller._compute_month_ids()
    return controller


@pytest.fixture
def base_config():
    n = 24
    return PLMOptimizedControllerConfig(
        max_capacity=10.0,
        max_soc_fraction=1.0,
        min_soc_fraction=0.0,
        init_soc_fraction=1.0,
        n_control_window=n,
        commodity="electricity",
        commodity_rate_units="kW",
        tech_name="battery",
        system_commodity_interface_limit=100.0,
        max_charge_rate=1.0,
        supervisory_signal=list(range(n)),
        peak_window={"start": "08:00:00", "end": "18:00:00"},
        performance_incentive=10.0,
        n_max_events=24,
        signal_threshold_percentile=0.0,
    )


@pytest.mark.unit
def test_parse_peak_window():
    controller = _make_controller()
    controller.config = SimpleNamespace(
        peak_window={"start": "08:00:00", "end": "18:40:20"}
    )
    start, end = controller._parse_peak_window()
    assert start.hour == 8
    assert start.minute == 0
    assert start.second == 0
    assert end.hour == 18
    assert end.minute == 40
    assert end.second == 20


@pytest.mark.unit
def test_parse_peak_window_invalid_format():
    controller = _make_controller()
    controller.config = SimpleNamespace(peak_window={"start": "08", "end": "18:40:20"})
    with pytest.raises(
        ValueError, match="peak_window start value must be a string in HH:MM:SS format"
    ):
        controller._parse_peak_window()


@pytest.mark.unit
def test_parse_peak_window_int_raises():
    controller = _make_controller()
    controller.config = SimpleNamespace(peak_window={"start": 8, "end": 9})
    with pytest.raises(ValueError):
        controller._parse_peak_window()


@pytest.mark.unit
def test_parse_peak_window_missing_key_raises():
    controller = _make_controller()
    controller.config = SimpleNamespace(peak_window={"start": "08:00:00"})
    with pytest.raises(
        ValueError, match="peak_window must contain 'start' and 'end' keys"
    ):
        controller._parse_peak_window()


@pytest.mark.unit
def test_compute_peak_window_mask():
    controller = _make_controller()
    controller.config = SimpleNamespace(
        peak_window={"start": "00:00:00", "end": "02:00:00"}
    )
    controller.time_index = pd.date_range("2024-01-01", periods=24, freq="h")
    mask = controller._compute_peak_window_mask()
    expected = np.array([i <= 2 for i in range(24)])
    assert isinstance(mask, np.ndarray)
    assert np.array_equal(mask, expected)


@pytest.mark.unit
def test_compute_month_ids():
    # Jan 2024: 744h, Feb 2024 (leap year): 696h, Mar 2024: 744h
    controller = _make_controller()
    controller.time_index = pd.date_range(
        "2024-01-01", periods=744 + 696 + 744, freq="h"
    )
    month_ids = controller._compute_month_ids()
    expected = np.array([1] * 744 + [2] * 696 + [3] * 744)
    assert np.array_equal(month_ids, expected)


@pytest.mark.unit
def test_compute_eligible_mask_zero_percentile_all_eligible():
    """All timesteps are eligible when signal_threshold_percentile=0."""
    controller = _make_controller()
    controller.config = SimpleNamespace(signal_threshold_percentile=0.0)
    signal = np.array([1.0, 5.0, 3.0, 2.0, 8.0])
    mask = controller._compute_eligible_mask(signal)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == len(signal)
    assert mask.all()


@pytest.mark.unit
def test_compute_eligible_mask_50th_percentile():
    """Only values at or above the 50th percentile are eligible."""
    controller = _make_controller()
    controller.config = SimpleNamespace(signal_threshold_percentile=50.0)
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = controller._compute_eligible_mask(signal)
    expected = signal >= np.percentile(signal, 50.0)
    assert np.array_equal(mask, expected)


@pytest.mark.unit
def test_compute_eligible_mask_100th_percentile_only_max_eligible():
    """Only the maximum value(s) are eligible at percentile=100."""
    controller = _make_controller()
    controller.config = SimpleNamespace(signal_threshold_percentile=100.0)
    signal = np.array([1.0, 2.0, 10.0, 3.0, 10.0])
    mask = controller._compute_eligible_mask(signal)
    expected = np.array([False, False, True, False, True])
    assert np.array_equal(mask, expected)


@pytest.mark.unit
def test_compute_eligible_mask_uniform_signal_all_eligible():
    """All timesteps are eligible when the signal is uniform, regardless of percentile."""
    controller = _make_controller()
    controller.config = SimpleNamespace(signal_threshold_percentile=75.0)
    signal = np.full(10, 5.0)
    mask = controller._compute_eligible_mask(signal)
    assert mask.all()


@pytest.mark.regression
def test_optimizer_dispatch_only_in_peak_window(base_config):
    """Solver must never set 1 outside the peak window."""
    #
    controller = _make_controller_with_config(base_config)
    model = controller._build_dr_model(
        window_start=0,
        window_len=24,
        init_soc=base_config.init_soc_fraction,
        remaining_budget={1: base_config.n_max_events},
    )

    PLMOptimizedStorageController.glpk_solve_call(model)

    peak_start, peak_end = controller._parse_peak_window()
    print("Peak window:", peak_start, "-", peak_end)
    for t in range(24):
        hour = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=t)
        in_window = peak_start <= hour.time() <= peak_end
        if not in_window:
            assert pyomo.value(model.discharge[t]) < 1e-3


@pytest.mark.regression
def test_optimizer_dispatch_only_on_eligible_timesteps(base_config):
    """Solver must never set 1 on ineligible timesteps."""
    controller = _make_controller_with_config(base_config)
    model = controller._build_dr_model(
        window_start=0,
        window_len=24,
        init_soc=base_config.init_soc_fraction,
        remaining_budget={1: base_config.n_max_events},
    )

    PLMOptimizedStorageController.glpk_solve_call(model)

    signal = np.array(controller.config.supervisory_signal)
    eligible_mask = controller._compute_eligible_mask(signal)
    for t in range(24):
        if not eligible_mask[t]:
            assert pyomo.value(model.discharge[t]) < 1e-3


@pytest.mark.regression
def test_optimizer_dispatch_respects_event_budget(base_config):
    """Solver must never set more than n_max_events timesteps to 1."""
    controller = _make_controller_with_config(base_config)
    model = controller._build_dr_model(
        window_start=0,
        window_len=24,
        init_soc=base_config.init_soc_fraction,
        remaining_budget={1: base_config.n_max_events},
    )

    PLMOptimizedStorageController.glpk_solve_call(model)

    total_events = sum(pyomo.value(model.discharge[t]) for t in range(24))
    assert total_events <= base_config.n_max_events + 1e-3


@pytest.mark.regression
def test_optimizer_dispatch_respects_soc_constraints(base_config):
    """Solver must never violate SOC constraints."""
    controller = _make_controller_with_config(base_config)
    model = controller._build_dr_model(
        window_start=0,
        window_len=24,
        init_soc=base_config.init_soc_fraction,
        remaining_budget={2: base_config.n_max_events},
    )

    PLMOptimizedStorageController.glpk_solve_call(model)

    soc = base_config.init_soc_fraction * base_config.max_capacity
    for t in range(24):
        charge = pyomo.value(model.charge[t])
        discharge = pyomo.value(model.discharge[t])
        soc += charge - discharge
        assert soc >= base_config.min_soc_fraction * base_config.max_capacity
        assert soc <= base_config.max_soc_fraction * base_config.max_capacity


@pytest.mark.regression
def test_optimizer_dispatch_respects_charge_discharge_exclusivity(base_config):
    """Solver must never set charge and discharge to 1 in the same timestep."""
    controller = _make_controller_with_config(base_config)
    model = controller._build_dr_model(
        window_start=0,
        window_len=24,
        init_soc=base_config.init_soc_fraction,
        remaining_budget={2: base_config.n_max_events},
    )

    PLMOptimizedStorageController.glpk_solve_call(model)

    for t in range(24):
        charge = pyomo.value(model.charge[t])
        discharge = pyomo.value(model.discharge[t])
        assert not (charge > 0.5 and discharge > 0.5)
