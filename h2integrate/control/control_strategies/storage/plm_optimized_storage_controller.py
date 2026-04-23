from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyomo.environ as pyomo
from attrs import field, define

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.validators import range_val
from h2integrate.control.control_strategies.pyomo_storage_controller_baseclass import (
    SolverOptions,
    PyomoStorageControllerBaseClass,
    PyomoStorageControllerBaseConfig,
)
from h2integrate.control.control_strategies.controller_opt_problem_state import (
    DispatchProblemState,
)
from matplotlib import pyplot as plt


@define
class PLMOptimizedControllerConfig(PyomoStorageControllerBaseConfig):
    """Configuration for the PLM DR optimized storage controller.

    Inherits base fields from ``PyomoControllerBaseConfig``:
    ``max_capacity``, ``max_soc_fraction``, ``min_soc_fraction``,
    ``init_soc_fraction``, ``n_control_window``, ``commodity``,
    ``commodity_rate_units``, ``tech_name``,
    ``system_commodity_interface_limit``, ``round_digits``.

    Attributes:
        max_charge_rate (float): Maximum charge and discharge rate (kW).
        supervisory_signal (list[float]): Price, demand, or price*demand
            forecast time series. The rolling solver uses one window of
            length ``n_control_window`` per solve.
        peak_window (dict): Hours eligible for dispatch. Keys ``'start'``
            and ``'end'`` must be strings in ``HH:MM:SS`` format.
        performance_incentive (float): Incentive revenue ($/kW per
            dispatch hour).
        charge_efficiency (float): Charge efficiency in [0, 1].
            Defaults to 1.0.
        discharge_efficiency (float): Discharge efficiency in [0, 1].
            Defaults to 1.0.
        n_max_events (int): Maximum discharge events per calendar month.
            Defaults to 10.
        n_control_window (int): Number of timesteps per rolling solve
            window. Defaults to ``24 * 30`` (one month of hourly data).
        signal_threshold_percentile (float): Percentile (0-100) used to
            compute the signal threshold for each rolling window. Only
            hours at or above this percentile of the window signal are
            eligible for dispatch. Defaults to 0.0 (all hours eligible).
    """

    max_charge_rate: float = field()
    supervisory_signal: list = field()
    peak_window: dict = field()
    performance_incentive: float = field()
    charge_efficiency: float = field(validator=range_val(0, 1), default=1.0)
    discharge_efficiency: float = field(validator=range_val(0, 1), default=1.0)
    n_max_events: int = field(default=10)
    n_control_window: int = field(default=24 * 30)  # one month of hourly data
    signal_threshold_percentile: float = field(default=0.0, validator=range_val(0,100)) # make sure this is valid


class PLMOptimizedStorageController(PyomoStorageControllerBaseClass):
    """Demand-response storage controller using a rolling-horizon MILP.

    Each call to the dispatch solver iterates over the full simulation in
    windows of length ``n_control_window``. For each window it receives
    the monthly LMP forecast, solves the MILP to maximize incentive
    revenue, then passes the resulting dispatch commands to the
    performance model. The terminal SOC of each window is carried forward
    as the initial SOC of the next window.
    """

    def setup(self):
        """Initialize config, register OpenMDAO inputs, and pre-compute static masks.

        Raises:
            ValueError: If the length of the time series built from
                ``plant_config`` does not match ``n_timesteps``.
        """
        self.config = PLMOptimizedControllerConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "control")
        )

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=self.config.commodity_rate_units,
            desc="Maximum charge/discharge rate P_max",
        )
        self.add_input(
            "storage_capacity",
            val=self.config.max_capacity,
            units=f"{self.config.commodity_rate_units}*h",
            desc="Total storage capacity",
        )

        self.n_timesteps = self.options["plant_config"]["plant"]["simulation"][
            "n_timesteps"
        ]
        super().setup()

        self.updated_initial_soc = self.config.init_soc_fraction

        self.commodity_info = {
            "commodity_name": self.config.commodity,
            "commodity_storage_units": self.config.commodity_rate_units,
        }

        self.time_index = self._build_time_index(self.options["plant_config"])

        if len(self.time_index) != self.n_timesteps:
            raise ValueError(
                f"Time series length {len(self.time_index)} != n_timesteps {self.n_timesteps}"
            )

        self.in_peak_window = self._compute_peak_window_mask()  # bool array, shape (T,)
        self.month_ids = self._compute_month_ids()              # int array,  shape (T,)

    @staticmethod
    def _build_time_index(plant_config: dict) -> pd.DatetimeIndex:
        """Build a timezone-aware DatetimeIndex from simulation settings in plant_config.

        Args:
            plant_config (dict): Plant configuration dict. Must contain
                ``plant.simulation`` with keys ``n_timesteps`` (int),
                ``dt`` (int, seconds), ``timezone`` (int, UTC offset),
                and ``start_time`` (str).

        Returns:
            pd.DatetimeIndex: DatetimeIndex of length ``n_timesteps`` spaced
            ``dt`` seconds apart, starting at ``start_time`` in the given timezone.
        """
        sim = plant_config["plant"]["simulation"]
        n_timesteps = int(sim["n_timesteps"])
        dt_seconds = int(sim["dt"])
        tz = int(sim["timezone"])
        start = pd.Timestamp(sim["start_time"], tz=tz)
        freq = pd.to_timedelta(dt_seconds, unit="s")
        return pd.date_range(start=start, periods=n_timesteps, freq=freq)

    def _parse_peak_window(self) -> tuple:
        """Parse the ``peak_window`` config entry into ``datetime.time`` objects.

        Returns:
            tuple[datetime.time, datetime.time]: ``(start, end)`` times.

        Raises:
            ValueError: If ``'start'`` or ``'end'`` keys are missing, or
                if either value is not a string in ``HH:MM:SS`` format.
        """
        pw = dict(self.config.peak_window)
        if "start" not in pw or "end" not in pw:
            raise ValueError("peak_window must contain 'start' and 'end' keys")
        for key in ("start", "end"):
            val = pw[key]
            if not isinstance(val, str) or len(val.split(":")) != 3:
                raise ValueError(
                    f"peak_window {key} value must be a string in HH:MM:SS format, got {val}."
                )
            pw[key] = datetime.strptime(val, "%H:%M:%S").time()
        return pw["start"], pw["end"]

    def _compute_peak_window_mask(self) -> np.ndarray:
        """Build a boolean mask that is ``True`` for timesteps inside the peak window.

        Returns:
            np.ndarray: Boolean array of shape ``(n_timesteps,)``.

        Raises:
            ValueError: If ``peak_window`` end time is before start time.
        """
        start, end = self._parse_peak_window()
        times = pd.DatetimeIndex(self.time_index).time
        if end < start:
            raise ValueError("peak_window end time must be after start time.")
        return np.array([start <= t <= end for t in times])

    def _compute_month_ids(self) -> np.ndarray:
        """Return the calendar month index (1-12) for each timestep.

        Returns:
            np.ndarray: Integer array of shape ``(n_timesteps,)``.
        """
        return pd.DatetimeIndex(self.time_index).month.to_numpy()

    def _compute_eligible_mask(self, signal_window: np.ndarray) -> np.ndarray:
        """Build a boolean mask for timesteps whose signal meets the dispatch threshold.

        The threshold is derived only from ``signal_window`` — it does not
        assume the full simulation signal is known in advance. When
        ``signal_threshold_percentile`` is 0.0 all hours are eligible.

        Args:
            signal_window (np.ndarray): Signal values for the current
                rolling window.

        Returns:
            np.ndarray: Boolean array of shape ``(len(signal_window),)``.
                ``True`` where ``signal_t >= threshold``.
        """
        eligible = np.ones(len(signal_window), dtype=bool)

        if self.config.signal_threshold_percentile > 0.0:
            threshold = np.percentile(
                signal_window, self.config.signal_threshold_percentile
            )
            eligible = signal_window >= threshold

        return eligible

    def pyomo_setup(self, discrete_inputs):
        """Return the rolling-horizon dispatch solver callable.

        Args:
            discrete_inputs (dict): OpenMDAO discrete inputs. 

        Returns:
            callable: ``pyomo_dispatch_solver(performance_model,
            performance_model_kwargs, inputs)`` that iterates over the
            simulation in windows of ``n_control_window`` timesteps.
            For each window it:

            1. Builds a fresh MILP from the window's signal slice.
            2. Solves the MILP with GLPK.
            3. Calls ``performance_model`` with the resulting dispatch
               commands.
            4. Carries the terminal SOC into the next window.

            Returns ``(storage_out, soc_out)`` - two ``np.ndarray`` of
            length ``n_timesteps``.
        """

        def pyomo_dispatch_solver(
            performance_model,
            performance_model_kwargs,
            inputs,
            commodity_name=self.config.commodity,
        ):
            storage_out = np.zeros(self.n_timesteps)
            soc_out = np.zeros(self.n_timesteps)

            # Track events used per calendar month so the monthly cap is
            # respected across window boundaries.
            events_used_per_month = {}

            n_w = self.config.n_control_window
            window_start_indices = list(range(0, self.n_timesteps, n_w))

            for window_start in window_start_indices:
                window_len = min(n_w, self.n_timesteps - window_start)

                n_windows = len(window_start_indices)
                report_every = max(1, n_windows // 4)
                window_idx = window_start // n_w
                if window_idx % report_every == 0:
                    pct = round(window_start / self.n_timesteps * 100)
                    print(f"{pct}% done with PLM rolling dispatch")

                month_ids_w = self.month_ids[window_start : window_start + window_len]
                remaining_budget = {
                    int(m): max(
                        0,
                        self.config.n_max_events - events_used_per_month.get(int(m), 0),
                    )
                    for m in np.unique(month_ids_w)
                }

                self.dr_model = self._build_dr_model(
                    window_start=window_start,
                    window_len=window_len,
                    init_soc=self.updated_initial_soc,
                    remaining_budget=remaining_budget,
                )
                self.problem_state = DispatchProblemState()

                self.solve_dispatch_model(
                    start_time=window_start,
                    n_days=self.n_timesteps // 24,
                )

                for t in range(window_len):
                    if pyomo.value(self.dr_model.discharge[t]) > 0.5:
                        m = int(month_ids_w[t])
                        events_used_per_month[m] = events_used_per_month.get(m, 0) + 1

                storage_out_window, soc_window = performance_model(
                    self.storage_dispatch_commands,
                    **performance_model_kwargs,
                    sim_start_index=window_start,
                )

                # Performance model returns SOC in percent.
                self.updated_initial_soc = soc_window[-1] / 100.0

                for j in range(window_len):
                    storage_out[window_start + j] = storage_out_window[j]
                    soc_out[window_start + j] = soc_window[j]

            print(
                f"  Events per month: {dict(events_used_per_month)} "
                f"(limit: {self.config.n_max_events}/month)"
            )
            return storage_out, soc_out

        return pyomo_dispatch_solver

    def initialize_parameters(self, inputs):
        """Sync OpenMDAO inputs into the config.

        Args:
            inputs (dict): OpenMDAO inputs dict. Recognised keys are
                ``'max_charge_rate'`` and ``'storage_capacity'``.
        """
        if "max_charge_rate" in inputs:
            object.__setattr__(
                self.config, "max_charge_rate", float(inputs["max_charge_rate"][0])
            )
        if "storage_capacity" in inputs:
            object.__setattr__(
                self.config, "max_capacity", float(inputs["storage_capacity"][0])
            )

    def _build_dr_model(
        self,
        window_start: int,
        window_len: int,
        init_soc: float,
        remaining_budget: dict,
    ) -> pyomo.ConcreteModel:
        """Build the DR MILP for a single rolling window.

        Args:
            window_start (int): Global timestep index of the first hour
                in this window.
            window_len (int): Number of timesteps in this window
                (``n_control_window`` except possibly the last window).
            init_soc (float): State-of-charge fraction at the start of
                this window.
            remaining_budget (dict): Mapping of ``month_id (int)`` to
                remaining event slots for that month. Computed by
                subtracting events already dispatched in earlier windows
                from ``n_max_events``, so the monthly cap is respected
                across windows.

        Returns:
            pyomo.ConcreteModel: Fully formed MILP ready to solve.
        """
        m = pyomo.ConcreteModel(name="plm_dr")

        P_max = self.config.max_charge_rate
        E_max = self.config.max_capacity * (
            self.config.max_soc_fraction - self.config.min_soc_fraction
        )
        eta_c = self.config.charge_efficiency
        eta_d = self.config.discharge_efficiency
        soc_max = self.config.max_soc_fraction
        soc_min = self.config.min_soc_fraction
        incentive = self.config.performance_incentive
        N_max = self.config.n_max_events

        w = slice(window_start, window_start + window_len)
        in_peak_window_w = self.in_peak_window[w]
        month_ids_w = self.month_ids[w]
        signal_w = np.asarray(self.config.supervisory_signal, dtype=float)[w]
        eligible_t_w = self._compute_eligible_mask(signal_w)

        months_in_window = np.unique(month_ids_w).tolist()

        m.T = pyomo.Set(initialize=range(window_len), doc="Timesteps in window")
        m.M = pyomo.Set(initialize=months_in_window, doc="Months in window")

        m.discharge = pyomo.Var(m.T, domain=pyomo.Binary, doc="Discharge binary: 1 = discharging at timestep t")
        m.charge = pyomo.Var(m.T, domain=pyomo.Binary, doc="Charge binary: 1 = charging at timestep t")
        m.soc = pyomo.Var(
            m.T,
            domain=pyomo.NonNegativeReals,
            bounds=(soc_min, soc_max),
            doc="State of charge SoC_t",
        )

        m.objective = pyomo.Objective(
            expr=-incentive * P_max * sum(m.discharge[t] for t in m.T),
            sense=pyomo.minimize,
        )

        m.peak_window_only = pyomo.Constraint(
            m.T,
            rule=lambda mdl, t: (
                mdl.discharge[t] == 0 if not in_peak_window_w[t] else pyomo.Constraint.Skip
            ),
        )

        m.high_signal_only = pyomo.Constraint(
            m.T,
            rule=lambda mdl, t: mdl.discharge[t] <= int(eligible_t_w[t]),
        )

        def max_events_rule(mdl, month):
            ts_in_month = [t for t in mdl.T if month_ids_w[t] == month]
            if not ts_in_month:
                return pyomo.Constraint.Skip
            budget = remaining_budget.get(month, N_max)
            return sum(mdl.discharge[t] for t in ts_in_month) <= budget

        m.max_events = pyomo.Constraint(m.M, rule=max_events_rule)

        m.soc_init = pyomo.Constraint(expr=m.soc[0] == init_soc)

        def soc_evolution_rule(mdl, t):
            if t == 0:
                return pyomo.Constraint.Skip
            return mdl.soc[t] == (
                mdl.soc[t - 1]
                + eta_c * mdl.charge[t] * P_max / E_max
                - mdl.discharge[t] * P_max / (eta_d * E_max)
            )

        m.soc_evolution = pyomo.Constraint(m.T, rule=soc_evolution_rule)

        m.no_simultaneous = pyomo.Constraint(
            m.T,
            rule=lambda mdl, t: mdl.discharge[t] + mdl.charge[t] <= 1,
        )

        m.no_charge_in_window = pyomo.Constraint(
            m.T,
            rule=lambda mdl, t: (
                mdl.charge[t] == 0 if in_peak_window_w[t] else pyomo.Constraint.Skip
            ),
        )

        return m

    def solve_dispatch_model(self, start_time: int = 0, n_days: int = 0):
        """Solve the DR MILP for the current window and record solver metrics.

        Args:
            start_time (int): Timestep index of the window start.
                Used only for error messages and metrics. Defaults to 0.
            n_days (int): Total simulation days. Passed to
                ``DispatchProblemState.store_problem_metrics``.
                Defaults to 0.

        Raises:
            RuntimeError: If GLPK returns a non-OK status or an
                unacceptable termination condition.
        """
        from pyomo.opt import SolverStatus, TerminationCondition

        solver_results = self.glpk_solve_call(self.dr_model)

        status = solver_results.solver.status
        tc = solver_results.solver.termination_condition
        acceptable = (
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            TerminationCondition.maxTimeLimit,
        )
        if status != SolverStatus.ok or tc not in acceptable:
            raise RuntimeError(
                f"PLM MILP solver failed at window start={start_time}: "
                f"status={status}, termination={tc}. "
                f"init_soc={self.updated_initial_soc:.4f}, "
                f"window_len={len(list(self.dr_model.T))}"
            )
        if tc == TerminationCondition.maxTimeLimit:
            print(
                f"  WARNING: solver hit time limit at window start={start_time} "
                f"— using best solution found so far"
            )

        self.problem_state.store_problem_metrics(
            solver_results,
            start_time,
            n_days,
            pyomo.value(self.dr_model.objective),
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Build the DR dispatch solver and write it to discrete outputs.

        Args:
            inputs (dict): OpenMDAO continuous inputs.
            outputs (dict): OpenMDAO continuous outputs.
            discrete_inputs (dict): OpenMDAO discrete inputs.
            discrete_outputs (dict): OpenMDAO discrete outputs. The key
                ``'pyomo_dispatch_solver'`` is set to the callable
                returned by :meth:`pyomo_setup`.
        """
        discrete_outputs["pyomo_dispatch_solver"] = self.pyomo_setup(discrete_inputs)

    @staticmethod
    def glpk_solve_call(
        pyomo_model: pyomo.ConcreteModel,
        log_name: str = "",
        user_solver_options: dict | None = None,
    ):
        """Solve a Pyomo MILP with GLPK.

        Args:
            pyomo_model (pyomo.ConcreteModel): The model to solve.
            log_name (str): Optional log file name passed to
                ``SolverOptions``. Defaults to ``''``.
            user_solver_options (dict | None): Optional overrides for
                GLPK solver options. Defaults to ``None``.

        Returns:
            pyomo.opt.SolverResults: Raw results object from GLPK.
        """
        glpk_solver_options = {"cuts": None, "presol": None, "tmlim": 300}
        solver_options = SolverOptions(
            glpk_solver_options, log_name, user_solver_options, "log"
        )
        with pyomo.SolverFactory("glpk") as solver:
            results = solver.solve(
                pyomo_model, options=solver_options.constructed, tee=False
            )
        return results

    @property
    def storage_dispatch_commands(self) -> list:
        """Net dispatch commands for the solved window.

        Returns:
            list[float]: ``(u_t - v_t) * P_max`` for each timestep in
            the solved window. Positive = discharge, negative = charge.
        """
        P_max = self.config.max_charge_rate
        return [
            (pyomo.value(self.dr_model.discharge[t]) - pyomo.value(self.dr_model.charge[t])) * P_max
            for t in self.dr_model.T
        ]
