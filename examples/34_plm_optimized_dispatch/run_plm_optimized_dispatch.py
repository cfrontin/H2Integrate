"""

This example demonstrates demand-response storage dispatch using a rolling-horizon
MILP controller. The battery is scheduled to discharge during high-LMP peak hours
to reduce facility demand charges and earn performance incentives.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel


EXAMPLE_DIR = Path(__file__).parent


model = H2IntegrateModel(EXAMPLE_DIR / "34_plm_optimized_dispatch.yaml")
model.setup()

N = model.plant_config["plant"]["simulation"]["n_timesteps"]
percentile = model.technology_config["technologies"]["battery"]["model_inputs"][
    "control_parameters"
]["signal_threshold_percentile"]

model.run()

lmp = np.array(
    model.technology_config["technologies"]["battery"]["model_inputs"]["control_parameters"][
        "supervisory_signal"
    ]
)[:N]

sim = model.plant_config["plant"]["simulation"]
n_timesteps = int(sim["n_timesteps"])
dt_seconds = int(sim["dt"])
tz = int(sim["timezone"])
start = pd.Timestamp(sim["start_time"], tz=tz)
freq = pd.to_timedelta(dt_seconds, unit="s")
time_index = pd.date_range(start=start, periods=n_timesteps, freq=freq)

battery_power = model.prob.get_val("battery.storage_electricity_discharge", units="kW")
soc_pct = model.prob.get_val("battery.SOC", units="percent")

control_params = model.technology_config["technologies"]["battery"]["model_inputs"][
    "control_parameters"
]
pw_cfg = control_params["peak_window"]
event_dur_cfg = control_params.get("event_duration")


def _peak_window_to_hour(val) -> int:
    if isinstance(val, int | float):
        return int(val) // 3600
    return int(str(val).split(":")[0])


pw_start_h = _peak_window_to_hour(pw_cfg["start"])
pw_end_h = _peak_window_to_hour(pw_cfg["end"])

half_td = None
if event_dur_cfg is not None:
    half_td = pd.Timedelta(value=event_dur_cfg["val"], unit=event_dur_cfg["units"]) / 2

threshold_pct = np.percentile(lmp, percentile)
discharge_mask = battery_power > 0


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
days = pd.date_range(time_index[0].normalize(), periods=14, freq="D", tz=time_index.tz)
time_window = min(n_timesteps, int(14 * 24 * 3600 / dt_seconds))  # 14 days


def shade_peaks(ax):
    for day in days:
        # light background: static peak_window
        ax.axvspan(
            day + pd.Timedelta(hours=pw_start_h),
            day + pd.Timedelta(hours=pw_end_h),
            color="orange",
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )
        if half_td is None:
            continue
        # darker band: event window centered on the daily LMP peak
        pw_start_ts = day + pd.Timedelta(hours=pw_start_h)
        pw_end_ts = day + pd.Timedelta(hours=pw_end_h)
        in_pw = (time_index >= pw_start_ts) & (time_index <= pw_end_ts)
        if not in_pw.any():
            continue
        peak_idx = np.where(in_pw)[0][np.argmax(lmp[in_pw])]
        peak_ts = time_index[peak_idx]
        ax.axvspan(
            peak_ts - half_td,
            peak_ts + half_td,
            color="darkorange",
            alpha=0.30,
            linewidth=0,
            zorder=0,
        )


w_discharge = discharge_mask[:time_window]

ax = axes[0]
shade_peaks(ax)
ax.plot(time_index[:time_window], lmp[:time_window], color="steelblue", linewidth=1.0)
ax.axhline(threshold_pct, color="k", linestyle="--", linewidth=0.8)
ax.plot(
    time_index[:time_window][w_discharge],
    lmp[:time_window][w_discharge],
    "r*",
    markersize=8,
    zorder=5,
)
ax.set_ylabel("LMP ($/MWh)", fontsize=8)
ax.set_ylim(bottom=0)

ax = axes[1]
shade_peaks(ax)
ax.plot(time_index[:time_window], soc_pct[:time_window], color="g", linewidth=1.0)
ax.axhline(90, color="gray", linestyle=":", linewidth=0.7)
ax.axhline(10, color="gray", linestyle=":", linewidth=0.7)
ax.set_ylabel("SOC (%)", fontsize=8)
ax.set_ylim([0, 105])


plt.tight_layout()
plt.savefig("plm_optimized_dispatch.png", dpi=150, bbox_inches="tight")
