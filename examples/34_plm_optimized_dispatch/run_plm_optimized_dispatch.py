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
    model.technology_config["technologies"]["battery"]["model_inputs"][
        "control_parameters"
    ]["supervisory_signal"]
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

peak_mask = (time_index.hour >= 14) & (time_index.hour <= 18)
threshold_pct = np.percentile(lmp, percentile)
discharge_mask = battery_power > 0


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
days = pd.date_range(time_index[0].normalize(), periods=14, freq="D", tz=time_index.tz)
time_window = 14 * 24


def shade_peaks(ax):
    for day in days:
        ax.axvspan(
            day + pd.Timedelta(hours=14),
            day + pd.Timedelta(hours=18),
            color="orange",
            alpha=0.10,
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
