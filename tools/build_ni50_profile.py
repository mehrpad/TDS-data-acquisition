"""Build a provisional 50 um Ni profile from run 117, without instrument access."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.build_ramp_profiles import ramp_points


PROFILE_NAME = "Ni_50_200"


def build_profile(run, template):
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    program = metadata.get("experiment_program", [])
    if (metadata.get("experiment_mode") != "TEMPERATURE" or len(program) != 1
            or program[0].get("ramp_speed_min") != 10):
        raise ValueError("Expected run 117's single 10 C/min temperature ramp.")
    data = pd.read_csv(run / "data.csv")
    diagnostics = pd.read_json(run / "control_diagnostics.jsonl", lines=True)
    if len(data) != len(diagnostics) or not np.all(np.abs(data.time - diagnostics.time) < .1):
        raise ValueError("CSV and diagnostics must align.")
    valid = np.isfinite(data[["time", "T", "set_T", "C_V", "I", "P"]]).all(axis=1)
    valid &= (diagnostics.status == "accepted") & (data.I > 0) & (data.C_V > 0) & (data.P > 0)
    valid &= data.time - data.time.iloc[0] >= 120
    for i in np.flatnonzero(diagnostics.range_change_count.diff().fillna(0).to_numpy() > 0):
        valid.iloc[i:i+3] = False
    # Whole target bins include both phases of the growing oscillation, avoiding
    # a bias toward hot/low-current or cool/high-current instantaneous samples.
    points, bins = ramp_points(data[valid], 40, 190, 10)
    if len(points) < 10 or points[-1]["temperature_c"] < 175:
        raise ValueError("Insufficient accepted ramp coverage for this profile.")
    currents = np.array([p["current_a"] for p in points])
    monotone = np.maximum.accumulate(currents)
    correction = float(np.max(monotone - currents))
    if correction > .001:
        raise ValueError("More than one current step of monotonic correction; review the run.")
    for point, current in zip(points, monotone):
        point["current_a"] = float(current)
    fit = points[-8:]
    slope = float(np.polyfit([p["temperature_c"] for p in fit],
                             [p["current_a"] ** 2 for p in fit], 1)[0])
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("Cannot construct a positive I squared extension.")
    last = points[-1]
    table = [{"temperature_c": 23., "current_a": .005}] + points
    for temperature in (200., 250., 300., 350., 400., 450., 500., 550., 600.):
        table.append({"temperature_c": temperature, "current_a": round(float(
            np.sqrt(last["current_a"] ** 2 + slope * (temperature - last["temperature_c"]))), 6)})
    result = copy.deepcopy(template)
    result.update(
        profile_name=PROFILE_NAME, current_feedforward_table=table,
        pid_kp=.0002, pid_ki=.000002, pid_integral_time_s=100., pid_kd=0.,
        pid_gain_schedule=[], pid_integral_current_limit_a=.02,
        measurement_filter_samples=1, temperature_prediction_time_s=2.,
        startup_current=.005, measurement_current_floor=.005,
        t0_current_search_start=.005, tuning_start_current=.005,
        measurement_retry_consensus_ratio=.005, invalid_measurement_policy="backoff",
        max_current=5., max_power_w=20., max_sample_voltage=20., compliance_voltage=30.,
        max_temperature_c=600., trial_max_temperature_c=600.,
        curve_extrapolation_enabled=True, curve_extrapolation_max_temperature_c=600.,
    )
    result["current_feedforward_provenance"] = {
        "kind": "provisional_ramp", "ramp_rate_c_min": 10.,
        "run": "117_Ni200_50_uncharged_cur_1", "wire_diameter_um": 50.,
        "method": "RMS applied command per complete target bin, paired with mean accepted indicated T",
        "selection": "Accepted finite positive samples; exclude first 120 s, invalid feedback and range-change cycle plus next two cycles",
        "derived_temperature_range_c": [points[0]["temperature_c"], last["temperature_c"]],
        "bins": bins, "monotonic_current_correction_a": correction,
        "startup_anchor": "23 C / 5 mA chosen as a reduced startup command; not measured equilibrium. Interpolation below first derived bin is provisional.",
        "unmeasured_extension": {"range_c": [last["temperature_c"], 600.], "measured": False,
            "method": "Linear I squared versus indicated T over last eight ramp bins, anchored at last derived point",
            "slope_a2_per_c": slope, "fit_range_c": [fit[0]["temperature_c"], fit[-1]["temperature_c"]]},
        "source_curve_temperature_bounds_c": json.loads((run / "curve_metadata.json").read_text())["source_temperature_bounds_c"],
        "reference_curve_file": PROFILE_NAME + "_R_vs_T.csv",
        "electrical_limits": {"max_current_a": 5., "max_power_w": 20.,
            "max_sample_voltage_v": 20., "compliance_voltage_v": 30.,
            "method": "Retain run 117 current/power ceilings and add sample voltage headroom at user request; not validated wire ratings"},
        "limitations": "Ramp-derived bias, not equilibrium calibration. Run stopped around 190 C; higher current biases are estimates. Source R(T) ends at 283.66 C; higher indicated temperatures are extrapolated. Lower PI gains are initial trial settings, not validated tuning. No guarantee of smooth tracking or reaching 600 C.",
    }
    for name in ("data.csv", "control_diagnostics.jsonl", "run_metadata.json", "r_vs_t_source.csv"):
        result["current_feedforward_provenance"][name.replace('.', '_') + "_sha256"] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("files/material_profiles"))
    parser.add_argument("--template", type=Path, default=Path("files/material_profiles/Ni_100_152.json"))
    args = parser.parse_args()
    profile = build_profile(args.run, json.loads(args.template.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / (PROFILE_NAME + ".json")).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(profile["current_feedforward_table"]).assign(
        derived_from_run=lambda frame: (frame.temperature_c >= profile["current_feedforward_provenance"]["derived_temperature_range_c"][0])
            & (frame.temperature_c <= profile["current_feedforward_provenance"]["derived_temperature_range_c"][1])
    ).to_csv(args.output / (PROFILE_NAME + "_current_table.csv"), index=False)
    (args.output / (PROFILE_NAME + "_R_vs_T.csv")).write_bytes((args.run / "r_vs_t_source.csv").read_bytes())
    print(PROFILE_NAME, len(profile["current_feedforward_table"]), "points; 600 C estimate", profile["current_feedforward_table"][-1])


if __name__ == "__main__":
    main()
