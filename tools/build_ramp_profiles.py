"""Build explicitly provisional 10 C/min feed-forward profiles from exported ramps.

This tool reads experiments; it never connects to instruments or changes source runs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ramp_points(data, start, stop, width):
    """Average entire target-temperature bins, using RMS commanded current.

    Bin by target rather than instantaneous measured T to include both phases
    of an oscillation. Pair RMS current with mean filtered measured temperature.
    RMS preserves average I^2 heating approximately within each narrow bin.
    """
    points, evidence = [], []
    for low in np.arange(start, stop, width):
        group = data[(data.set_T >= low) & (data.set_T < low + width)]
        group = group[np.isfinite(group["T"]) & np.isfinite(group.C_V) & (group.C_V > 0)]
        if len(group) < 20:
            continue
        point = {"temperature_c": round(float(group["T"].mean()), 3),
                 "current_a": round(float(np.sqrt(np.mean(group.C_V ** 2))), 6)}
        points.append(point)
        evidence.append({"target_bin_c": [float(low), float(low + width)], "samples": len(group),
                         "mean_tracking_error_c": round(float((group["T"] - group.set_T).mean()), 3)})
    if len(points) < 2 or any(b["temperature_c"] <= a["temperature_c"] for a,b in zip(points,points[1:])):
        raise ValueError("Insufficient ordered ramp data for a feed-forward table.")
    return points, evidence


def build_profile(source, run_name, profile_name, nicr):
    csv = source / run_name / "data.csv"
    data = pd.read_csv(csv)
    # NiCr excludes the unreliable startup/range change and all late collapse data.
    points, evidence = ramp_points(data, 100 if nicr else 30, 260 if nicr else 120, 20 if nicr else 10)
    profile = json.loads((source / (profile_name + ".json")).read_text(encoding="utf-8"))
    kp = .0001 if nicr else .001
    profile.update(
        profile_name=profile_name, controller_mode="PI", pid_kp=kp, pid_ki=kp/100,
        pid_integral_time_s=100., pid_kd=0., pid_gain_schedule=[],
        current_feedforward_table=[{"temperature_c": 23., "current_a": .01}] + points,
        current_feedforward_provenance={
            "kind": "provisional_ramp", "ramp_rate_c_min": 10.,
            "run": run_name, "data_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
            "method": "RMS commanded current per target-temperature bin, paired with mean filtered measured T",
            "derived_temperature_range_c": [points[0]["temperature_c"], points[-1]["temperature_c"]],
            "startup_anchor": "23 C / 10 mA is the recorded startup command, not a measured equilibrium point",
            "limitations": "Ramp-derived bias only; not equilibrium calibration or validated tuning. Endpoints clamp. Recalibrate for the actual wire and mounting.",
            "bins": evidence,
        },
        trial_max_temperature_c=250. if nicr else 110.,
        experiment_frequency=.5, max_current_step_up=.001, max_current_step_down=.001,
        minimum_current_change=.001, min_current=0.,
        startup_current=.01, measurement_current_floor=.01,
        t0_current_search_start=.01, tuning_start_current=.01, startup_settle_time_s=2.,
        temperature_prediction_time_s=0., temperature_rate_window_s=8.,
        measurement_filter_samples=3, pid_tracking_time_s=30.,
        dmm_voltage_range_v=2., dmm_current_range_a=.02, DMM_speed=10,
        dmm_synchronized_reading=True, dmm_staged_ranging_enabled=True,
        dmm_range_settle_time_s=.5, dmm_range_discard_readings=3, current_settle_time_s=.5,
        measurement_temperature_jump_guard_enabled=False, measurement_resistance_retry_enabled=True,
        measurement_retry_temperature_jump_c=8., measurement_retry_temperature_consensus_c=5.,
        measurement_retry_attempts=3, measurement_retry_delay_s=.2,
        measurement_retry_consensus_ohm=.015, resistance_glitch_jump_ohm=.03,
        resistance_glitch_jump_ratio=.015, invalid_measurement_policy="hold", measurement_fail_limit=10,
        resistance_power_guard_enabled=True, resistance_power_guard_window_s=30.,
        resistance_power_guard_drop_c=15., resistance_power_guard_power_ratio=1.2,
        resistance_power_guard_min_current_a=.02,
        curve_extrapolation_enabled=False,
        # Bounded limits for this next trial, not inferred wire ratings.
        max_current=.1, max_power_w=.25 if nicr else .05,
    )
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [("136_NiCr_100_hydrogen_cur_3_10k_min", "NiCr_100_163", True),
            ("139_Ni100_uncharged_test_1", "Ni_100_152", False)]
    for run, name, nicr in runs:
        profile=build_profile(args.source,run,name,nicr)
        (args.output/(name+".json")).write_text(json.dumps(profile,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        print(name, json.dumps(profile["current_feedforward_table"]))

if __name__ == "__main__":
    main()
