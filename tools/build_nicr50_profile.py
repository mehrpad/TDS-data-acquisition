"""Create an explicitly unmeasured 50 um NiCr trial profile from the 100 um donor.

No instruments are accessed. Scaling assumes the same alloy and similar mounting;
the reference resistance scale assumes the donor's length until T0 calibration.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path


PROFILE_NAME = "NiCr_50_provisional"
CURRENT_SCALE = .30
RESISTANCE_SCALE = 4.
DESIGN_REFERENCE = "https://prodshop.kanthal.com/en/knowledge-hub/heating-material-knowledge/design-calculations-and-standard-tolerances/design-calculations/"


def build_profile(donor, source_curve):
    if donor.get("profile_name") != "NiCr_100_163":
        raise ValueError("Expected the existing 100 um NiCr profile, not a Ni profile.")
    source_hash = hashlib.sha256(source_curve.read_bytes()).hexdigest()
    source = donor["current_feedforward_provenance"]
    if source_hash != source["r_vs_t_source_csv_sha256"]:
        raise ValueError("Reference curve does not match the donor NiCr profile.")
    result = copy.deepcopy(donor)
    result.update(
        profile_name=PROFILE_NAME, startup_current=.002, measurement_current_floor=.002,
        t0_current_search_start=.002, tuning_start_current=.002,
        pid_kp=.00003, pid_ki=.0000003, pid_integral_time_s=100., pid_kd=0.,
        pid_gain_schedule=[], pid_integral_current_limit_a=.01,
        measurement_filter_samples=3, temperature_prediction_time_s=2.,
        invalid_measurement_policy="backoff", measurement_retry_consensus_ratio=.005,
        resistance_power_guard_min_current_a=.005,
        max_current=.10, max_power_w=.50, max_sample_voltage=20., compliance_voltage=30.,
        max_temperature_c=600., trial_max_temperature_c=600.,
        curve_extrapolation_enabled=True, curve_extrapolation_max_temperature_c=600.,
    )
    table = [{"temperature_c": float(p["temperature_c"]),
              "current_a": round(float(p["current_a"])*CURRENT_SCALE, 9)}
             for p in donor["current_feedforward_table"]]
    # Startup is a lower measurement command, not an inferred room-temperature hold.
    table[0]["current_a"] = result["startup_current"]
    if (table[0]["temperature_c"] != 23 or table[-1]["temperature_c"] != 600
            or any(b["temperature_c"] <= a["temperature_c"] or b["current_a"] < a["current_a"]
                   for a, b in zip(table, table[1:]))):
        raise ValueError("Donor cannot provide an ordered 23..600 C trial table.")
    result["current_feedforward_table"] = table
    result["current_feedforward_provenance"] = {
        "kind": "provisional_ramp", "estimate_type": "diameter_scaling",
        "measured_on_this_wire": False, "measured_temperature_range_c": [],
        "estimated_temperature_range_c": [23., 600.], "ramp_rate_c_min": 10.,
        "source_profile": donor["profile_name"], "source_run": source["run"],
        "source_profile_sha256": hashlib.sha256(json.dumps(donor, sort_keys=True).encode()).hexdigest(),
        "source_reference_curve_sha256": source_hash,
        "wire_diameter_um": 50., "wire_length_known": False,
        "reference_wire_diameter_um": 100., "reference_wire_length_mm": 163.,
        "current_scale_factor": CURRENT_SCALE, "resistance_scale_factor": RESISTANCE_SCALE,
        "method": "0.30 times the 100 um NiCr ramp bias at each temperature, with a lower 2 mA startup anchor",
        "assumptions": "Same NiCr alloy, similar surroundings and mounting. Exact active length is unknown. The R(T) scale initially assumes the same 163 mm length; T0 calibration must rescale to the actual cold resistance.",
        "scaling_rationale": "For a halved diameter, R scales by 4 at fixed length. Surface-loss-dominated heating gives I proportional to d^1.5 (factor 0.354); end-conduction or ramp thermal mass can give factor 0.25. Factor 0.30 is an engineering starting choice within these simplified estimates, not a measured scaling law for this apparatus.",
        "design_reference": DESIGN_REFERENCE,
        "reference_curve_file": PROFILE_NAME + "_R_vs_T_estimated.csv",
        "source_curve_temperature_bounds_c": source["source_curve_temperature_bounds_c"],
        "startup_anchor": "23 C / 2 mA chosen for startup and measurement; not measured equilibrium",
        "unmeasured_extension": {"range_c": [23., 600.], "measured": False,
            "method": "All points estimated by diameter scaling; donor itself extrapolates above about 500 C"},
        "electrical_limits": {"max_current_a": .10, "max_power_w": .50,
            "max_sample_voltage_v": 20., "compliance_voltage_v": 30.,
            "method": "Initial trial headroom above the estimated 25 mA bias; software limits, not verified wire ratings"},
        "limitations": "No 50 um NiCr heating measurements or validated PI tuning. Assumed matching alloy is essential; cold-resistance calibration cannot fix a different TCR. The donor R(T) ends at 293.4 C; higher indicated temperatures are extrapolated. Earlier NiCr calibration scatter and resistance drift remain unresolved. No guarantee of physical temperature accuracy or smooth tracking to 600 C. Replace these estimates after measurement.",
    }
    return result


def write_outputs(profile, source_curve, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/(PROFILE_NAME + ".json")).write_text(json.dumps(profile, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    with (output/(PROFILE_NAME + "_current_table.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["temperature_c", "current_a", "measured_on_this_wire"])
        writer.writeheader()
        writer.writerows({**point, "measured_on_this_wire": False} for point in profile["current_feedforward_table"])
    with source_curve.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (output/(PROFILE_NAME + "_R_vs_T_estimated.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["resistivity", "temperature", "measured_on_this_wire"])
        writer.writeheader()
        writer.writerows({"resistivity": float(row["resistivity"])*RESISTANCE_SCALE,
                          "temperature": float(row["temperature"]), "measured_on_this_wire": False} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-curve", type=Path, required=True)
    parser.add_argument("--donor", type=Path, default=Path("files/material_profiles/NiCr_100_163.json"))
    parser.add_argument("--output", type=Path, default=Path("files/material_profiles"))
    args = parser.parse_args()
    profile = build_profile(json.loads(args.donor.read_text(encoding="utf-8")), args.source_curve)
    write_outputs(profile, args.source_curve, args.output)
    print(PROFILE_NAME, "all points estimated; 600 C bias", profile["current_feedforward_table"][-1]["current_a"], "A")


if __name__ == "__main__":
    main()
