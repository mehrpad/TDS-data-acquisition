import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

from . import pid
from . import siglent


CONTROL_DEFAULTS = {
    "controller_mode": "PI",
    "experiment_mode": "TEMPERATURE",
    "compliance_voltage": 30.0,
    "max_current": 1.0,
    "max_sample_voltage": 15.0,
    "max_power_w": 10.0,
    "dmm_voltage_range_v": 20.0,
    "dmm_current_range_a": 2.0,
    "dmm_staged_ranging_enabled": True,
    "dmm_range_switch_fraction": 0.8,
    "dmm_range_settle_time_s": 0.3,
    "dmm_range_discard_readings": 2,
    "dmm_range_recovery_attempts": 5,
    "resistivity_mode": "V_OVER_I",
    "resistivity_heat_time_s": 4.0,
    "resistivity_measure_time_s": 1.0,
    "resistivity_output_settle_s": 0.3,
    "dmm_resistance_range_ohm": 200.0,
    "pid_kp": 0.0004,
    "pid_ki": 0.00002,
    "pid_kd": 0.0,
    # Optional multi-point tuning result: a list of {current_a, kp, ki, kd}
    # points, sorted ascending by current_a. When set, the controller
    # interpolates gains by present_current instead of using the flat
    # pid_kp/pid_ki/pid_kd above. Empty by default (single fixed-gain PID).
    "pid_gain_schedule": [],
    "pid_integral_limit": 400.0,
    "pid_derivative_filter": 0.6,
    "startup_current": 0.005,
    "min_current": 0.0,
    "psu_keepalive_current": 0.001,
    "fixed_series_resistance_ohm": 0.0,
    "max_current_step_up": 0.01,
    "max_current_step_down": 0.01,
    "temperature_tolerance_c": 2.0,
    "hold_entry_tolerance_c": 3.0,
    "safety_temp_margin_c": 15.0,
    "soft_temp_rate_margin_c_min": 1.0,
    "hard_temp_rate_margin_c_min": 4.0,
    "measurement_fail_limit": 20,
    "minimum_current_a": 1e-4,
    "minimum_current_change": 0.001,
    "measurement_current_floor": 0.01,
    "measurement_filter_samples": 3,
    "dmm_synchronized_reading": True,
    "resistance_outlier_mad_multiplier": 4.0,
    "resistance_outlier_min_ohm": 0.0005,
    "stable_resistance_spread_ratio": 0.005,
    "stable_resistance_spread_ohm": 0.03,
    "startup_settle_time_s": 1.0,
    "resistance_range_margin_ratio": 0.0,
    "resistance_range_margin_ohm": 0.01,
    "curve_extrapolation_enabled": False,
    "curve_extrapolation_min_temperature_c": 0.0,
    "curve_extrapolation_max_temperature_c": 1000.0,
    "curve_extrapolation_fit_points": 20,
    "curve_extrapolation_min_fit_span_c": 5.0,
    "curve_monotonic_correction_ratio": 0.02,
    "curve_extrapolation_max_monotonic_correction_ratio": 0.02,
    "curve_smoothing_enabled": True,
    "curve_smoothing_min_points": 100,
    "curve_smoothing_temperature_bin_c": 0.5,
    "curve_smoothing_max_window_c": 15.0,
    "curve_smoothing_max_residual_ratio": 0.10,
    "resistance_glitch_jump_ohm": 0.03,
    "resistance_glitch_jump_ratio": 0.015,
    "measurement_retry_attempts": 2,
    "measurement_retry_delay_s": 0.15,
    "measurement_retry_consensus_ohm": 0.015,
    "stable_current_invalid_advance_count": 5,
    # When False, every reading trusts the resistance-derived temperature
    # directly: skips the low-signal/temperature-jump confirmation machinery
    # (holding, probing, NaN rows while unconfirmed) and the resistance-glitch
    # retry/reject in _measure_with_retry. Nothing measured is ever discarded
    # or replaced with a stale value - the current slew-rate limit
    # (max_current_step_up/down) is the only thing bounding how fast control
    # can react to a reading. Default false: repeated field use found the
    # confirmation machinery's own hold-and-probe behavior (stale dataset
    # rows, stalled control while waiting on consensus, several distinct
    # bugs in the probing itself) cost more than the protection was worth,
    # given the slew limit already caps how far one reading can move the
    # command. Set true to restore it.
    "measurement_temperature_jump_guard_enabled": False,
    "measurement_temp_jump_c": 8.0,
    "measurement_temp_jump_up_c": 20.0,
    "measurement_temp_jump_down_c": 8.0,
    "measurement_jump_confirm_min_current_a": 0.02,
    # Was 0.1: a voltage-era threshold carried over unchanged when this key
    # was renamed from measurement_jump_confirm_min_voltage during the
    # constant-current conversion. Combined with the old ignore_invalid_
    # below_current * 2.0 floor (also removed), 0.1 A sat above where a
    # sensitive low-resistance wire actually operates, permanently disabling
    # jump confirmation there: a real, sustained temperature rise could never
    # be confirmed once it drifted from a stale trusted value, and the
    # controller would ignore it indefinitely (see docs/MEASUREMENT_SETUP.md).
    "measurement_jump_confirm_min_current": 0.02,
    "measurement_temp_jump_accept_up_c": 35.0,
    "measurement_temp_jump_accept_setpoint_margin_c": 15.0,
    "low_signal_jump_confirm_samples": 3,
    "low_signal_jump_temperature_tolerance_c": 10.0,
    "low_signal_jump_resistance_tolerance_ohm": 0.015,
    "low_signal_recovery_trigger_cycles": 5,
    "low_signal_recovery_observe_cycles": 5,
    "low_signal_recovery_current_step": 0.01,
    "low_signal_recovery_max_attempts": 5,
    "measurement_cooldown_confirm_samples": 2,
    "measurement_heatup_confirm_samples": 2,
    "measurement_jump_probe_threshold_c": 35.0,
    "measurement_jump_probe_current_step": 0.002,
    "measurement_jump_probe_temperature_tolerance_c": 50.0,
    "measurement_jump_probe_resistance_ratio": 0.02,
    "measurement_jump_probe_max_samples": 20,
    "ignore_invalid_below_current": 0.05,
    "invalid_current_step_down": 0.01,
    "invalid_reuse_hold_after": 8,
    "invalid_max_drop_from_recent_peak_a": 0.1,
    "invalid_reuse_stop_after": 30,
    "rate_limit_activation_band_c": 2.0,
    "under_target_no_decrease_band_c": 1.5,
    "autosave_flush_interval_s": 5.0,
    "autosave_batch_size": 10,
    "tuning_current_step": 0.001,
    "tuning_start_current": 0.005,
    # Effectively unbounded: max_current is the real ceiling. A separate,
    # lower number here used to cut the geometric retry climb short well
    # below what max_current would otherwise allow.
    "tuning_search_max_current": 1000.0,
    "tuning_settle_time_s": 0.3,
    "tuning_response_current_step": 0.01,
    # Minimum step size as a fraction of the current baseline/candidate, so
    # the induced excitation stays meaningful in absolute power (P ~ I^2)
    # however small or large that baseline is, rather than being capped at
    # a fixed 0.01 A that is negligible power near a low baseline.
    "tuning_response_relative_step": 0.5,
    "tuning_between_attempts_s": 0.5,
    "tuning_max_duration_s": 300.0,
    "tuning_baseline_samples": 2,
    "tuning_stable_current_samples": 3,
    "tuning_stable_current_a": 1e-4,
    "tuning_temperature_window_c": 40.0,
    "tuning_target_rise_c": 8.0,
    "tuning_min_temperature_rise_c": 3.0,
    # A high-gain point can cross the rise threshold on its very first sample,
    # before the response curve has had time to reveal its actual shape. With
    # only one or two points, dead_time_s and time_constant_s collapse toward
    # loop_time (see _estimate_pid_from_step), producing a falsely tiny tau
    # and an over-aggressive Ki that then gets clamped across the rest of the
    # schedule's range. Require this many samples before accepting a response
    # as usable, regardless of how quickly the rise threshold was crossed.
    "tuning_min_response_samples": 5,
    # Max rise permitted when jumping a schedule point's baseline toward the
    # next target current, computed from the previous point's own measured
    # gain (see _cap_next_tuning_target). Independent of tuning_target_rise_c,
    # which bounds the rise during one point's own response measurement.
    "tuning_schedule_jump_max_rise_c": 10.0,
    "tuning_no_response_timeout_s": 45.0,
    "tuning_min_observable_rise_c": 0.25,
    "tuning_plateau_timeout_s": 45.0,
    "tuning_plateau_idle_timeout_s": 20.0,
    "max_current_step_up_far": 0.01,
    "aggressive_step_band_c": 4.0,
    "tuning_plateau_growth_c": 0.04,
    "t0_calibration_current": 0.05,
    "t0_current_search_start": 0.005,
    "t0_current_step": 0.001,
    "t0_settle_time_s": 3.0,
    "t0_calibration_samples": 5,
    "t0_warmup_samples": 1,
    "t0_stable_current_samples": 3,
    "t0_stable_current_a": 1e-4,
    "t0_max_temp_error_c": 80.0,
    "t0_temperature_spread_warning_c": 5.0,
}


class ExperimentSafetyError(RuntimeError):
    """Raised when the experiment should stop to protect the sample or setup."""


@dataclass
class TemperatureJumpProbe:
    direction: Optional[str] = None
    candidate_temperature: float = np.nan
    candidate_resistance: float = np.nan
    origin_current: float = np.nan
    confirmations: int = 0
    attempts: int = 0

    @property
    def active(self):
        return self.direction in {"up", "down"}

    def reset(self):
        self.direction = None
        self.candidate_temperature = np.nan
        self.candidate_resistance = np.nan
        self.origin_current = np.nan
        self.confirmations = 0
        self.attempts = 0


@dataclass
class LowSignalTemperatureConfirmation:
    direction: Optional[str] = None
    candidate_temperature: float = np.nan
    candidate_resistance: float = np.nan
    confirmations: int = 0

    @property
    def active(self):
        return self.direction in {"up", "down"}

    def reset(self):
        self.direction = None
        self.candidate_temperature = np.nan
        self.candidate_resistance = np.nan
        self.confirmations = 0


@dataclass
class LowSignalCurrentRecovery:
    active: bool = False
    attempts: int = 0
    invalid_samples_since_step: int = 0

    def reset(self):
        self.active = False
        self.attempts = 0
        self.invalid_samples_since_step = 0


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


def _sample_power_w(measured_voltage, measured_current):
    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current):
        return np.nan
    return abs(float(measured_voltage) * float(measured_current))


def _is_finite_scalar(value):
    """Return False for None/object values instead of letting numpy raise TypeError."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _enforce_electrical_safety(measured_voltage, measured_current, config):
    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current):
        return
    if abs(float(measured_current)) > float(config["max_current"]):
        raise ExperimentSafetyError(
            f"Measured current {measured_current:.4e} A exceeded max_current "
            f"{float(config['max_current']):.4e} A."
        )
    max_sample_voltage = float(config.get("max_sample_voltage", CONTROL_DEFAULTS["max_sample_voltage"]))
    if not np.isfinite(max_sample_voltage) or max_sample_voltage <= 0:
        raise ValueError("max_sample_voltage must be positive and finite.")
    if abs(float(measured_voltage)) > max_sample_voltage:
        # In constant-current mode the supply raises its terminal voltage to hold
        # the set current, so a failing contact shows up here before anywhere else.
        raise ExperimentSafetyError(
            f"Sample voltage {measured_voltage:.4f} V exceeded max_sample_voltage "
            f"{max_sample_voltage:.4f} V at {measured_current:.4e} A. In constant-current "
            "mode this is what an open or degrading contact looks like."
        )
    max_power_w = float(config.get("max_power_w", CONTROL_DEFAULTS["max_power_w"]))
    if not np.isfinite(max_power_w) or max_power_w <= 0:
        raise ValueError("max_power_w must be positive and finite.")
    measured_power_w = _sample_power_w(measured_voltage, measured_current)
    if measured_power_w >= max_power_w:
        raise ExperimentSafetyError(
            f"Measured sample power {measured_power_w:.6f} W exceeded max_power_w "
            f"{max_power_w:.6f} W (Vsample={measured_voltage:.6f} V, "
            f"I={measured_current:.6e} A)."
        )


def _measurement_current_floor(config):
    minimum = float(config["min_current"])
    maximum = float(config["max_current"])
    candidates = (
        minimum,
        float(config.get("measurement_current_floor", minimum)),
        float(config.get("startup_current", minimum)),
        float(config.get("t0_current_search_start", minimum)),
    )
    if not all(np.isfinite(value) for value in candidates) or not np.isfinite(maximum):
        raise ValueError("Initial and minimum voltage settings must be finite.")
    return _clamp(max(candidates), minimum, maximum)


def current_step_scale(config):
    """Scale per-loop voltage limits to the loop period actually in use.

    The step limits are expressed per control cycle but were chosen for the
    1/experiment_frequency period. A duty-cycled resistivity mode runs a much
    longer cycle, so holding the per-loop step fixed would quietly divide the
    achievable ramp rate by the ratio between the two periods. Scaling keeps
    the volts-per-minute the limits were tuned for.
    """
    try:
        reference_period_s = 1.0 / float(config["experiment_frequency"])
        actual_period_s = float(resistivity_loop_time(config))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 1.0
    if not np.isfinite(reference_period_s) or reference_period_s <= 0:
        return 1.0
    if not np.isfinite(actual_period_s) or actual_period_s <= 0:
        return 1.0
    return max(actual_period_s / reference_period_s, 1.0)


def _limit_current_slew(target_current, present_current, min_current, max_current, config):
    if not np.isfinite(target_current) or not np.isfinite(present_current):
        return _clamp(present_current, min_current, max_current)
    step_scale = current_step_scale(config)
    max_step_up = float(config.get("max_current_step_up", 0.01)) * step_scale
    max_step_down = float(config.get("max_current_step_down", 0.01)) * step_scale
    if max_step_up <= 0 or max_step_down <= 0:
        raise ValueError("Voltage slew limits must be positive.")
    delta = target_current - present_current
    if delta > max_step_up:
        target_current = present_current + max_step_up
    elif delta < -max_step_down:
        target_current = present_current - max_step_down
    return _clamp(target_current, min_current, max_current)


def _current_ramp_command(start_current, ramp_speed_min, elapsed_s, applied_current, config):
    """Return the elapsed-time voltage-ramp command after applying the normal slew limit."""
    values = (start_current, ramp_speed_min, elapsed_s, applied_current)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Voltage-ramp inputs must be finite.")
    if ramp_speed_min <= 0 or elapsed_s < 0:
        raise ValueError("Voltage-ramp speed must be positive and elapsed time cannot be negative.")

    minimum_current = _measurement_current_floor(config)
    maximum_current = float(config["max_current"])
    requested_current = float(start_current) + float(ramp_speed_min) * float(elapsed_s) / 60.0
    requested_current = _clamp(requested_current, minimum_current, maximum_current)
    return _limit_current_slew(
        requested_current,
        float(applied_current),
        minimum_current,
        maximum_current,
        config,
    )


def get_controller_mode(config):
    mode = str(config.get("controller_mode", CONTROL_DEFAULTS["controller_mode"])).strip().upper()
    return mode if mode in {"PI", "PID"} else CONTROL_DEFAULTS["controller_mode"]


def pid_gains_for_current(config, current_a):
    """Return (kp, ki, kd) for the given operating current.

    Interpolates config["pid_gain_schedule"] (points from multi-point PI/PID
    tuning, run at different currents to cover a range where the process
    gain is not constant) when present, clamping at the ends rather than
    extrapolating past the tuned range. Falls back to the flat
    pid_kp/pid_ki/pid_kd values from a single-point tune when no schedule
    has been set.
    """
    kd_enabled = get_controller_mode(config) == "PID"
    schedule = config.get("pid_gain_schedule") or []
    if not schedule:
        kp = float(config.get("pid_kp", 0.0))
        ki = float(config.get("pid_ki", 0.0))
        kd = float(config.get("pid_kd", 0.0)) if kd_enabled else 0.0
        return kp, ki, kd

    points = sorted(schedule, key=lambda point: float(point["current_a"]))
    try:
        current_a = float(current_a)
    except (TypeError, ValueError):
        current_a = float(points[0]["current_a"])
    if not np.isfinite(current_a):
        current_a = float(points[0]["current_a"])

    if current_a <= float(points[0]["current_a"]):
        chosen = points[0]
        kp, ki, kd = float(chosen["kp"]), float(chosen["ki"]), float(chosen.get("kd", 0.0))
    elif current_a >= float(points[-1]["current_a"]):
        chosen = points[-1]
        kp, ki, kd = float(chosen["kp"]), float(chosen["ki"]), float(chosen.get("kd", 0.0))
    else:
        kp = ki = kd = None
        for lower, upper in zip(points, points[1:]):
            lower_current = float(lower["current_a"])
            upper_current = float(upper["current_a"])
            if lower_current <= current_a <= upper_current:
                span = upper_current - lower_current
                fraction = 0.0 if span <= 0 else (current_a - lower_current) / span
                kp = float(lower["kp"]) + fraction * (float(upper["kp"]) - float(lower["kp"]))
                ki = float(lower["ki"]) + fraction * (float(upper["ki"]) - float(lower["ki"]))
                kd = float(lower.get("kd", 0.0)) + fraction * (
                    float(upper.get("kd", 0.0)) - float(lower.get("kd", 0.0))
                )
                break
        if kp is None:
            chosen = points[-1]
            kp, ki, kd = float(chosen["kp"]), float(chosen["ki"]), float(chosen.get("kd", 0.0))

    return kp, ki, (kd if kd_enabled else 0.0)


def get_experiment_mode(config):
    raw_mode = config.get("experiment_mode", config.get("measurement_conversion_mode", "TEMPERATURE"))
    mode = str(raw_mode).strip().upper()
    if mode in {"CONTROLLED", "INTERPOLATE"}:
        return "TEMPERATURE"
    if mode in {"CURVE_SWEEP", "LINEAR_TEMP", "VOLTAGE"}:
        return "CURRENT"
    return mode if mode in {"TEMPERATURE", "CURRENT"} else CONTROL_DEFAULTS["experiment_mode"]


RESISTIVITY_MODES = ("V_OVER_I", "OFFSET_CORRECTED", "FOUR_WIRE")


def get_resistivity_mode(config):
    """Resolve how sample resistance is measured.

    V_OVER_I         continuous R = V / I while the heating current flows.
    OFFSET_CORRECTED duty-cycled; the reading taken with the output off is the
                     contact thermal EMF, which is subtracted from the live
                     sense voltage before dividing by the current.
    FOUR_WIRE        duty-cycled; the voltage DMM sources its own test current
                     and reports resistance directly while the output is off.
    """
    mode = str(config.get("resistivity_mode", CONTROL_DEFAULTS["resistivity_mode"])).strip().upper()
    return mode if mode in RESISTIVITY_MODES else CONTROL_DEFAULTS["resistivity_mode"]


def resistivity_mode_needs_power_supply(config):
    """Report whether the active mode has to switch the PSU output during a sample."""
    return get_resistivity_mode(config) != "V_OVER_I"


def resistivity_loop_time(config):
    """Seconds per control cycle.

    The duty-cycled modes set their own period from the heat and measure
    windows, because the sample is only heating for part of each cycle.
    """
    if not resistivity_mode_needs_power_supply(config):
        return 1.0 / float(config["experiment_frequency"])

    heat_time_s = float(config.get("resistivity_heat_time_s", CONTROL_DEFAULTS["resistivity_heat_time_s"]))
    measure_time_s = float(
        config.get("resistivity_measure_time_s", CONTROL_DEFAULTS["resistivity_measure_time_s"])
    )
    if not np.isfinite(heat_time_s) or heat_time_s <= 0:
        raise ValueError("resistivity_heat_time_s must be a positive finite number of seconds.")
    if not np.isfinite(measure_time_s) or measure_time_s <= 0:
        raise ValueError("resistivity_measure_time_s must be a positive finite number of seconds.")
    return heat_time_s + measure_time_s


# Keys from the constant-voltage era, mapped to their constant-current
# equivalents so an existing config.toml keeps loading.
LEGACY_CONFIG_KEYS = {
    "startup_voltage": "startup_current",
    "min_voltage": "min_current",
    "psu_keepalive_voltage": "psu_keepalive_current",
    "max_voltage_step_up": "max_current_step_up",
    "max_voltage_step_down": "max_current_step_down",
    "max_voltage_step_up_far": "max_current_step_up_far",
    "minimum_voltage_change": "minimum_current_change",
    "measurement_voltage_floor": "measurement_current_floor",
    "invalid_voltage_step_down": "invalid_current_step_down",
    "ignore_invalid_below_voltage": "ignore_invalid_below_current",
    "low_signal_recovery_voltage_step": "low_signal_recovery_current_step",
    "measurement_jump_probe_voltage_step": "measurement_jump_probe_current_step",
    "measurement_jump_confirm_min_voltage": "measurement_jump_confirm_min_current",
    "invalid_max_drop_from_recent_peak_v": "invalid_max_drop_from_recent_peak_a",
    "t0_calibration_voltage": "t0_calibration_current",
    "t0_voltage_search_start": "t0_current_search_start",
    "t0_voltage_step": "t0_current_step",
    "tuning_voltage_step": "tuning_current_step",
    "tuning_start_voltage": "tuning_start_current",
    "tuning_search_max_voltage": "tuning_search_max_current",
    "tuning_response_voltage_step": "tuning_response_current_step",
    # max_voltage was the actuator ceiling; as a supply setting it is now the
    # CV compliance limit, and the ceiling is max_current.
    "max_voltage": "compliance_voltage",
}


def migrate_legacy_config(config):
    """Move constant-voltage keys onto their constant-current names.

    The numbers themselves are not converted: volts and amps are not
    interchangeable, so a migrated file keeps whatever the operator set and the
    defaults fill in anything the old file never had.
    """
    migrated = dict(config)
    renamed = []
    for legacy_key, current_key in LEGACY_CONFIG_KEYS.items():
        if legacy_key in migrated:
            value = migrated.pop(legacy_key)
            if current_key not in migrated:
                migrated[current_key] = value
                renamed.append(f"{legacy_key} -> {current_key}")
    if renamed:
        print(
            "Migrated constant-voltage configuration keys to constant-current names: "
            + ", ".join(renamed)
            + ". Review the values: they were carried over unchanged, and volts do not "
            "convert to amps."
        )
    return migrated


def build_control_config(config):
    merged = migrate_legacy_config(config)
    for key, value in CONTROL_DEFAULTS.items():
        merged.setdefault(key, value)
    merged["controller_mode"] = get_controller_mode(merged)
    merged["experiment_mode"] = get_experiment_mode(merged)
    merged["resistivity_mode"] = get_resistivity_mode(merged)
    return merged


@dataclass
class ResistanceTemperatureModel:
    mode: str
    resistance_axis: np.ndarray
    interpolator: object = None
    linear_coefficients: Optional[Tuple[float, float]] = None
    temperature_bounds: Optional[Tuple[float, float]] = None
    source_temperature_bounds: Optional[Tuple[float, float]] = None

    @property
    def x(self):
        return self.resistance_axis

    def __call__(self, resistance):
        return self.interpolator(resistance)


def _temperature_sorted_curve(r_vs_t):
    curve = np.asarray(r_vs_t, dtype=float)
    if curve.shape[0] != 2 or curve.shape[1] < 2:
        raise ValueError("R vs. T data must have shape 2 x N with at least two points.")
    if not np.all(np.isfinite(curve)):
        raise ValueError("R vs. T data must contain only finite values.")

    temperature_order = np.argsort(curve[1, :])
    temperature_curve = curve[:, temperature_order]
    _, unique_indices = np.unique(temperature_curve[1, :], return_index=True)
    temperature_curve = temperature_curve[:, np.sort(unique_indices)]
    if temperature_curve.shape[1] < 2:
        raise ValueError("R vs. T data must contain at least two unique temperatures.")
    return temperature_curve


def _isotonic_resistance(values, direction):
    """Least-squares monotonic fit using the pool-adjacent-violators algorithm."""
    source = np.asarray(values, dtype=float)
    working = source if direction > 0 else -source
    block_values = []
    block_weights = []
    block_starts = []
    block_ends = []
    for index, value in enumerate(working):
        block_values.append(float(value))
        block_weights.append(1.0)
        block_starts.append(index)
        block_ends.append(index + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            combined_weight = block_weights[-2] + block_weights[-1]
            combined_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / combined_weight
            block_values[-2:] = [combined_value]
            block_weights[-2:] = [combined_weight]
            block_ends[-2:] = [block_ends[-1]]
            block_starts.pop()

    fitted = np.empty_like(working)
    for value, start, end in zip(block_values, block_starts, block_ends):
        fitted[start:end] = value
    return fitted if direction > 0 else -fitted


def _centered_rolling_median(values, window_points):
    values = np.asarray(values, dtype=float)
    window_points = max(int(window_points), 1)
    if window_points % 2 == 0:
        window_points += 1
    radius = window_points // 2
    return np.array(
        [
            np.median(values[max(0, index - radius):min(values.size, index + radius + 1)])
            for index in range(values.size)
        ],
        dtype=float,
    )


def _temperature_bin_medians(temperature_curve, bin_width_c):
    temperatures = np.asarray(temperature_curve[1, :], dtype=float)
    resistances = np.asarray(temperature_curve[0, :], dtype=float)
    origin = float(temperatures[0])
    bin_ids = np.floor((temperatures - origin) / bin_width_c + 1e-12).astype(np.int64)
    unique_bins = np.unique(bin_ids)
    binned_temperatures = np.array(
        [np.median(temperatures[bin_ids == bin_id]) for bin_id in unique_bins],
        dtype=float,
    )
    binned_resistances = np.array(
        [np.median(resistances[bin_ids == bin_id]) for bin_id in unique_bins],
        dtype=float,
    )
    return np.vstack((binned_resistances, binned_temperatures))


def _robust_monotonic_curve(temperature_curve, direction, config, correction_limit, tolerance):
    original_resistance = np.asarray(temperature_curve[0, :], dtype=float)
    resistance_span = float(np.ptp(original_resistance))
    monotonic_resistance = _isotonic_resistance(original_resistance, direction)
    maximum_correction = float(np.max(np.abs(monotonic_resistance - original_resistance)))
    if maximum_correction <= resistance_span * correction_limit + tolerance:
        corrected = temperature_curve.copy()
        corrected[0, :] = monotonic_resistance
        return corrected, maximum_correction, None

    smoothing_enabled = bool(config.get("curve_smoothing_enabled", True))
    minimum_points = int(config.get("curve_smoothing_min_points", 100))
    if not smoothing_enabled or temperature_curve.shape[1] < minimum_points:
        raise ValueError(
            "R vs. T conversion found resistance reversals larger than the configured monotonic "
            "correction limit. Clean the R vs. T file or enable robust curve smoothing."
        )

    bin_width_c = float(config.get("curve_smoothing_temperature_bin_c", 0.5))
    max_window_c = float(config.get("curve_smoothing_max_window_c", 15.0))
    max_residual_ratio = float(config.get("curve_smoothing_max_residual_ratio", 0.10))
    if not np.isfinite(bin_width_c) or bin_width_c <= 0:
        raise ValueError("curve_smoothing_temperature_bin_c must be positive and finite.")
    if not np.isfinite(max_window_c) or max_window_c < bin_width_c:
        raise ValueError("curve_smoothing_max_window_c must be finite and at least one bin wide.")
    if not np.isfinite(max_residual_ratio) or max_residual_ratio < 0:
        raise ValueError("curve_smoothing_max_residual_ratio must be non-negative and finite.")

    binned_curve = _temperature_bin_medians(temperature_curve, bin_width_c)
    if binned_curve.shape[1] < 3:
        raise ValueError("Robust curve smoothing produced fewer than three temperature bins.")

    max_window_points = max(3, int(np.ceil(max_window_c / bin_width_c)))
    if max_window_points % 2 == 0:
        max_window_points += 1
    max_window_points = min(max_window_points, binned_curve.shape[1])
    if max_window_points % 2 == 0:
        max_window_points -= 1

    selected_curve = None
    selected_correction = np.inf
    selected_window_points = None
    binned_span = float(np.ptp(binned_curve[0, :]))
    binned_tolerance = max(binned_span * 1e-12, 1e-15)
    for window_points in range(3, max_window_points + 1, 2):
        smoothed_resistance = _centered_rolling_median(binned_curve[0, :], window_points)
        monotonic_smoothed = _isotonic_resistance(smoothed_resistance, direction)
        correction = float(np.max(np.abs(monotonic_smoothed - smoothed_resistance)))
        selected_correction = correction
        if correction <= binned_span * correction_limit + binned_tolerance:
            selected_curve = binned_curve.copy()
            selected_curve[0, :] = monotonic_smoothed
            selected_window_points = window_points
            break

    if selected_curve is None:
        raise ValueError(
            "R vs. T conversion could not obtain a reliable monotonic trend after robust smoothing. "
            "Use a cleaner or wider-temperature R vs. T file."
        )

    fitted_at_raw_temperatures = np.interp(
        temperature_curve[1, :], selected_curve[1, :], selected_curve[0, :]
    )
    residual_99 = float(np.quantile(np.abs(original_resistance - fitted_at_raw_temperatures), 0.99))
    residual_ratio = residual_99 / max(resistance_span, tolerance)
    if residual_ratio > max_residual_ratio:
        raise ValueError(
            "R vs. T conversion found too many large deviations from the smoothed monotonic trend. "
            "Use a cleaner R vs. T file or increase curve_smoothing_max_residual_ratio only after review."
        )

    details = {
        "raw_points": int(temperature_curve.shape[1]),
        "binned_points": int(selected_curve.shape[1]),
        "window_c": float(selected_window_points * bin_width_c),
        "correction": selected_correction,
        "residual_99": residual_99,
        "residual_ratio": residual_ratio,
    }
    return selected_curve, selected_correction, details


def _condition_curve_for_inversion(temperature_curve, config):
    """Return a monotonic R(T) curve that can be inverted without branch ambiguity."""
    original_resistance = np.asarray(temperature_curve[0, :], dtype=float)
    overall_change = float(original_resistance[-1] - original_resistance[0])
    direction = float(np.sign(overall_change))
    resistance_span = float(np.ptp(original_resistance))
    tolerance = max(resistance_span * 1e-12, 1e-15)
    if direction == 0 or resistance_span <= tolerance:
        raise ValueError("R vs. T conversion requires resistance to change with temperature.")

    maximum_correction_ratio = float(
        config.get(
            "curve_monotonic_correction_ratio",
            config.get("curve_extrapolation_max_monotonic_correction_ratio", 0.02),
        )
    )
    if not np.isfinite(maximum_correction_ratio) or maximum_correction_ratio < 0:
        raise ValueError("curve_monotonic_correction_ratio must be non-negative.")

    conditioned_curve, maximum_correction, smoothing_details = _robust_monotonic_curve(
        temperature_curve,
        direction,
        config,
        maximum_correction_ratio,
        tolerance,
    )
    if maximum_correction > tolerance:
        if smoothing_details is None:
            print(
                "WARNING: corrected small non-monotonic resistance steps before R vs. T inversion "
                f"with a least-squares monotonic fit; maximum correction={maximum_correction:.6g} Ohm."
            )
        else:
            print(
                "WARNING: robustly smoothed a dense/noisy R vs. T curve before R vs. T inversion; "
                f"points={smoothing_details['raw_points']}->{smoothing_details['binned_points']}, "
                f"median window={smoothing_details['window_c']:.2f} C, "
                f"monotonic correction={smoothing_details['correction']:.6g} Ohm, "
                f"99th-percentile raw residual={smoothing_details['residual_99']:.6g} Ohm "
                f"({100.0 * smoothing_details['residual_ratio']:.2f}% of span). "
                "The source file was not modified."
            )
    return conditioned_curve, direction


def _fit_endpoint_resistance(
    temperature_curve,
    target_temperature,
    fit_points,
    side,
    expected_direction=None,
    minimum_temperature_span=0.0,
):
    requested_points = max(int(fit_points), 2)
    temperatures = temperature_curve[1, :]
    resistances = temperature_curve[0, :]
    full_temperature_span = float(np.ptp(temperatures))
    full_resistance_span = float(np.ptp(resistances))
    if full_temperature_span <= 0:
        raise ValueError("Cannot extrapolate an R vs. T curve with no temperature span.")

    required_temperature_span = min(
        max(float(minimum_temperature_span), 0.0),
        full_temperature_span,
    )
    if side == "lower":
        span_limit = temperatures[0] + required_temperature_span
        span_points = int(np.searchsorted(temperatures, span_limit, side="left")) + 1
        edge_resistance = float(resistances[0])
        target_side = -1.0
    elif side == "upper":
        span_limit = temperatures[-1] - required_temperature_span
        span_points = temperature_curve.shape[1] - int(
            np.searchsorted(temperatures, span_limit, side="left")
        )
        edge_resistance = float(resistances[-1])
        target_side = 1.0
    else:
        raise ValueError("Endpoint fit side must be 'lower' or 'upper'.")

    point_count = min(
        max(requested_points, span_points, 2),
        temperature_curve.shape[1],
    )
    representative_slope = full_resistance_span / full_temperature_span
    slope_tolerance = max(representative_slope * 1e-6, 1e-15)
    while True:
        endpoint = temperature_curve[:, :point_count] if side == "lower" else temperature_curve[:, -point_count:]
        slope, intercept = np.polyfit(endpoint[1, :], endpoint[0, :], 1)
        endpoint_resistance = float(slope * target_temperature + intercept)
        direction_is_valid = (
            expected_direction is None
            or (
                expected_direction * slope > slope_tolerance
                and expected_direction * target_side * (endpoint_resistance - edge_resistance) > 0
            )
        )
        if (
            np.isfinite(slope)
            and np.isfinite(intercept)
            and np.isfinite(endpoint_resistance)
            and abs(slope) >= slope_tolerance
            and direction_is_valid
        ):
            if point_count > requested_points:
                print(
                    f"WARNING: expanded the {side} endpoint fit from {requested_points} to {point_count} "
                    "points to cover a meaningful temperature span and reject flat/noisy endpoint behavior."
                )
            return endpoint_resistance, float(slope)
        if point_count >= temperature_curve.shape[1]:
            raise ValueError(
                f"Cannot extrapolate the {side} end of the R vs. T curve: "
                "no non-flat endpoint fit follows the curve's overall direction."
            )
        point_count = min(temperature_curve.shape[1], max(point_count + 1, point_count * 2))


def _extend_curve_for_configured_extrapolation(r_vs_t, config):
    temperature_curve = _temperature_sorted_curve(r_vs_t)
    if not bool(config.get("curve_extrapolation_enabled", False)):
        temperature_curve, _ = _condition_curve_for_inversion(temperature_curve, config)
        source_min = float(temperature_curve[1, 0])
        source_max = float(temperature_curve[1, -1])
        source_bounds = (source_min, source_max)
        return temperature_curve, source_bounds, source_bounds

    allowed_min = float(config["curve_extrapolation_min_temperature_c"])
    allowed_max = float(config["curve_extrapolation_max_temperature_c"])
    if not np.isfinite(allowed_min) or not np.isfinite(allowed_max) or allowed_min >= allowed_max:
        raise ValueError("Curve extrapolation temperature limits must be finite and increasing.")

    configured_rows = (temperature_curve[1, :] >= allowed_min) & (temperature_curve[1, :] <= allowed_max)
    temperature_curve = temperature_curve[:, configured_rows]
    if temperature_curve.shape[1] < 2:
        raise ValueError("R vs. T data must contain at least two rows inside the configured conversion range.")
    source_min = float(temperature_curve[1, 0])
    source_max = float(temperature_curve[1, -1])
    source_bounds = (source_min, source_max)

    temperature_curve, direction = _condition_curve_for_inversion(temperature_curve, config)
    source_min = float(temperature_curve[1, 0])
    source_max = float(temperature_curve[1, -1])
    source_bounds = (source_min, source_max)

    fit_points = int(config.get("curve_extrapolation_fit_points", 20))
    minimum_fit_span = float(config.get("curve_extrapolation_min_fit_span_c", 5.0))
    if not np.isfinite(minimum_fit_span) or minimum_fit_span < 0:
        raise ValueError("curve_extrapolation_min_fit_span_c must be finite and non-negative.")
    extended_points = [temperature_curve]
    if allowed_min < source_min:
        lower_resistance, lower_slope = _fit_endpoint_resistance(
            temperature_curve,
            allowed_min,
            fit_points,
            "lower",
            expected_direction=direction,
            minimum_temperature_span=minimum_fit_span,
        )
        if direction * lower_slope <= 0 or direction * (temperature_curve[0, 0] - lower_resistance) <= 0:
            raise ValueError("Lower curve extrapolation is not monotonic; use a better low-temperature curve.")
        extended_points.insert(0, np.array([[lower_resistance], [allowed_min]], dtype=float))
    if allowed_max > source_max:
        upper_resistance, upper_slope = _fit_endpoint_resistance(
            temperature_curve,
            allowed_max,
            fit_points,
            "upper",
            expected_direction=direction,
            minimum_temperature_span=minimum_fit_span,
        )
        if direction * upper_slope <= 0 or direction * (upper_resistance - temperature_curve[0, -1]) <= 0:
            raise ValueError("Upper curve extrapolation is not monotonic; use a better high-temperature curve.")
        extended_points.append(np.array([[upper_resistance], [allowed_max]], dtype=float))

    extended_curve = np.hstack(extended_points)
    print(
        "WARNING: R vs. T extrapolation is enabled. "
        f"Measured curve range={source_min:.2f}..{source_max:.2f} C; "
        f"allowed conversion range={allowed_min:.2f}..{allowed_max:.2f} C. "
        "Temperatures outside the measured curve are estimates."
    )
    return extended_curve, (allowed_min, allowed_max), source_bounds


def _build_temperature_interpolator_from_curve(
    curve,
    temperature_bounds,
    source_temperature_bounds,
):
    resistance_order = np.argsort(curve[0, :])
    resistance_curve = curve[:, resistance_order]
    _, unique_indices = np.unique(resistance_curve[0, :], return_index=True)
    resistance_curve = resistance_curve[:, np.sort(unique_indices)]
    return ResistanceTemperatureModel(
        mode="INTERPOLATE",
        resistance_axis=np.asarray(resistance_curve[0, :], dtype=float),
        temperature_bounds=temperature_bounds,
        source_temperature_bounds=source_temperature_bounds,
        interpolator=interp1d(
            resistance_curve[0, :],
            resistance_curve[1, :],
            kind="linear",
            fill_value="extrapolate",
        ),
    )


def build_temperature_interpolator(r_vs_t, config=None):
    config = build_control_config(config or {})
    curve, temperature_bounds, source_temperature_bounds = _extend_curve_for_configured_extrapolation(
        r_vs_t, config
    )
    return _build_temperature_interpolator_from_curve(
        curve,
        temperature_bounds,
        source_temperature_bounds,
    )


def _validate_temperature_program_bounds(experiment_params, temperature_interp):
    bounds = getattr(temperature_interp, "temperature_bounds", None)
    if bounds is None:
        return
    lower_bound, upper_bound = bounds
    for index, parameters in enumerate(experiment_params, start=1):
        for key in ("start_T", "target_T"):
            temperature = float(parameters[key])
            if temperature < lower_bound or temperature > upper_bound:
                raise ValueError(
                    f"Experiment step {index} {key}={temperature:.2f} C is outside the allowed "
                    f"R vs. T conversion range {lower_bound:.2f}..{upper_bound:.2f} C."
                )


@dataclass
class TemperatureProgram:
    start_T: float
    step_T: float
    target_T: float
    ramp_speed_min: float
    hold_step_time_min: float
    temperature_tolerance_c: float
    hold_entry_tolerance_c: float

    def __post_init__(self):
        if self.target_T < self.start_T:
            raise ValueError("target_T must be greater than or equal to start_T.")
        if self.ramp_speed_min <= 0:
            raise ValueError("ramp_speed_min must be greater than zero.")
        if self.hold_step_time_min < 0:
            raise ValueError("hold_step_time_min must be non-negative.")

        self.simple_ramp = self.step_T <= 0 or self.step_T >= (self.target_T - self.start_T)
        self.ramp_speed_c_s = self.ramp_speed_min / 60.0
        self.hold_step_time_s = self.hold_step_time_min * 60.0
        self.phase = "warmup"
        self.scheduled_target = self.start_T
        self.current_plateau = self.start_T
        self.hold_elapsed_s = 0.0

    def initialize(self, initial_temperature):
        self.scheduled_target = min(initial_temperature, self.start_T)
        self.current_plateau = self.start_T
        self.hold_elapsed_s = 0.0
        self.phase = "warmup"

    def _advance_target(self, target_limit, dt):
        self.scheduled_target = min(target_limit, self.scheduled_target + self.ramp_speed_c_s * dt)
        return self.scheduled_target

    def update(self, measured_temperature, dt):
        while True:
            if self.phase == "warmup":
                target = self._advance_target(self.start_T, dt)
                # The programmed target follows elapsed time. It must not stop
                # advancing merely because a noisy measurement trails start_T.
                if target >= self.start_T:
                    self.scheduled_target = self.start_T
                    if self.simple_ramp:
                        self.phase = "final_ramp"
                    else:
                        self.phase = "step_ramp"
                        self.current_plateau = min(self.start_T + self.step_T, self.target_T)
                    dt = 0.0
                    continue
                return target, self.phase, False

            if self.phase == "final_ramp":
                target = self._advance_target(self.target_T, dt)
                finished = (
                    target >= self.target_T
                    and measured_temperature >= self.target_T - self.temperature_tolerance_c
                )
                return target, self.phase, finished

            if self.phase == "step_ramp":
                target = self._advance_target(self.current_plateau, dt)
                plateau_reached = (
                    target >= self.current_plateau
                    and abs(measured_temperature - self.current_plateau) <= self.hold_entry_tolerance_c
                )
                if plateau_reached:
                    self.scheduled_target = self.current_plateau
                    if self.current_plateau >= self.target_T:
                        return self.current_plateau, self.phase, True
                    self.phase = "hold"
                    self.scheduled_target = self.current_plateau
                    self.hold_elapsed_s = 0.0
                    dt = 0.0
                    continue
                return target, self.phase, False

            if self.phase == "hold":
                self.hold_elapsed_s += dt
                finished = self.current_plateau >= self.target_T and self.hold_elapsed_s >= self.hold_step_time_s
                if finished:
                    return self.current_plateau, self.phase, True
                if self.hold_elapsed_s >= self.hold_step_time_s:
                    self.phase = "step_ramp"
                    self.current_plateau = min(self.current_plateau + self.step_T, self.target_T)
                    self.scheduled_target = min(self.scheduled_target, self.current_plateau)
                    self.hold_elapsed_s = 0.0
                    dt = 0.0
                    continue
                return self.current_plateau, self.phase, False

            raise RuntimeError(f"Unknown experiment phase: {self.phase}")


def _emit_measurement(
    emitter,
    target_temperature,
    temperature,
    measured_voltage,
    measured_current,
    pid_current,
    measured_resistance,
):
    measured_power = _sample_power_w(measured_voltage, measured_current)
    emitter.experiment_signal.emit(
        [
            time.time(),
            target_temperature,
            temperature,
            0,
            measured_voltage,
            measured_current,
            pid_current,
            measured_power,
            measured_resistance,
        ]
    )


def _persist_measurement(
    data_saver,
    target_temperature,
    temperature,
    measured_voltage,
    measured_current,
    pid_current,
    measured_resistance,
):
    if data_saver is None:
        return
    measured_power = _sample_power_w(measured_voltage, measured_current)
    data_saver.enqueue(
        [
            time.time(),
            target_temperature,
            temperature,
            0,
            measured_voltage,
            measured_current,
            pid_current,
            measured_power,
            measured_resistance,
        ]
    )


def _is_valid_measurement(measured_voltage, measured_current, temperature, config):
    if not all(_is_finite_scalar(value) for value in (measured_voltage, measured_current, temperature)):
        return False
    if abs(measured_current) < config["minimum_current_a"]:
        return False
    return True


def _temperature_rate_c_min(current_temperature, previous_temperature, dt):
    if previous_temperature is None or dt <= 0:
        return None
    return (current_temperature - previous_temperature) * 60.0 / dt


def _is_low_signal_state(applied_current, config):
    if not np.isfinite(applied_current):
        return False
    return applied_current <= float(
        config.get(
            "ignore_invalid_below_current",
            max(config.get("measurement_current_floor", 0.01) * 5.0, 0.05),
        )
    )


def _temperature_filter(history, temperature, window):
    if np.isfinite(temperature):
        history.append(float(temperature))
    max_samples = max(int(window), 1)
    if len(history) > max_samples:
        del history[:-max_samples]
    if not history:
        return np.nan
    return float(np.median(np.array(history, dtype=float)))


def _calculate_resistance(measured_voltage, measured_current, config=None):
    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current):
        return np.nan
    if abs(measured_current) < 1e-12:
        return np.nan
    resistance = measured_voltage / measured_current
    if config is not None:
        resistance -= float(config.get("fixed_series_resistance_ohm", 0.0))
    if not np.isfinite(resistance) or resistance <= 0:
        return np.nan
    return float(resistance)


def _resistance_jump_limit(previous_resistance, config):
    base_jump_limit = float(config.get("resistance_glitch_jump_ohm", 0.03))
    if previous_resistance is None or not np.isfinite(previous_resistance):
        return base_jump_limit
    return max(base_jump_limit, abs(float(previous_resistance)) * float(config.get("resistance_glitch_jump_ratio", 0.0)))


def _resistance_in_curve_bounds(resistance, temperature_interp, config):
    resistance_axis = getattr(temperature_interp, "x", None)
    if resistance_axis is None:
        return True

    resistance_axis = np.asarray(resistance_axis, dtype=float)
    if resistance_axis.size < 2 or not np.all(np.isfinite(resistance_axis)):
        return True

    lower_bound = float(np.min(resistance_axis))
    upper_bound = float(np.max(resistance_axis))
    margin = max(
        (upper_bound - lower_bound) * float(config.get("resistance_range_margin_ratio", 0.05)),
        float(config.get("resistance_range_margin_ohm", 0.0)),
    )
    return lower_bound - margin <= resistance <= upper_bound + margin


def _robust_resistance_inlier_mask(resistances, config):
    resistance_array = np.asarray(resistances, dtype=float)
    if resistance_array.size == 0 or not np.all(np.isfinite(resistance_array)):
        return np.zeros(resistance_array.shape, dtype=bool)

    median_resistance = float(np.median(resistance_array))
    mad = float(np.median(np.abs(resistance_array - median_resistance)))
    robust_sigma = 1.4826 * mad
    allowed_deviation = max(
        float(
            config.get(
                "resistance_outlier_min_ohm",
                config.get("startup_outlier_min_resistance_ohm", 0.0005),
            )
        ),
        float(
            config.get(
                "resistance_outlier_mad_multiplier",
                config.get("startup_outlier_mad_multiplier", 4.0),
            )
        )
        * robust_sigma,
    )
    return np.abs(resistance_array - median_resistance) <= allowed_deviation


def _advance_low_signal_current_recovery(
    recovery,
    invalid_reuse_streak,
    low_signal_state,
    applied_current,
    measured_current,
    config,
):
    """Probe upward in fixed steps when invalid low-voltage readings would otherwise deadlock control."""
    trigger_cycles = max(int(config.get("low_signal_recovery_trigger_cycles", 5)), 1)
    observe_cycles = max(int(config.get("low_signal_recovery_observe_cycles", 5)), 1)
    maximum_attempts = max(int(config.get("low_signal_recovery_max_attempts", 5)), 1)

    if not recovery.active:
        if not low_signal_state or invalid_reuse_streak < trigger_cycles:
            return None, False
        recovery.active = True
        # The trigger samples already provide the observation period for the first probe.
        recovery.invalid_samples_since_step = observe_cycles
    else:
        recovery.invalid_samples_since_step += 1

    if recovery.attempts >= maximum_attempts:
        return float(applied_current), False
    if recovery.invalid_samples_since_step < observe_cycles:
        return float(applied_current), False

    if not np.isfinite(measured_current) or abs(measured_current) >= 0.95 * float(config["max_current"]):
        return float(applied_current), False

    current_step = max(float(config.get("low_signal_recovery_current_step", 0.01)), 0.0)
    requested_current = _clamp(
        float(applied_current) + current_step,
        _measurement_current_floor(config),
        float(config["max_current"]),
    )
    minimum_change = max(float(config.get("minimum_current_change", 1e-4)), 0.0)
    if requested_current < float(applied_current) + minimum_change:
        return float(applied_current), False

    recovery.attempts += 1
    recovery.invalid_samples_since_step = 0
    return requested_current, True


def _start_control_at_initial_current(
    power_supply,
    config,
    previous_current,
    loop_time,
):
    initial_current = _measurement_current_floor(config)
    previous_current = _set_current_if_needed(power_supply, initial_current, previous_current, config)
    settle_time = max(float(config.get("startup_settle_time_s", 1.0)), 0.0)
    print(
        f"Starting controller directly at Initial Current {initial_current:.4f} A; "
        "this value remains the experiment current floor."
    )
    time.sleep(max(settle_time, loop_time))
    return initial_current, previous_current


def _measure_with_retry(
    dmm_v,
    dmm_i,
    siglent_module,
    temperature_interp,
    *,
    config,
    previous_resistance=None,
    power_supply=None,
):
    measured_voltage, measured_current, temperature, resistance = measure_resistivity(
        dmm_v,
        dmm_i,
        siglent_module,
        temperature_interp,
        config=config,
        power_supply=power_supply,
    )
    if not bool(config.get("measurement_temperature_jump_guard_enabled", True)):
        # Trust every reading directly instead of retrying/rejecting it against
        # the previous one; the current slew-rate limit is the only thing left
        # bounding how fast control can react to it.
        return measured_voltage, measured_current, temperature, resistance, np.isfinite(resistance)

    jump_limit = _resistance_jump_limit(previous_resistance, config)
    consensus_limit = max(
        float(config.get("measurement_retry_consensus_ohm", 0.015)),
        jump_limit * 0.5,
    )

    if (
        previous_resistance is None
        or not np.isfinite(previous_resistance)
        or not np.isfinite(resistance)
        or abs(resistance - previous_resistance) <= jump_limit
    ):
        return measured_voltage, measured_current, temperature, resistance, np.isfinite(resistance)

    print(
        f"Resistance jump detected: previous={previous_resistance:.4f} Ohm, "
        f"new={resistance:.4f} Ohm. Retrying measurement."
    )
    best = (measured_voltage, measured_current, temperature, resistance)
    best_distance = abs(resistance - previous_resistance)
    candidates = [best]

    for _ in range(int(config.get("measurement_retry_attempts", 2))):
        time.sleep(float(config.get("measurement_retry_delay_s", 0.15)))
        retry_voltage, retry_current, retry_temperature, retry_resistance = measure_resistivity(
            dmm_v,
            dmm_i,
            siglent_module,
            temperature_interp,
            config=config,
            power_supply=power_supply,
        )
        candidates.append((retry_voltage, retry_current, retry_temperature, retry_resistance))
        if np.isfinite(retry_resistance):
            retry_distance = abs(retry_resistance - previous_resistance)
            if retry_distance < best_distance:
                best = (retry_voltage, retry_current, retry_temperature, retry_resistance)
                best_distance = retry_distance
            if retry_distance <= jump_limit:
                return best[0], best[1], best[2], best[3], True

    if np.isfinite(best[3]) and best_distance <= jump_limit:
        return best[0], best[1], best[2], best[3], True

    valid_candidates = [candidate for candidate in candidates if np.isfinite(candidate[3])]
    if len(valid_candidates) >= 2:
        resistances = np.array([candidate[3] for candidate in valid_candidates], dtype=float)
        if float(np.max(resistances) - np.min(resistances)) <= consensus_limit:
            accepted_voltage = float(np.median(np.array([candidate[0] for candidate in valid_candidates], dtype=float)))
            accepted_current = float(np.median(np.array([candidate[1] for candidate in valid_candidates], dtype=float)))
            accepted_temperature_candidates = [
                candidate[2] for candidate in valid_candidates if np.isfinite(candidate[2])
            ]
            accepted_temperature = (
                float(np.median(np.array(accepted_temperature_candidates, dtype=float)))
                if accepted_temperature_candidates
                else np.nan
            )
            accepted_resistance = float(np.median(resistances))
            print(
                f"Accepting stable retried measurement at {accepted_resistance:.4f} Ohm "
                f"despite jump from previous {previous_resistance:.4f} Ohm."
            )
            return accepted_voltage, accepted_current, accepted_temperature, accepted_resistance, True

    print(
        f"Rejecting measurement after retries; best resistance {best[3]:.4f} Ohm "
        f"is still too far from previous {previous_resistance:.4f} Ohm."
    )
    return best[0], best[1], np.nan, best[3], False


def _set_current_if_needed(power_supply, current, previous_current, config):
    if previous_current is None or abs(current - previous_current) >= config["minimum_current_change"]:
        siglent.set_current(power_supply, current=current)
        return current
    return previous_current


def _curve_ordered_temperature_profile(r_vs_t):
    curve = np.asarray(r_vs_t, dtype=float)
    temperature_order = np.argsort(curve[1, :])
    ordered = curve[:, temperature_order]
    _, unique_indices = np.unique(ordered[1, :], return_index=True)
    ordered = ordered[:, np.sort(unique_indices)]
    return ordered[0, :], ordered[1, :]


def build_curve_shaped_current_schedule(r_vs_t, start_current, end_current, steps):
    if steps < 2:
        raise ValueError("Curve sweep requires at least two voltage points.")

    resistance_axis, temperature_axis = _curve_ordered_temperature_profile(r_vs_t)
    if temperature_axis.size < 2:
        raise ValueError("R vs. T data must contain at least two unique temperature points.")

    target_temperatures = np.linspace(float(temperature_axis[0]), float(temperature_axis[-1]), steps)
    target_resistances = np.interp(target_temperatures, temperature_axis, resistance_axis)

    resistance_span = float(np.max(target_resistances) - np.min(target_resistances))
    if resistance_span <= 1e-12:
        current_fractions = np.linspace(0.0, 1.0, steps)
    else:
        if target_resistances[-1] >= target_resistances[0]:
            current_fractions = (target_resistances - float(np.min(target_resistances))) / resistance_span
        else:
            current_fractions = (float(np.max(target_resistances)) - target_resistances) / resistance_span
        current_fractions = np.maximum.accumulate(np.clip(current_fractions, 0.0, 1.0))
        current_fractions[0] = 0.0
        current_fractions[-1] = 1.0

    voltages = start_current + current_fractions * (end_current - start_current)
    voltages = np.maximum.accumulate(np.asarray(voltages, dtype=float))
    voltages[-1] = end_current
    return voltages, target_temperatures


def _compute_next_current(
    pid_controller,
    temperature,
    setpoint,
    present_current,
    measured_current,
    target_temperature,
    temp_rate_c_min,
    ramp_speed_min,
    config,
    loop_time,
):
    control_min_current = _measurement_current_floor(config)
    if abs(measured_current) > config["max_current"]:
        raise ExperimentSafetyError(
            f"Measured current {measured_current:.4e} A exceeded max_current {config['max_current']:.4e} A."
        )

    if temperature > target_temperature + config["safety_temp_margin_c"]:
        raise ExperimentSafetyError(
            f"Measured temperature {temperature:.2f} C exceeded the safety limit near target {target_temperature:.2f} C."
        )

    pid_controller.kp, pid_controller.ki, pid_controller.kd = pid_gains_for_current(config, present_current)
    delta_current = pid_controller.compute(temperature, dt=loop_time, setpoint=setpoint)
    if not np.isfinite(delta_current):
        raise ExperimentSafetyError("PID requested a non-finite voltage change.")

    under_target_band = float(
        config.get("under_target_no_decrease_band_c", config.get("temperature_tolerance_c", 2.0))
    )
    rate_limit_band = float(
        config.get("rate_limit_activation_band_c", config.get("temperature_tolerance_c", under_target_band))
    )
    rate_limit_band = min(
        rate_limit_band,
        max(float(config.get("temperature_tolerance_c", 2.0)), under_target_band),
    )
    current_limited = abs(measured_current) >= 0.95 * config["max_current"]

    if temperature <= setpoint - under_target_band and delta_current < 0.0:
        delta_current = 0.0

    step_scale = current_step_scale(config)
    aggressive_step = (
        float(config.get("max_current_step_up_far", config["max_current_step_up"])) * step_scale
    )
    catchup_step = float(config["max_current_step_up"]) * step_scale
    far_below_setpoint = temperature <= setpoint - config.get("aggressive_step_band_c", 4.0)
    significantly_below_setpoint = temperature <= setpoint - rate_limit_band
    catchup_rate_c_min = max(ramp_speed_min * 0.6, ramp_speed_min - 3.0, 1.0)
    if far_below_setpoint and not current_limited:
        if temp_rate_c_min is None or not np.isfinite(temp_rate_c_min) or temp_rate_c_min < catchup_rate_c_min:
            delta_current = max(delta_current, aggressive_step)
        else:
            delta_current = max(delta_current, catchup_step)
    elif significantly_below_setpoint and not current_limited:
        delta_current = max(delta_current, catchup_step)

    if temperature >= setpoint + config["temperature_tolerance_c"]:
        delta_current = min(delta_current, 0.0)

    near_setpoint = temperature >= setpoint - rate_limit_band
    soft_rate_limit = max(
        ramp_speed_min + config["soft_temp_rate_margin_c_min"],
        config["soft_temp_rate_margin_c_min"],
    )
    hard_rate_limit = max(
        ramp_speed_min + config["hard_temp_rate_margin_c_min"],
        config["hard_temp_rate_margin_c_min"],
    )

    if temp_rate_c_min is not None and np.isfinite(temp_rate_c_min):
        if near_setpoint and temp_rate_c_min > soft_rate_limit and delta_current > 0.0:
            delta_current = 0.0
        if (
            near_setpoint
            and temp_rate_c_min > soft_rate_limit
            and temperature >= setpoint - config["temperature_tolerance_c"]
        ):
            delta_current = min(delta_current, -config["max_current_step_down"] / 2.0)
        if near_setpoint and temp_rate_c_min > hard_rate_limit:
            pid_controller.reset(measurement=temperature)
            delta_current = -config["max_current_step_down"]

    if current_limited and delta_current > 0.0:
        delta_current = 0.0

    if temperature >= target_temperature and setpoint >= target_temperature:
        delta_current = min(delta_current, 0.0)

    requested_current = _clamp(present_current + delta_current, control_min_current, config["max_current"])
    new_voltage = _limit_current_slew(
        requested_current,
        present_current,
        control_min_current,
        config["max_current"],
        config,
    )
    return new_voltage


def _confirmed_upward_temperature_jump(
    temperature,
    previous_temperature,
    measured_resistance,
    previous_resistance,
    measured_current,
    applied_current,
    resistance_confirmed,
    setpoint,
    config,
):
    if not resistance_confirmed:
        return False
    if not all(
        _is_finite_scalar(value)
        for value in (
            temperature,
            previous_temperature,
            measured_resistance,
            previous_resistance,
            measured_current,
            applied_current,
            setpoint,
        )
    ):
        return False
    if temperature <= previous_temperature or measured_resistance <= previous_resistance:
        return False

    minimum_confirm_current = max(
        config["minimum_current_a"] * 20.0,
        float(config.get("measurement_jump_confirm_min_current_a", 0.02)),
    )
    # Anything reaching this check has already cleared _is_low_signal_state's
    # own ignore_invalid_below_current gate (that regime has its own working
    # recovery path), so no extra multiple of it is needed here on top.
    minimum_confirm_applied_current = max(
        float(config.get("ignore_invalid_below_current", 0.05)),
        float(config.get("measurement_jump_confirm_min_current", 0.02)),
    )
    if abs(measured_current) < minimum_confirm_current or applied_current < minimum_confirm_applied_current:
        return False

    if temperature - previous_temperature > float(config.get("measurement_temp_jump_accept_up_c", 35.0)):
        return False

    if temperature > setpoint + float(config.get("measurement_temp_jump_accept_setpoint_margin_c", 15.0)):
        return False

    return True


def _confirmed_downward_temperature_jump(
    temperature,
    previous_temperature,
    measured_resistance,
    previous_resistance,
    measured_current,
    applied_current,
    resistance_confirmed,
    setpoint,
    config,
):
    if not all(
        _is_finite_scalar(value)
        for value in (
            temperature,
            previous_temperature,
            measured_resistance,
            previous_resistance,
            measured_current,
            applied_current,
            setpoint,
        )
    ):
        return False
    if temperature >= previous_temperature or measured_resistance >= previous_resistance:
        return False

    # Do not accept cooldown jumps when we are still far below the heating setpoint.
    # In this regime large downward jumps are almost always measurement artifacts.
    if previous_temperature < setpoint - float(config.get("temperature_tolerance_c", 2.0)):
        return False

    minimum_confirm_current = max(
        config["minimum_current_a"] * 20.0,
        float(config.get("measurement_jump_confirm_min_current_a", 0.02)),
    )
    # Anything reaching this check has already cleared _is_low_signal_state's
    # own ignore_invalid_below_current gate (that regime has its own working
    # recovery path), so no extra multiple of it is needed here on top.
    minimum_confirm_applied_current = max(
        float(config.get("ignore_invalid_below_current", 0.05)),
        float(config.get("measurement_jump_confirm_min_current", 0.02)),
    )
    return abs(measured_current) >= minimum_confirm_current and applied_current >= minimum_confirm_applied_current


def _screen_low_signal_temperature(
    temperature,
    resistance,
    trusted_temperature,
    confirmation,
    config,
):
    """Require repeated agreement before a low-signal jump replaces T0/trusted T."""
    if not np.isfinite(temperature) or not np.isfinite(trusted_temperature):
        return temperature, False, False

    temperature_delta = float(temperature) - float(trusted_temperature)
    upward_limit = float(
        config.get("measurement_temp_jump_up_c", config.get("measurement_temp_jump_c", 8.0) * 2.5)
    )
    downward_limit = float(
        config.get("measurement_temp_jump_down_c", config.get("measurement_temp_jump_c", 8.0))
    )
    if -downward_limit <= temperature_delta <= upward_limit:
        confirmation.reset()
        return float(temperature), False, False

    direction = "up" if temperature_delta > 0 else "down"
    required_confirmations = max(int(config.get("low_signal_jump_confirm_samples", 3)), 2)
    temperature_tolerance = max(
        float(config.get("low_signal_jump_temperature_tolerance_c", 10.0)),
        0.0,
    )
    resistance_tolerance = max(
        float(config.get("low_signal_jump_resistance_tolerance_ohm", 0.015)),
        0.0,
    )

    candidate_matches = (
        confirmation.active
        and confirmation.direction == direction
        and np.isfinite(resistance)
        and np.isfinite(confirmation.candidate_resistance)
        and abs(float(temperature) - confirmation.candidate_temperature) <= temperature_tolerance
        and abs(float(resistance) - confirmation.candidate_resistance) <= resistance_tolerance
    )
    if candidate_matches:
        previous_count = confirmation.confirmations
        confirmation.candidate_temperature = (
            confirmation.candidate_temperature * previous_count + float(temperature)
        ) / (previous_count + 1)
        confirmation.candidate_resistance = (
            confirmation.candidate_resistance * previous_count + float(resistance)
        ) / (previous_count + 1)
        confirmation.confirmations += 1
    else:
        confirmation.direction = direction
        confirmation.candidate_temperature = float(temperature)
        confirmation.candidate_resistance = float(resistance) if np.isfinite(resistance) else np.nan
        confirmation.confirmations = 1

    if confirmation.confirmations >= required_confirmations:
        confirmed_temperature = float(confirmation.candidate_temperature)
        confirmation.reset()
        return confirmed_temperature, False, True

    return np.nan, True, False


def _temperature_jump_probe_eligible(
    direction,
    temperature,
    previous_temperature,
    measured_resistance,
    previous_resistance,
    measured_current,
    applied_current,
    resistance_confirmed,
    config,
):
    if direction not in {"up", "down"} or not resistance_confirmed:
        return False
    if not all(
        _is_finite_scalar(value)
        for value in (
            temperature,
            previous_temperature,
            measured_resistance,
            previous_resistance,
            measured_current,
            applied_current,
        )
    ):
        return False

    if direction == "up":
        if temperature <= previous_temperature or measured_resistance <= previous_resistance:
            return False
    elif temperature >= previous_temperature or measured_resistance >= previous_resistance:
        return False

    minimum_confirm_current = max(
        config["minimum_current_a"] * 20.0,
        float(config.get("measurement_jump_confirm_min_current_a", 0.02)),
    )
    # Anything reaching this check has already cleared _is_low_signal_state's
    # own ignore_invalid_below_current gate (that regime has its own working
    # recovery path), so no extra multiple of it is needed here on top.
    minimum_confirm_applied_current = max(
        float(config.get("ignore_invalid_below_current", 0.05)),
        float(config.get("measurement_jump_confirm_min_current", 0.02)),
    )
    return abs(measured_current) >= minimum_confirm_current and applied_current >= minimum_confirm_applied_current


def _temperature_jump_probe_voltage(direction, applied_current, measured_current, config):
    step = max(
        float(config.get("measurement_jump_probe_current_step", 0.002)),
        float(config.get("minimum_current_change", 1e-4)),
    )
    lower_bound = max(
        float(config["min_current"]),
        float(config.get("measurement_current_floor", config["min_current"])),
    )
    if direction == "up":
        return _clamp(applied_current - step, lower_bound, float(config["max_current"]))

    if not np.isfinite(measured_current) or abs(measured_current) >= 0.95 * float(config["max_current"]):
        return float(applied_current)
    return _clamp(applied_current + step, lower_bound, float(config["max_current"]))


def _advance_temperature_jump_probe(
    probe,
    direction,
    temperature,
    resistance,
    applied_current,
    measured_current,
    config,
):
    required_confirmations = max(
        int(
            config.get(
                "measurement_heatup_confirm_samples" if direction == "up" else "measurement_cooldown_confirm_samples",
                2,
            )
        ),
        2,
    )
    maximum_attempts = max(
        int(config.get("measurement_jump_probe_max_samples", 20)),
        required_confirmations,
    )

    if not probe.active or probe.direction != direction:
        probe.direction = direction
        probe.candidate_temperature = float(temperature)
        probe.candidate_resistance = float(resistance)
        probe.origin_current = float(applied_current)
        probe.confirmations = 1
        probe.attempts = 1
    else:
        probe.attempts += 1
        temperature_tolerance = max(
            float(config.get("measurement_jump_probe_temperature_tolerance_c", 50.0)),
            0.0,
        )
        resistance_tolerance = max(
            float(config.get("measurement_retry_consensus_ohm", 0.015)),
            abs(probe.candidate_resistance)
            * float(config.get("measurement_jump_probe_resistance_ratio", 0.02)),
        )
        current_step = max(
            float(config.get("measurement_jump_probe_current_step", 0.002)),
            float(config.get("minimum_current_change", 1e-4)),
        )
        minimum_probe_change = max(
            float(config.get("minimum_current_change", 1e-4)),
            current_step * 0.25,
        )
        voltage_was_probed = (
            applied_current <= probe.origin_current - minimum_probe_change
            if direction == "up"
            else applied_current >= probe.origin_current + minimum_probe_change
        )
        candidate_is_consistent = (
            abs(temperature - probe.candidate_temperature) <= temperature_tolerance
            and abs(resistance - probe.candidate_resistance) <= resistance_tolerance
        )

        if candidate_is_consistent and voltage_was_probed:
            probe.confirmations += 1
            # Slide the reference to this sample. A real, still-settling trend
            # (long thermal tau, not yet plateaued) keeps each step small and
            # consistent even while the total drift since the first probe
            # sample grows past tolerance - anchoring to the latest accepted
            # sample instead of the first one lets that be confirmed quickly,
            # while a step that is itself too large or reverses direction
            # still fails candidate_is_consistent and resets the count below.
            probe.candidate_temperature = float(temperature)
            probe.candidate_resistance = float(resistance)
        else:
            # An inconsistent reading resets the count, but does not move the
            # reference: the next sample is judged against the same point,
            # not against the noisy one that just failed.
            probe.confirmations = 0

    if probe.confirmations >= required_confirmations:
        attempts = probe.attempts
        probe.reset()
        return True, None, attempts

    if probe.attempts >= maximum_attempts:
        # A signal that never settles within tolerance of any single reference
        # point (whether from noise or an unusually long transient) is not
        # distinguishable here from truly bad data, but killing the whole
        # experiment over one unconfirmed jump is disproportionate: the
        # general invalid-measurement handling already has its own, more
        # patient safety nets (invalid_reuse_stop_after, measurement_fail_limit)
        # that will stop the run if the reading never becomes trustworthy
        # again. Give up on this probe instead and let the next sample start a
        # fresh one - a real, still-moving trend gets caught again immediately
        # with an up-to-date reference.
        attempts = probe.attempts
        probe.reset()
        return False, None, attempts

    requested_current = _temperature_jump_probe_voltage(
        direction,
        float(applied_current),
        float(measured_current),
        config,
    )
    return False, requested_current, probe.attempts


def _psu_keepalive_current(config):
    try:
        keepalive_current = float(config.get("psu_keepalive_current", 0.001))
    except (TypeError, ValueError) as exc:
        raise ValueError("psu_keepalive_current must be a positive finite voltage.") from exc

    max_current = float(config["max_current"])
    if not np.isfinite(keepalive_current) or keepalive_current <= 0 or keepalive_current > max_current:
        raise ValueError(
            "psu_keepalive_current must be positive, finite, and no greater than max_current."
        )
    return keepalive_current


def prepare_power_supply_output(power_supply, config):
    """Keep CH1 enabled at a negligible voltage while a run is prepared."""
    keepalive_current = _psu_keepalive_current(config)
    compliance_voltage = float(config["compliance_voltage"])
    if not np.isfinite(compliance_voltage) or compliance_voltage <= 0:
        raise ValueError("compliance_voltage must be a positive finite voltage.")
    siglent.unlock_panel(power_supply)
    # Constant-current operation: the supply holds the set current and the
    # compliance voltage only bounds what an open circuit can be driven to.
    siglent.set_compliance_voltage(power_supply, voltage=compliance_voltage)
    time.sleep(0.05)
    siglent.set_current(power_supply, current=keepalive_current)
    time.sleep(0.05)
    siglent.set_output(power_supply, state="ON")
    print(
        f"Power supply output enabled in constant-current mode at keep-alive current "
        f"{keepalive_current:.6f} A with {compliance_voltage:.3f} V compliance."
    )


def _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager):
    if power_supply is not None:
        try:
            siglent.set_current(power_supply, current=0.0)
            time.sleep(0.1)
        except Exception as exc:
            print(f"An error occurred zeroing the PSU current before shutdown: {exc}")
        try:
            siglent.set_output(power_supply, state="OFF")
            print("Power supply output switched OFF.")
        except Exception as exc:
            print(f"An error occurred switching the power supply off: {exc}")
    for instrument in (dmm_v, dmm_i, power_supply):
        if instrument is not None:
            try:
                instrument.close()
            except Exception as exc:
                print(f"An error occurred while closing an instrument: {exc}")
    if resource_manager is not None:
        try:
            resource_manager.close()
        except Exception as exc:
            print(f"An error occurred while closing the VISA resource manager: {exc}")


def curve_sweep(emitter, sweep_params, r_vs_t, config, data_saver=None):
    if r_vs_t is None:
        raise ValueError("A resistivity-versus-temperature table must be loaded before starting a curve sweep.")

    config = build_control_config(config)
    temperature_interp = build_temperature_interpolator(r_vs_t, config=config)
    loop_time = resistivity_loop_time(config)
    max_current = min(float(config["max_current"]), float(sweep_params.get("max_current", config["max_current"])))
    start_current = max(
        float(config.get("curve_sweep_start_voltage", 0.01)),
        float(config.get("measurement_current_floor", 0.01)),
        0.01,
    )
    start_current = _clamp(start_current, float(config["min_current"]), max_current)
    current_step = max(float(config.get("curve_sweep_voltage_step", 0.005)), 1e-6)
    requested_steps = max(2, int(np.ceil(max_current / current_step)))

    resource_manager = None
    dmm_v = None
    dmm_i = None
    power_supply = None

    try:
        resource_manager = pyvisa.ResourceManager()
        dmm_v = resource_manager.open_resource(config["DMM_v"])
        dmm_i = resource_manager.open_resource(config["DMM_i"])
        power_supply = resource_manager.open_resource(config["PS"])
        power_supply.write_termination = "\n"
        power_supply.read_termination = "\n"

        prepare_power_supply_output(power_supply, config)
        siglent.configure_dc_range_from_config(dmm_v, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm_i, "CURR", config)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        time.sleep(1.0)

        schedule_voltages, schedule_temperatures = build_curve_shaped_current_schedule(
            r_vs_t,
            start_current=start_current,
            end_current=max_current,
            steps=requested_steps,
        )
        print(
            f"Curve sweep: start={start_current:.4f} A, end={max_current:.4f} A, "
            f"steps={requested_steps}, step_basis={current_step:.4f} A"
        )

        previous_current = None
        previous_resistance = None
        for target_temperature, target_current in zip(schedule_temperatures, schedule_voltages):
            if emitter.stopped:
                print("Stop signal received.")
                break

            loop_started = time.time()
            previous_current = _set_current_if_needed(power_supply, float(target_current), previous_current, config)
            time.sleep(max(loop_time, 0.2))

            measured_voltage, measured_current, temperature, measured_resistance, _ = _measure_with_retry(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                config=config,
                previous_resistance=previous_resistance,
                power_supply=power_supply,
            )
            if abs(measured_current) > config["max_current"]:
                raise ExperimentSafetyError(
                    f"Measured current {measured_current:.4e} A exceeded max_current {config['max_current']:.4e} A."
                )
            if not np.isfinite(temperature):
                raise ExperimentSafetyError(
                    "Curve sweep produced a temperature outside the configured R vs. T conversion range."
                )

            if np.isfinite(measured_resistance):
                previous_resistance = measured_resistance

            print(
                f"Curve sweep, T: {temperature:.2f} C, Target curve T: {target_temperature:.2f} C, "
                f"Vsample: {measured_voltage:.6f} V, Current: {measured_current:.4e} A, "
                f"PSU command: {previous_current:.4f} A"
            )
            _persist_measurement(
                data_saver,
                float(target_temperature),
                temperature,
                measured_voltage,
                measured_current,
                float(previous_current),
                measured_resistance,
            )
            _emit_measurement(
                emitter,
                float(target_temperature),
                temperature,
                measured_voltage,
                measured_current,
                float(previous_current),
                measured_resistance,
            )

            elapsed = time.time() - loop_started
            if elapsed < loop_time:
                time.sleep(loop_time - elapsed)

    finally:
        _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)
        if data_saver is not None:
            data_saver.finalize()
        print("Curve sweep thread finished.")


def current_ramp(emitter, ramp_params, r_vs_t, config, data_saver=None):
    """Ramp the PSU command by elapsed time while retaining current, power, and voltage safety limits."""
    if r_vs_t is None:
        raise ValueError("An R-vs-T table must be loaded before starting a voltage ramp.")

    config = build_control_config(config)
    temperature_interp = build_temperature_interpolator(r_vs_t, config=config)
    loop_time = resistivity_loop_time(config)
    ramp_speed_a_min = float(ramp_params["ramp_speed_min"])
    if not np.isfinite(ramp_speed_a_min) or ramp_speed_a_min <= 0:
        raise ValueError("Voltage-mode ramp_speed_min must be positive and finite.")

    measurement_current_floor = _measurement_current_floor(config)
    maximum_current = float(config["max_current"])
    max_power_w = float(config["max_power_w"])
    if not np.isfinite(max_power_w) or max_power_w <= 0:
        raise ValueError("max_power_w must be positive and finite.")

    resource_manager = None
    dmm_v = None
    dmm_i = None
    power_supply = None

    try:
        resource_manager = pyvisa.ResourceManager()
        dmm_v = resource_manager.open_resource(config["DMM_v"])
        dmm_i = resource_manager.open_resource(config["DMM_i"])
        power_supply = resource_manager.open_resource(config["PS"])
        power_supply.write_termination = "\n"
        power_supply.read_termination = "\n"

        prepare_power_supply_output(power_supply, config)
        siglent.configure_dc_range_from_config(dmm_v, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm_i, "CURR", config)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        time.sleep(1.0)

        commanded_current, previous_current = _start_control_at_initial_current(
            power_supply,
            config,
            previous_current=None,
            loop_time=loop_time,
        )
        ramp_started = time.monotonic()
        previous_resistance = None
        print(
            f"Current mode: start={commanded_current:.6f} A, "
            f"ramp_speed={ramp_speed_a_min:.6f} A/min, max_power={max_power_w:.6f} W, "
            f"absolute_current_ceiling={maximum_current:.6f} A."
        )

        while not emitter.stopped:
            loop_started = time.monotonic()
            applied_current = float(commanded_current)
            measured_voltage, measured_current, temperature, measured_resistance, _ = _measure_with_retry(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                config=config,
                previous_resistance=previous_resistance,
                power_supply=power_supply,
            )
            _enforce_electrical_safety(measured_voltage, measured_current, config)
            measured_power_w = _sample_power_w(measured_voltage, measured_current)
            if np.isfinite(measured_resistance):
                previous_resistance = measured_resistance

            print(
                f"Current mode: T={temperature if np.isfinite(temperature) else float('nan'):.2f} C, "
                f"Vsample={measured_voltage:.6f} V, Current={measured_current:.6e} A, "
                f"Power={measured_power_w:.6f} W, PSU command={applied_current:.6f} A."
            )
            _persist_measurement(
                data_saver,
                np.nan,
                temperature,
                measured_voltage,
                measured_current,
                applied_current,
                measured_resistance,
            )
            _emit_measurement(
                emitter,
                np.nan,
                temperature,
                measured_voltage,
                measured_current,
                applied_current,
                measured_resistance,
            )

            elapsed_ramp_s = time.monotonic() - ramp_started
            commanded_current = _current_ramp_command(
                measurement_current_floor,
                ramp_speed_a_min,
                elapsed_ramp_s,
                applied_current,
                config,
            )
            if (
                applied_current >= maximum_current - float(config.get("minimum_current_change", 1e-4))
                and commanded_current <= applied_current + 1e-12
            ):
                raise ExperimentSafetyError(
                    f"Current mode reached the absolute software current ceiling {maximum_current:.6f} A "
                    f"before reaching max_power_w {max_power_w:.6f} W."
                )
            previous_current = _set_current_if_needed(
                power_supply,
                commanded_current,
                previous_current,
                config,
            )

            elapsed_loop_s = time.monotonic() - loop_started
            if elapsed_loop_s < loop_time:
                time.sleep(loop_time - elapsed_loop_s)

        if emitter.stopped:
            print("Stop signal received.")
    finally:
        _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)
        if data_saver is not None:
            data_saver.finalize()
        print("Voltage-ramp thread finished.")


def tds(emitter, experiment_params, r_vs_t, config, t_zero, data_saver=None):
    if r_vs_t is None:
        raise ValueError("A resistivity-versus-temperature table must be loaded before starting an experiment.")

    config = build_control_config(config)
    temperature_interp = build_temperature_interpolator(r_vs_t, config=config)
    _validate_temperature_program_bounds(experiment_params, temperature_interp)
    loop_time = resistivity_loop_time(config)

    resource_manager = None
    dmm_v = None
    dmm_i = None
    power_supply = None

    try:
        resource_manager = pyvisa.ResourceManager()
        dmm_v = resource_manager.open_resource(config["DMM_v"])
        dmm_i = resource_manager.open_resource(config["DMM_i"])
        power_supply = resource_manager.open_resource(config["PS"])
        power_supply.write_termination = "\n"
        power_supply.read_termination = "\n"

        prepare_power_supply_output(power_supply, config)
        siglent.configure_dc_range_from_config(dmm_v, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm_i, "CURR", config)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        time.sleep(1.0)

        previous_current = None
        for ex_param in experiment_params:
            print("Experiment parameters:", ex_param)
            program = TemperatureProgram(
                start_T=ex_param["start_T"],
                step_T=ex_param["step_T"],
                target_T=ex_param["target_T"],
                ramp_speed_min=ex_param["ramp_speed_min"],
                hold_step_time_min=ex_param["hold_step_time_min"],
                temperature_tolerance_c=config["temperature_tolerance_c"],
                hold_entry_tolerance_c=config["hold_entry_tolerance_c"],
            )

            controller_mode = get_controller_mode(config)
            pid_controller = pid.PIDController(
                kp=config["pid_kp"],
                ki=config["pid_ki"],
                kd=config["pid_kd"] if controller_mode == "PID" else 0.0,
                setpoint=t_zero,
                output_limits=(
                    -config["max_current_step_down"] * current_step_scale(config),
                    config["max_current_step_up"] * current_step_scale(config),
                ),
                integral_limits=(-config["pid_integral_limit"], config["pid_integral_limit"]),
                derivative_filter=config["pid_derivative_filter"],
            )

            measurement_current_floor = _measurement_current_floor(config)
            pid_current, previous_current = _start_control_at_initial_current(
                power_supply=power_supply,
                config=config,
                previous_current=previous_current,
                loop_time=loop_time,
            )
            print(
                f"Using calibrated T0 {float(t_zero):.2f} C as the initial trusted temperature; "
                f"the first live reading will control the next current from {pid_current:.4f} A."
            )

            program.initialize(float(t_zero))
            pid_controller.reset(measurement=float(t_zero))
            invalid_measurements = 0
            invalid_reuse_streak = 0
            invalid_recovery_peak_voltage = None
            temperature_history = [float(t_zero)]
            filtered_temperature = _temperature_filter(
                temperature_history,
                float(t_zero),
                config.get("measurement_filter_samples", 3),
            )
            previous_temperature = filtered_temperature
            previous_resistance = None
            previous_phase = None
            pending_cooldown_jump_count = 0
            pending_heatup_jump_count = 0
            temperature_jump_probe = TemperatureJumpProbe()
            low_signal_confirmation = LowSignalTemperatureConfirmation()
            low_signal_voltage_recovery = LowSignalCurrentRecovery()
            last_program_update_time = time.monotonic()

            while not emitter.stopped:
                loop_started = time.time()
                program_update_time = time.monotonic()
                program_dt = max(program_update_time - last_program_update_time, 0.0)
                last_program_update_time = program_update_time
                applied_current = pid_current
                measurement_resistance_reference = (
                    temperature_jump_probe.candidate_resistance
                    if temperature_jump_probe.active
                    and np.isfinite(temperature_jump_probe.candidate_resistance)
                    else previous_resistance
                )
                measured_voltage, measured_current, temperature, measured_resistance, resistance_confirmed = _measure_with_retry(
                    dmm_v,
                    dmm_i,
                    siglent,
                    temperature_interp,
                    config=config,
                    previous_resistance=measurement_resistance_reference,
                    power_supply=power_supply,
                )
                raw_temperature = temperature
                low_signal_state = _is_low_signal_state(applied_current, config)
                jump_guard_enabled = bool(
                    config.get("measurement_temperature_jump_guard_enabled", True)
                )
                low_signal_jump_pending = False
                low_signal_jump_confirmed = False
                reset_temperature_reference = False
                jump_probe_current_request = None

                if jump_guard_enabled:
                    if low_signal_state:
                        temperature, low_signal_jump_pending, low_signal_jump_confirmed = (
                            _screen_low_signal_temperature(
                                temperature,
                                measured_resistance,
                                previous_temperature,
                                low_signal_confirmation,
                                config,
                            )
                        )
                        if low_signal_jump_pending:
                            print(
                                "Large low-signal temperature jump is not yet trusted: "
                                f"candidate={raw_temperature:.2f} C, trusted={previous_temperature:.2f} C. "
                                f"Confirmation {low_signal_confirmation.confirmations}/"
                                f"{max(int(config.get('low_signal_jump_confirm_samples', 3)), 2)}; "
                                "waiting for confirmation while the target continues to ramp."
                            )
                        elif low_signal_jump_confirmed:
                            print(
                                f"Confirmed repeated low-signal temperature state at {temperature:.2f} C; "
                                "replacing the previous trusted temperature."
                            )
                            temperature_history[:] = [float(temperature)]
                    else:
                        low_signal_confirmation.reset()
                    confirmed_upward_jump = _confirmed_upward_temperature_jump(
                        temperature=temperature,
                        previous_temperature=previous_temperature,
                        measured_resistance=measured_resistance,
                        previous_resistance=previous_resistance,
                        measured_current=measured_current,
                        applied_current=applied_current,
                        resistance_confirmed=resistance_confirmed,
                        setpoint=float(program.scheduled_target),
                        config=config,
                    )
                    confirmed_downward_jump = _confirmed_downward_temperature_jump(
                        temperature=temperature,
                        previous_temperature=previous_temperature,
                        measured_resistance=measured_resistance,
                        previous_resistance=previous_resistance,
                        measured_current=measured_current,
                        applied_current=applied_current,
                        resistance_confirmed=resistance_confirmed,
                        setpoint=float(program.scheduled_target),
                        config=config,
                    )
                    reset_temperature_reference = low_signal_jump_confirmed
                    jump_probe_current_request = None

                    if (
                        np.isfinite(temperature)
                        and previous_temperature is not None
                        and np.isfinite(previous_temperature)
                        and not low_signal_state
                    ):
                        temperature_delta = temperature - previous_temperature
                        jump_up_limit = float(
                            config.get(
                                "measurement_temp_jump_up_c",
                                config.get("measurement_temp_jump_c", 8.0) * 2.5,
                            )
                        )
                        jump_down_limit = float(
                            config.get(
                                "measurement_temp_jump_down_c",
                                config.get("measurement_temp_jump_c", 8.0),
                            )
                        )
                        probe_threshold = max(
                            float(config.get("measurement_jump_probe_threshold_c", 35.0)),
                            jump_up_limit,
                            jump_down_limit,
                        )
                        if temperature_delta < -jump_down_limit:
                            large_jump = abs(temperature_delta) >= probe_threshold
                            # Once a probe is already running for this direction, its own
                            # corrective current step can legitimately push resistance back
                            # toward previous_resistance (that is the whole point of probing);
                            # re-running eligibility's frozen-reference check on every later
                            # sample would then disqualify a probe that is working exactly as
                            # intended. Eligibility only gates whether to START a new probe -
                            # _advance_temperature_jump_probe's own rolling consistency check
                            # is what should filter samples once one is already active.
                            probe_eligible = (
                                temperature_jump_probe.active and temperature_jump_probe.direction == "down"
                            ) or _temperature_jump_probe_eligible(
                                "down",
                                temperature,
                                previous_temperature,
                                measured_resistance,
                                previous_resistance,
                                measured_current,
                                applied_current,
                                resistance_confirmed,
                                config,
                            )
                            if large_jump:
                                pending_cooldown_jump_count = 0
                                pending_heatup_jump_count = 0
                                if probe_eligible:
                                    probe_confirmed, jump_probe_current_request, probe_attempt = (
                                        _advance_temperature_jump_probe(
                                            temperature_jump_probe,
                                            "down",
                                            temperature,
                                            measured_resistance,
                                            applied_current,
                                            measured_current,
                                            config,
                                        )
                                    )
                                    if probe_confirmed:
                                        print(
                                            f"Controlled downward-jump probe confirmed a stable new state after "
                                            f"{probe_attempt} samples: previous={previous_temperature:.2f} C, "
                                            f"new={temperature:.2f} C, R={measured_resistance:.4f} Ohm. "
                                            "Accepting it and resetting the temperature filter."
                                        )
                                        temperature_history[:] = [float(temperature)]
                                        reset_temperature_reference = True
                                    elif jump_probe_current_request is None:
                                        print(
                                            f"Downward-jump probe could not reach consensus after {probe_attempt} samples: "
                                            f"previous={previous_temperature:.2f} C, candidate={temperature:.2f} C, "
                                            f"R={measured_resistance:.4f} Ohm. Giving up on this probe and treating the "
                                            "reading as invalid instead of stopping the experiment."
                                        )
                                        temperature = np.nan
                                    else:
                                        probe_action = (
                                            "increasing"
                                            if jump_probe_current_request > applied_current + 1e-9
                                            else "holding"
                                        )
                                        print(
                                            f"Large downward temperature jump detected: previous={previous_temperature:.2f} C, "
                                            f"candidate={temperature:.2f} C, R={measured_resistance:.4f} Ohm. "
                                            f"Probe sample {probe_attempt}: {probe_action} PSU slightly from "
                                            f"{applied_current:.4f} to {jump_probe_current_request:.4f} A before deciding."
                                        )
                                        temperature = np.nan
                                else:
                                    temperature_jump_probe.reset()
                                    print(
                                        f"Large downward temperature jump detected: previous={previous_temperature:.2f} C, "
                                        f"candidate={temperature:.2f} C. Signal or resistance confirmation was insufficient; "
                                        "treating this reading as invalid."
                                    )
                                    temperature = np.nan
                            elif confirmed_downward_jump:
                                if temperature_jump_probe.active:
                                    temperature_jump_probe.reset()
                                pending_cooldown_jump_count += 1
                                pending_heatup_jump_count = 0
                                required_cooldown_confirms = max(
                                    int(config.get("measurement_cooldown_confirm_samples", 2)),
                                    1,
                                )
                                if pending_cooldown_jump_count >= required_cooldown_confirms:
                                    print(
                                        f"Confirmed downward temperature jump: previous={previous_temperature:.2f} C, "
                                        f"new={temperature:.2f} C. Accepting it and resetting the temperature filter."
                                    )
                                    temperature_history[:] = [float(temperature)]
                                    pending_cooldown_jump_count = 0
                                    reset_temperature_reference = True
                                else:
                                    print(
                                        f"Potential downward temperature jump detected: previous={previous_temperature:.2f} C, "
                                        f"new={temperature:.2f} C. Waiting for confirmation."
                                    )
                                    temperature = np.nan
                            else:
                                temperature_jump_probe.reset()
                                pending_cooldown_jump_count = 0
                                pending_heatup_jump_count = 0
                                print(
                                    f"Temperature jump detected: previous={previous_temperature:.2f} C, "
                                    f"new={temperature:.2f} C. Treating this reading as invalid."
                                )
                                temperature = np.nan
                        elif temperature_delta > jump_up_limit:
                            large_jump = abs(temperature_delta) >= probe_threshold
                            # See the matching comment in the downward branch: a probe already
                            # running for this direction must not be re-disqualified by
                            # eligibility's frozen-reference check reacting to the probe's own
                            # corrective current step.
                            probe_eligible = (
                                temperature_jump_probe.active and temperature_jump_probe.direction == "up"
                            ) or _temperature_jump_probe_eligible(
                                "up",
                                temperature,
                                previous_temperature,
                                measured_resistance,
                                previous_resistance,
                                measured_current,
                                applied_current,
                                resistance_confirmed,
                                config,
                            )
                            if large_jump:
                                pending_cooldown_jump_count = 0
                                pending_heatup_jump_count = 0
                                if probe_eligible:
                                    probe_confirmed, jump_probe_current_request, probe_attempt = (
                                        _advance_temperature_jump_probe(
                                            temperature_jump_probe,
                                            "up",
                                            temperature,
                                            measured_resistance,
                                            applied_current,
                                            measured_current,
                                            config,
                                        )
                                    )
                                    if probe_confirmed:
                                        print(
                                            f"Controlled upward-jump probe confirmed a stable new state after "
                                            f"{probe_attempt} samples: previous={previous_temperature:.2f} C, "
                                            f"new={temperature:.2f} C, R={measured_resistance:.4f} Ohm. "
                                            "Accepting it and resetting the temperature filter."
                                        )
                                        temperature_history[:] = [float(temperature)]
                                        reset_temperature_reference = True
                                    elif jump_probe_current_request is None:
                                        print(
                                            f"Upward-jump probe could not reach consensus after {probe_attempt} samples: "
                                            f"previous={previous_temperature:.2f} C, candidate={temperature:.2f} C, "
                                            f"R={measured_resistance:.4f} Ohm. Giving up on this probe and treating the "
                                            "reading as invalid instead of stopping the experiment."
                                        )
                                        temperature = np.nan
                                    else:
                                        probe_action = (
                                            "decreasing"
                                            if jump_probe_current_request < applied_current - 1e-9
                                            else "holding"
                                        )
                                        print(
                                            f"Large upward temperature jump detected: previous={previous_temperature:.2f} C, "
                                            f"candidate={temperature:.2f} C, R={measured_resistance:.4f} Ohm. "
                                            f"Probe sample {probe_attempt}: {probe_action} PSU slightly from "
                                            f"{applied_current:.4f} to {jump_probe_current_request:.4f} A before deciding."
                                        )
                                        temperature = np.nan
                                else:
                                    temperature_jump_probe.reset()
                                    print(
                                        f"Large upward temperature jump detected: previous={previous_temperature:.2f} C, "
                                        f"candidate={temperature:.2f} C. Signal or resistance confirmation was insufficient; "
                                        "treating this reading as invalid."
                                    )
                                    temperature = np.nan
                            elif confirmed_upward_jump:
                                if temperature_jump_probe.active:
                                    temperature_jump_probe.reset()
                                pending_heatup_jump_count += 1
                                pending_cooldown_jump_count = 0
                                required_heatup_confirms = max(
                                    int(config.get("measurement_heatup_confirm_samples", 2)),
                                    1,
                                )
                                if pending_heatup_jump_count >= required_heatup_confirms:
                                    print(
                                        f"Confirmed upward temperature jump: previous={previous_temperature:.2f} C, "
                                        f"new={temperature:.2f} C. Accepting it and resetting the temperature filter."
                                    )
                                    temperature_history[:] = [float(temperature)]
                                    pending_heatup_jump_count = 0
                                    reset_temperature_reference = True
                                else:
                                    print(
                                        f"Potential upward temperature jump detected: previous={previous_temperature:.2f} C, "
                                        f"new={temperature:.2f} C. Waiting for confirmation."
                                    )
                                    temperature = np.nan
                            else:
                                temperature_jump_probe.reset()
                                pending_cooldown_jump_count = 0
                                pending_heatup_jump_count = 0
                                print(
                                    f"Temperature jump detected: previous={previous_temperature:.2f} C, "
                                    f"new={temperature:.2f} C. Treating this reading as invalid."
                                )
                                temperature = np.nan
                        else:
                            if temperature_jump_probe.active:
                                print("Temperature-jump probe cancelled because the measurement returned to the trusted range.")
                                temperature_jump_probe.reset()
                            pending_cooldown_jump_count = 0
                            pending_heatup_jump_count = 0
                    else:
                        pending_cooldown_jump_count = 0
                        pending_heatup_jump_count = 0
                        if temperature_jump_probe.active:
                            temperature_jump_probe.attempts += 1
                            maximum_probe_attempts = max(
                                int(config.get("measurement_jump_probe_max_samples", 20)),
                                2,
                            )
                            if temperature_jump_probe.attempts >= maximum_probe_attempts:
                                print(
                                    "Temperature-jump probe could not obtain stable readings after "
                                    f"{temperature_jump_probe.attempts} samples. Giving up on this probe and treating "
                                    "the reading as invalid instead of stopping the experiment."
                                )
                                temperature_jump_probe.reset()
                                jump_probe_current_request = None
                            else:
                                jump_probe_current_request = _temperature_jump_probe_voltage(
                                    temperature_jump_probe.direction,
                                    applied_current,
                                    measured_current,
                                    config,
                                )
                                print(
                                    "Temperature-jump probe received an unusable measurement; repeating the small "
                                    f"current probe at {jump_probe_current_request:.4f} A."
                                )
                    # The temperature-jump/low-signal confirmation machinery is disabled:
                    # trust each resistance-derived reading directly (still subject to the
                    # basic finiteness checks and the resistance-glitch retry in
                    # _measure_with_retry) and rely solely on the current slew-rate limit
                    # (max_current_step_up/down) to bound how fast control can react to it.
                else:
                    low_signal_confirmation.reset()
                    temperature_jump_probe.reset()
                    pending_cooldown_jump_count = 0
                    pending_heatup_jump_count = 0
                if not _is_valid_measurement(measured_voltage, measured_current, temperature, config):
                    target_reference_temperature = (
                        float(previous_temperature)
                        if previous_temperature is not None and np.isfinite(previous_temperature)
                        else float(t_zero)
                    )
                    setpoint, phase, finished = program.update(target_reference_temperature, program_dt)
                    can_reuse_last_temperature = (
                        previous_temperature is not None
                        and np.isfinite(previous_temperature)
                        and np.isfinite(measured_voltage)
                        and np.isfinite(measured_current)
                        and abs(measured_current) <= config["max_current"]
                    )
                    if can_reuse_last_temperature:
                        invalid_reuse_streak += 1
                        invalid_reuse_stop_after = max(int(config.get("invalid_reuse_stop_after", 30)), 1)
                        if invalid_reuse_streak >= invalid_reuse_stop_after:
                            raise ExperimentSafetyError(
                                "Persistent invalid measurement loop detected while reusing the last trusted "
                                f"temperature {previous_temperature:.2f} C for {invalid_reuse_streak} cycles. "
                                "Stopping to avoid blind control on corrupted data."
                            )
                        if invalid_recovery_peak_voltage is None or not np.isfinite(invalid_recovery_peak_voltage):
                            invalid_recovery_peak_voltage = float(applied_current)
                        else:
                            invalid_recovery_peak_voltage = max(float(invalid_recovery_peak_voltage), float(applied_current))
                        recovery_temperature = previous_temperature

                        if phase != previous_phase:
                            pid_controller.reset(measurement=recovery_temperature)
                            previous_phase = phase
                        else:
                            pid_controller.reset(measurement=recovery_temperature)

                        pid_current = _compute_next_current(
                            pid_controller=pid_controller,
                            temperature=recovery_temperature,
                            setpoint=setpoint,
                            present_current=pid_current,
                            measured_current=measured_current,
                            target_temperature=program.target_T,
                            temp_rate_c_min=0.0,
                            ramp_speed_min=program.ramp_speed_min,
                            config=config,
                            loop_time=loop_time,
                        )
                        if low_signal_jump_pending:
                            pid_current = min(pid_current, applied_current)
                        if invalid_reuse_streak >= max(int(config.get("invalid_reuse_hold_after", 8)), 1):
                            # Prevent runaway voltage escalation when we are reusing stale temperature for too long.
                            pid_current = min(
                                pid_current,
                                applied_current - 0.5 * config.get("invalid_current_step_down", config["max_current_step_up"]),
                            )
                        recovery_under_target_band = float(
                            config.get("under_target_no_decrease_band_c", config.get("temperature_tolerance_c", 2.0))
                        )
                        resistance_jump_limit = _resistance_jump_limit(previous_resistance, config)
                        invalid_hot_hint = (
                            not low_signal_state
                            and (
                                (
                                    np.isfinite(raw_temperature)
                                    and raw_temperature >= setpoint + config["temperature_tolerance_c"]
                                )
                                or (
                                    np.isfinite(measured_resistance)
                                    and np.isfinite(previous_resistance)
                                    and measured_resistance
                                    >= previous_resistance
                                    + max(
                                        resistance_jump_limit * 0.5,
                                        float(config.get("measurement_retry_consensus_ohm", 0.015)),
                                    )
                                )
                            )
                        )
                        recovery_current_limited = (
                            np.isfinite(measured_current)
                            and abs(measured_current) >= 0.95 * config["max_current"]
                        )
                        if invalid_hot_hint:
                            pid_current = min(
                                pid_current,
                                applied_current
                                - max(
                                    float(config.get("invalid_current_step_down", config["max_current_step_up"])),
                                    config["max_current_step_up"],
                                ),
                            )
                        elif (
                            not low_signal_state
                            and pid_current >= applied_current
                            and invalid_reuse_streak < max(int(config.get("invalid_reuse_hold_after", 8)), 1)
                            and (
                                recovery_current_limited
                                or recovery_temperature >= setpoint - recovery_under_target_band
                            )
                        ):
                            pid_current = applied_current - config["max_current_step_up"]
                        pid_current = _limit_current_slew(
                            pid_current,
                            applied_current,
                            measurement_current_floor,
                            config["max_current"],
                            config,
                        )
                        max_invalid_drop = max(float(config.get("invalid_max_drop_from_recent_peak_a", 0.1)), 0.0)
                        invalid_recovery_floor = max(
                            measurement_current_floor,
                            float(invalid_recovery_peak_voltage) - max_invalid_drop,
                        )
                        pid_current = max(pid_current, invalid_recovery_floor)
                        if jump_probe_current_request is not None:
                            pid_current = _limit_current_slew(
                                jump_probe_current_request,
                                applied_current,
                                measurement_current_floor,
                                config["max_current"],
                                config,
                            )
                        low_signal_recovery_voltage, low_signal_recovery_stepped = (
                            _advance_low_signal_current_recovery(
                                recovery=low_signal_voltage_recovery,
                                invalid_reuse_streak=invalid_reuse_streak,
                                low_signal_state=low_signal_state,
                                applied_current=applied_current,
                                measured_current=measured_current,
                                config=config,
                            )
                        )
                        # Once this recovery has spent its attempts it keeps returning
                        # applied_current unchanged forever (nothing resets it outside the
                        # fully-valid-measurement path, which can't be reached while we're
                        # stuck) - that "hold" is only meant to pause its own stepping, not
                        # to veto a concurrent, more specific mechanism. Applying it while a
                        # jump probe is actively driving current silently discards every
                        # probe request forever, deadlocking the experiment.
                        if jump_probe_current_request is None and low_signal_recovery_voltage is not None:
                            pid_current = low_signal_recovery_voltage
                        if low_signal_recovery_stepped:
                            print(
                                "Low-signal recovery probe "
                                f"{low_signal_voltage_recovery.attempts}/"
                                f"{max(int(config.get('low_signal_recovery_max_attempts', 5)), 1)}: "
                                f"increasing commanded PSU from {applied_current:.4f} to {pid_current:.4f} A "
                                "and observing the next measurements."
                            )
                        previous_current = _set_current_if_needed(power_supply, pid_current, previous_current, config)
                        invalid_measurements = 0
                        if (
                            resistance_confirmed
                            and np.isfinite(measured_resistance)
                            and not temperature_jump_probe.active
                        ):
                            previous_resistance = measured_resistance
                        if pid_current > applied_current + 1e-9:
                            recovery_action = "continuing upward"
                        elif pid_current < applied_current - 1e-9:
                            recovery_action = "gently backing off"
                        else:
                            recovery_action = "holding"
                        print(
                            f"Ignoring {'low-signal' if low_signal_state else 'transient'} invalid measurement. "
                            f"Measured Vsample={measured_voltage}, I={measured_current} while commanded PSU was {applied_current:.4f} A. "
                            f"Reusing last trusted temperature {recovery_temperature:.2f} C and "
                            f"{recovery_action} to {pid_current:.4f} A."
                        )
                        # recovery_temperature (the last trusted value) drives the PID
                        # while a jump is unconfirmed, but it must not be written to the
                        # dataset/plot next to a live, still-changing resistance reading
                        # as if it were an actual measurement of the current instant.
                        _persist_measurement(
                            data_saver,
                            setpoint,
                            np.nan,
                            measured_voltage,
                            measured_current,
                            applied_current,
                            measured_resistance,
                        )
                        _emit_measurement(
                            emitter,
                            setpoint,
                            np.nan,
                            measured_voltage,
                            measured_current,
                            applied_current,
                            measured_resistance,
                        )
                        if finished:
                            print("Experiment step finished.")
                            break
                        elapsed = time.time() - loop_started
                        if elapsed < loop_time:
                            time.sleep(loop_time - elapsed)
                        else:
                            print(f"Loop time exceeded: {elapsed:.3f} s")
                        continue

                    invalid_measurements += 1
                    invalid_reuse_streak = 0
                    if invalid_recovery_peak_voltage is None or not np.isfinite(invalid_recovery_peak_voltage):
                        invalid_recovery_peak_voltage = float(applied_current)
                    else:
                        invalid_recovery_peak_voltage = max(float(invalid_recovery_peak_voltage), float(applied_current))
                    pid_controller.reset(measurement=previous_temperature)
                    pid_current = _clamp(
                        applied_current - config.get("invalid_current_step_down", config["max_current_step_down"]),
                        measurement_current_floor,
                        config["max_current"],
                    )
                    pid_current = _limit_current_slew(
                        pid_current,
                        applied_current,
                        measurement_current_floor,
                        config["max_current"],
                        config,
                    )
                    max_invalid_drop = max(float(config.get("invalid_max_drop_from_recent_peak_a", 0.1)), 0.0)
                    invalid_recovery_floor = max(
                        measurement_current_floor,
                        float(invalid_recovery_peak_voltage) - max_invalid_drop,
                    )
                    pid_current = max(pid_current, invalid_recovery_floor)
                    previous_current = _set_current_if_needed(power_supply, pid_current, previous_current, config)
                    print(
                        "Invalid measurement received. "
                        f"Measured Vsample={measured_voltage}, I={measured_current} while commanded PSU was {applied_current:.4f} A. "
                        f"Reducing PSU to {pid_current:.4f} A (attempt {invalid_measurements})."
                    )
                    if invalid_measurements >= config["measurement_fail_limit"]:
                        raise ExperimentSafetyError("Too many invalid measurements in a row.")
                    _persist_measurement(
                        data_saver,
                        setpoint,
                        np.nan,
                        measured_voltage,
                        measured_current,
                        applied_current,
                        measured_resistance,
                    )
                    _emit_measurement(
                        emitter,
                        setpoint,
                        np.nan,
                        measured_voltage,
                        measured_current,
                        applied_current,
                        measured_resistance,
                    )
                    elapsed = time.time() - loop_started
                    if elapsed < loop_time:
                        time.sleep(loop_time - elapsed)
                    continue

                invalid_measurements = 0
                invalid_reuse_streak = 0
                invalid_recovery_peak_voltage = None
                low_signal_voltage_recovery.reset()
                filtered_temperature = _temperature_filter(
                    temperature_history,
                    temperature,
                    config.get("measurement_filter_samples", 3),
                )
                rate_reference_temperature = (
                    filtered_temperature if reset_temperature_reference else previous_temperature
                )
                temp_rate_c_min = _temperature_rate_c_min(filtered_temperature, rate_reference_temperature, loop_time)
                setpoint, phase, finished = program.update(filtered_temperature, program_dt)

                if phase != previous_phase:
                    pid_controller.reset(measurement=filtered_temperature)
                    previous_phase = phase

                pid_current = _compute_next_current(
                    pid_controller=pid_controller,
                    temperature=filtered_temperature,
                    setpoint=setpoint,
                    present_current=pid_current,
                    measured_current=measured_current,
                    target_temperature=program.target_T,
                    temp_rate_c_min=temp_rate_c_min,
                    ramp_speed_min=program.ramp_speed_min,
                    config=config,
                    loop_time=loop_time,
                )
                previous_current = _set_current_if_needed(power_supply, pid_current, previous_current, config)

                print(
                    f"Phase: {phase}, T: {filtered_temperature:.2f} C, Setpoint: {setpoint:.2f} C, "
                    f"Vsample: {measured_voltage:.6f} V, Current: {measured_current:.4e} A, "
                    f"PSU command: {applied_current:.4f} -> {pid_current:.4f} A, "
                    f"Rate: {temp_rate_c_min if temp_rate_c_min is not None else 0.0:.2f} C/min"
                )
                _persist_measurement(
                    data_saver,
                    setpoint,
                    filtered_temperature,
                    measured_voltage,
                    measured_current,
                    applied_current,
                    measured_resistance,
                )
                _emit_measurement(
                    emitter,
                    setpoint,
                    filtered_temperature,
                    measured_voltage,
                    measured_current,
                    applied_current,
                    measured_resistance,
                )
                previous_temperature = filtered_temperature
                if np.isfinite(measured_resistance):
                    previous_resistance = measured_resistance

                if finished:
                    print("Experiment step finished.")
                    break

                elapsed = time.time() - loop_started
                if elapsed < loop_time:
                    time.sleep(loop_time - elapsed)
                else:
                    print(f"Loop time exceeded: {elapsed:.3f} s")

            if emitter.stopped:
                print("Stop signal received.")
                break

    finally:
        _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)
        if data_saver is not None:
            data_saver.finalize()
        print("TDS experiment thread finished.")


def measure_resistivity(
    dmm_v,
    dmm_i,
    siglent_module,
    temperature_interp,
    calibration=False,
    config=None,
    power_supply=None,
):
    """Measure the sample and convert it to a temperature.

    Returns (measured_voltage, measured_current, temperature, resistance). The
    resistance is authoritative: in the duty-cycled modes it is not simply the
    returned voltage over the returned current.
    """
    overload_checker = (
        getattr(siglent_module, "is_overload_reading", None)
        if "is_overload_reading" in dir(siglent_module)
        else None
    )

    def parse_reading(raw_value):
        if callable(overload_checker) and overload_checker(raw_value):
            return np.nan, True
        try:
            return float(raw_value), False
        except (TypeError, ValueError):
            return np.nan, False

    def read_pair_once():
        synchronized_reader = getattr(siglent_module, "read_DMM_pair", None)
        use_synchronized_reading = config is None or bool(config.get("dmm_synchronized_reading", True))
        if use_synchronized_reading and synchronized_reader is not None:
            try:
                raw_voltage, raw_current = synchronized_reader(dmm_v, dmm_i)
                voltage, voltage_overload = parse_reading(raw_voltage)
                current, current_overload = parse_reading(raw_current)
            except Exception as exc:
                print(f"Synchronized DMM reading failed; retrying this sample with READ?: {exc}")
                voltage = np.nan
                current = np.nan
                voltage_overload = False
                current_overload = False
        else:
            voltage = np.nan
            current = np.nan
            voltage_overload = False
            current_overload = False

        if not np.isfinite(voltage) and not voltage_overload:
            try:
                voltage, voltage_overload = parse_reading(siglent_module.read_DMM(dmm_v))
                if not np.isfinite(voltage) and not voltage_overload:
                    print("Voltage DMM returned a non-numeric response.")
            except Exception as exc:
                print(f"An error occurred reading voltage DMM: {exc}")
                voltage = np.nan

        if not np.isfinite(current) and not current_overload:
            try:
                current, current_overload = parse_reading(siglent_module.read_DMM(dmm_i))
                if not np.isfinite(current) and not current_overload:
                    print("Current DMM returned a non-numeric response.")
            except Exception as exc:
                print(f"An error occurred reading current DMM: {exc}")
                current = np.nan
        return voltage, current, voltage_overload, current_overload

    def read_with_output_off(mode):
        """Open the heater circuit, take the quiet-window reading, restore the output.

        Returns (thermal_offset_v, four_wire_resistance_ohm); whichever the
        active mode does not produce comes back as NaN.
        """
        settle_s = float(config.get("resistivity_output_settle_s", 0.3))
        measure_s = float(config.get("resistivity_measure_time_s", 2.0))
        if not np.isfinite(settle_s) or settle_s < 0:
            raise ValueError("resistivity_output_settle_s must be finite and non-negative.")
        if not np.isfinite(measure_s) or measure_s <= 0:
            raise ValueError("resistivity_measure_time_s must be a positive finite number of seconds.")

        thermal_offset_v = np.nan
        four_wire_resistance = np.nan
        restore_volt_range = config.get("_active_dmm_volt_range", config.get("dmm_voltage_range_v"))

        siglent_module.set_output(power_supply, state="OFF")
        try:
            time.sleep(settle_s)
            if mode == "FOUR_WIRE":
                siglent_module.configure_fres_range(dmm_v, config["dmm_resistance_range_ohm"])
                time.sleep(max(measure_s - settle_s, 0.0))
                resistance_value, resistance_overload = parse_reading(
                    siglent_module.read_DMM_resistance(dmm_v)
                )
                if resistance_overload:
                    print(
                        "Four-wire resistance reading overloaded; increase "
                        "dmm_resistance_range_ohm for this sample."
                    )
                else:
                    four_wire_resistance = resistance_value
            else:
                time.sleep(max(measure_s - settle_s, 0.0))
                offset_voltage, _, offset_overload, _ = read_pair_once()
                if not offset_overload:
                    thermal_offset_v = offset_voltage
        finally:
            if mode == "FOUR_WIRE" and restore_volt_range is not None:
                siglent_module.configure_dc_range(dmm_v, "VOLT", restore_volt_range)
            siglent_module.set_output(power_supply, state="ON")
        return thermal_offset_v, four_wire_resistance

    resistivity_mode = get_resistivity_mode(config) if config is not None else "V_OVER_I"
    if resistivity_mode != "V_OVER_I":
        if power_supply is None:
            raise ValueError(
                f"Resistivity mode {resistivity_mode} has to switch the power-supply output "
                "off for every sample, so it needs the power-supply session."
            )
        # Heat at the commanded voltage first, so the live reading reflects a
        # settled sample; the quiet measurement window follows immediately.
        heat_time_s = float(config.get("resistivity_heat_time_s", 3.0))
        if not np.isfinite(heat_time_s) or heat_time_s < 0:
            raise ValueError("resistivity_heat_time_s must be finite and non-negative.")
        time.sleep(heat_time_s)

    measured_voltage, measured_current, voltage_overload, current_overload = read_pair_once()

    range_increaser = (
        getattr(siglent_module, "increase_dc_range_if_needed", None)
        if "increase_dc_range_if_needed" in dir(siglent_module)
        else None
    )
    if config is not None and callable(range_increaser):
        settle_time_s = float(config.get("dmm_range_settle_time_s", 0.3))
        discard_readings = int(config.get("dmm_range_discard_readings", 2))
        recovery_attempts = int(config.get("dmm_range_recovery_attempts", 5))
        if not np.isfinite(settle_time_s) or settle_time_s < 0:
            raise ValueError("dmm_range_settle_time_s must be finite and non-negative.")
        if discard_readings < 0:
            raise ValueError("dmm_range_discard_readings must be non-negative.")
        if recovery_attempts < 1:
            raise ValueError("dmm_range_recovery_attempts must be at least 1.")

        completed_range_changes = 0
        while completed_range_changes < recovery_attempts:
            voltage_range_change = range_increaser(
                dmm_v,
                "VOLT",
                measured_voltage,
                config,
                force_next=voltage_overload,
            )
            current_range_change = range_increaser(
                dmm_i,
                "CURR",
                measured_current,
                config,
                force_next=current_overload,
                step_margin=float(config.get("max_current_step_up", 0.01)) * current_step_scale(config),
            )
            if voltage_range_change is None and current_range_change is None:
                if voltage_overload or current_overload:
                    overloaded_meters = []
                    if voltage_overload:
                        overloaded_meters.append("voltage")
                    if current_overload:
                        overloaded_meters.append("current")
                    print(
                        f"DMM {' and '.join(overloaded_meters)} overload could not be recovered by "
                        "increasing the fixed range; treating this sample as invalid."
                    )
                break

            completed_range_changes += 1
            if settle_time_s:
                time.sleep(settle_time_s)
            for _ in range(discard_readings):
                read_pair_once()
            measured_voltage, measured_current, voltage_overload, current_overload = read_pair_once()

        if (voltage_overload or current_overload) and completed_range_changes >= recovery_attempts:
            print(
                f"DMM overload remained after {recovery_attempts} fixed-range recovery attempts; "
                "treating this sample as invalid."
            )

    four_wire_resistance = np.nan
    if resistivity_mode != "V_OVER_I":
        thermal_offset_v, four_wire_resistance = read_with_output_off(resistivity_mode)
        if resistivity_mode == "OFFSET_CORRECTED":
            if not np.isfinite(thermal_offset_v):
                print("Thermal-offset reading was invalid; treating this sample as invalid.")
                return measured_voltage, measured_current, np.nan, np.nan
            # The quiet-window reading is the contact thermal EMF, not sample drop.
            measured_voltage = measured_voltage - thermal_offset_v

    if config is not None and np.isfinite(measured_voltage) and np.isfinite(measured_current):
        _enforce_electrical_safety(measured_voltage, measured_current, config)

    if resistivity_mode == "FOUR_WIRE":
        resistance = four_wire_resistance
    else:
        if (
            not np.isfinite(measured_voltage)
            or not np.isfinite(measured_current)
            or abs(measured_current) < 1e-12
        ):
            return measured_voltage, measured_current, np.nan, np.nan
        resistance = _calculate_resistance(measured_voltage, measured_current, config=config)

    if not np.isfinite(resistance) or resistance <= 0:
        print(f"Invalid resistance calculated from V={measured_voltage}, I={measured_current}")
        return measured_voltage, measured_current, np.nan, np.nan
    try:
        temperature = float(temperature_interp(resistance))
    except Exception as exc:
        print(f"An error occurred interpolating temperature: {exc}")
        temperature = np.nan

    if config is not None and not _resistance_in_curve_bounds(resistance, temperature_interp, config):
        print(
            f"Measured resistance {resistance:.6f} Ohm is outside the configured R vs. T range; "
            "no temperature can be inferred from it."
        )
        # The resistance itself is a good measurement; only the conversion is out
        # of range. T0 calibration relies on that to anchor an unscaled curve.
        return measured_voltage, measured_current, np.nan, resistance

    temperature_bounds = getattr(temperature_interp, "temperature_bounds", None)
    if np.isfinite(temperature) and temperature_bounds is not None:
        lower_bound, upper_bound = temperature_bounds
        if temperature < lower_bound or temperature > upper_bound:
            print(
                f"Calculated temperature {temperature:.2f} C is outside the configured conversion range "
                f"{lower_bound:.2f}..{upper_bound:.2f} C; treating it as invalid."
            )
            temperature = np.nan

    if np.isfinite(temperature) and temperature < 0 and not calibration:
        print(f"Calculated temperature is {temperature}; treating it as invalid.")
        temperature = np.nan

    return measured_voltage, measured_current, temperature, resistance
