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
    "max_voltage": 30.0,
    "max_current": 3.0,
    "max_power_w": 2.5,
    "dmm_voltage_range_v": 0.2,
    "dmm_current_range_a": 0.002,
    "dmm_staged_ranging_enabled": True,
    "dmm_range_switch_fraction": 0.8,
    "dmm_range_settle_time_s": 0.3,
    "dmm_range_discard_readings": 2,
    "pid_kp": 0.008,
    "pid_ki": 0.0004,
    "pid_kd": 0.0,
    "pid_integral_limit": 400.0,
    "pid_derivative_filter": 0.6,
    "startup_voltage": 0.01,
    "min_voltage": 0.0,
    "psu_keepalive_voltage": 0.001,
    "fixed_series_resistance_ohm": 0.0,
    "max_voltage_step_up": 0.01,
    "max_voltage_step_down": 0.01,
    "low_voltage_step_threshold": 0.05,
    "low_voltage_max_step_up": 0.001,
    "low_voltage_max_step_down": 0.001,
    "temperature_tolerance_c": 2.0,
    "hold_entry_tolerance_c": 3.0,
    "safety_temp_margin_c": 15.0,
    "soft_temp_rate_margin_c_min": 1.0,
    "hard_temp_rate_margin_c_min": 4.0,
    "measurement_fail_limit": 20,
    "minimum_current_a": 1e-4,
    "minimum_voltage_change": 1e-4,
    "measurement_voltage_floor": 0.01,
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
    "curve_extrapolation_max_temperature_c": 600.0,
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
    "measurement_temp_jump_c": 8.0,
    "measurement_temp_jump_up_c": 20.0,
    "measurement_temp_jump_down_c": 8.0,
    "measurement_jump_confirm_min_current_a": 0.02,
    "measurement_jump_confirm_min_voltage": 0.1,
    "measurement_temp_jump_accept_up_c": 35.0,
    "measurement_temp_jump_accept_setpoint_margin_c": 15.0,
    "low_signal_jump_confirm_samples": 3,
    "low_signal_jump_temperature_tolerance_c": 10.0,
    "low_signal_jump_resistance_tolerance_ohm": 0.015,
    "low_signal_recovery_trigger_cycles": 5,
    "low_signal_recovery_observe_cycles": 5,
    "low_signal_recovery_voltage_step": 0.01,
    "low_signal_recovery_max_attempts": 5,
    "measurement_cooldown_confirm_samples": 2,
    "measurement_heatup_confirm_samples": 2,
    "measurement_jump_probe_threshold_c": 35.0,
    "measurement_jump_probe_voltage_step": 0.002,
    "measurement_jump_probe_temperature_tolerance_c": 50.0,
    "measurement_jump_probe_resistance_ratio": 0.02,
    "measurement_jump_probe_max_samples": 20,
    "ignore_invalid_below_voltage": 0.05,
    "invalid_voltage_step_down": 0.01,
    "invalid_reuse_hold_after": 8,
    "invalid_max_drop_from_recent_peak_v": 0.1,
    "invalid_reuse_stop_after": 30,
    "rate_limit_activation_band_c": 2.0,
    "under_target_no_decrease_band_c": 1.5,
    "autosave_flush_interval_s": 5.0,
    "autosave_batch_size": 10,
    "tuning_voltage_step": 0.01,
    "tuning_start_voltage": 0.01,
    "tuning_search_max_voltage": 0.5,
    "tuning_settle_time_s": 0.3,
    "tuning_response_voltage_step": 0.01,
    "tuning_between_attempts_s": 0.5,
    "tuning_max_duration_s": 180.0,
    "tuning_baseline_samples": 2,
    "tuning_stable_current_samples": 3,
    "tuning_stable_current_a": 1e-4,
    "tuning_temperature_window_c": 40.0,
    "tuning_target_rise_c": 1.2,
    "tuning_min_temperature_rise_c": 0.8,
    "tuning_no_response_timeout_s": 25.0,
    "tuning_min_observable_rise_c": 0.25,
    "tuning_plateau_timeout_s": 15.0,
    "tuning_plateau_idle_timeout_s": 6.0,
    "max_voltage_step_up_far": 0.01,
    "aggressive_step_band_c": 4.0,
    "tuning_plateau_growth_c": 0.08,
    "t0_calibration_voltage": 0.1,
    "t0_voltage_search_start": 0.01,
    "t0_voltage_step": 0.01,
    "t0_settle_time_s": 3.0,
    "t0_calibration_samples": 9,
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
    origin_voltage: float = np.nan
    confirmations: int = 0
    attempts: int = 0

    @property
    def active(self):
        return self.direction in {"up", "down"}

    def reset(self):
        self.direction = None
        self.candidate_temperature = np.nan
        self.candidate_resistance = np.nan
        self.origin_voltage = np.nan
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
class LowSignalVoltageRecovery:
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


def _enforce_electrical_safety(measured_voltage, measured_current, config):
    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current):
        return
    if abs(float(measured_current)) > float(config["max_current"]):
        raise ExperimentSafetyError(
            f"Measured current {measured_current:.4e} A exceeded max_current "
            f"{float(config['max_current']):.4e} A."
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


def _measurement_voltage_floor(config):
    minimum = float(config["min_voltage"])
    maximum = float(config["max_voltage"])
    candidates = (
        minimum,
        float(config.get("measurement_voltage_floor", minimum)),
        float(config.get("startup_voltage", minimum)),
        float(config.get("t0_voltage_search_start", minimum)),
    )
    if not all(np.isfinite(value) for value in candidates) or not np.isfinite(maximum):
        raise ValueError("Initial and minimum voltage settings must be finite.")
    return _clamp(max(candidates), minimum, maximum)


def _limit_voltage_slew(target_voltage, current_voltage, min_voltage, max_voltage, config):
    if not np.isfinite(target_voltage) or not np.isfinite(current_voltage):
        return _clamp(current_voltage, min_voltage, max_voltage)
    max_step_up = float(config.get("max_voltage_step_up", 0.01))
    max_step_down = float(config.get("max_voltage_step_down", 0.01))
    low_voltage_threshold = float(config.get("low_voltage_step_threshold", 0.05))
    if current_voltage <= low_voltage_threshold + 1e-12:
        max_step_up = min(max_step_up, float(config.get("low_voltage_max_step_up", 0.001)))
        max_step_down = min(max_step_down, float(config.get("low_voltage_max_step_down", 0.001)))
    if max_step_up <= 0 or max_step_down <= 0:
        raise ValueError("Voltage slew limits must be positive.")
    delta = target_voltage - current_voltage
    if delta > max_step_up:
        target_voltage = current_voltage + max_step_up
    elif delta < -max_step_down:
        target_voltage = current_voltage - max_step_down
    return _clamp(target_voltage, min_voltage, max_voltage)


def _voltage_ramp_command(start_voltage, ramp_speed_min, elapsed_s, applied_voltage, config):
    """Return the elapsed-time voltage-ramp command after applying the normal slew limit."""
    values = (start_voltage, ramp_speed_min, elapsed_s, applied_voltage)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Voltage-ramp inputs must be finite.")
    if ramp_speed_min <= 0 or elapsed_s < 0:
        raise ValueError("Voltage-ramp speed must be positive and elapsed time cannot be negative.")

    minimum_voltage = _measurement_voltage_floor(config)
    maximum_voltage = float(config["max_voltage"])
    requested_voltage = float(start_voltage) + float(ramp_speed_min) * float(elapsed_s) / 60.0
    requested_voltage = _clamp(requested_voltage, minimum_voltage, maximum_voltage)
    return _limit_voltage_slew(
        requested_voltage,
        float(applied_voltage),
        minimum_voltage,
        maximum_voltage,
        config,
    )


def get_controller_mode(config):
    mode = str(config.get("controller_mode", CONTROL_DEFAULTS["controller_mode"])).strip().upper()
    return mode if mode in {"PI", "PID"} else CONTROL_DEFAULTS["controller_mode"]


def get_experiment_mode(config):
    raw_mode = config.get("experiment_mode", config.get("measurement_conversion_mode", "TEMPERATURE"))
    mode = str(raw_mode).strip().upper()
    if mode in {"CONTROLLED", "INTERPOLATE"}:
        return "TEMPERATURE"
    if mode in {"CURVE_SWEEP", "LINEAR_TEMP"}:
        return "VOLTAGE"
    return mode if mode in {"TEMPERATURE", "VOLTAGE"} else CONTROL_DEFAULTS["experiment_mode"]


def build_control_config(config):
    merged = dict(config)
    for key, value in CONTROL_DEFAULTS.items():
        merged.setdefault(key, value)
    merged["controller_mode"] = get_controller_mode(merged)
    merged["experiment_mode"] = get_experiment_mode(merged)
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
    pid_voltage,
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
            pid_voltage,
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
    pid_voltage,
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
            pid_voltage,
            measured_power,
            measured_resistance,
        ]
    )


def _is_valid_measurement(measured_voltage, measured_current, temperature, config):
    if not all(np.isfinite(value) for value in (measured_voltage, measured_current, temperature)):
        return False
    if abs(measured_current) < config["minimum_current_a"]:
        return False
    return True


def _temperature_rate_c_min(current_temperature, previous_temperature, dt):
    if previous_temperature is None or dt <= 0:
        return None
    return (current_temperature - previous_temperature) * 60.0 / dt


def _is_low_signal_state(applied_voltage, config):
    if not np.isfinite(applied_voltage):
        return False
    return applied_voltage <= float(
        config.get(
            "ignore_invalid_below_voltage",
            max(config.get("measurement_voltage_floor", 0.01) * 5.0, 0.05),
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


def _advance_low_signal_voltage_recovery(
    recovery,
    invalid_reuse_streak,
    low_signal_state,
    applied_voltage,
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
        return float(applied_voltage), False
    if recovery.invalid_samples_since_step < observe_cycles:
        return float(applied_voltage), False

    if not np.isfinite(measured_current) or abs(measured_current) >= 0.95 * float(config["max_current"]):
        return float(applied_voltage), False

    voltage_step = max(float(config.get("low_signal_recovery_voltage_step", 0.01)), 0.0)
    requested_voltage = _clamp(
        float(applied_voltage) + voltage_step,
        _measurement_voltage_floor(config),
        float(config["max_voltage"]),
    )
    minimum_change = max(float(config.get("minimum_voltage_change", 1e-4)), 0.0)
    if requested_voltage < float(applied_voltage) + minimum_change:
        return float(applied_voltage), False

    recovery.attempts += 1
    recovery.invalid_samples_since_step = 0
    return requested_voltage, True


def _start_control_at_initial_voltage(
    power_supply,
    config,
    previous_voltage,
    loop_time,
):
    initial_voltage = _measurement_voltage_floor(config)
    previous_voltage = _set_voltage_if_needed(power_supply, initial_voltage, previous_voltage, config)
    settle_time = max(float(config.get("startup_settle_time_s", 1.0)), 0.0)
    print(
        f"Starting controller directly at Initial Voltage {initial_voltage:.4f} V; "
        "this value remains the experiment voltage floor."
    )
    time.sleep(max(settle_time, loop_time))
    return initial_voltage, previous_voltage


def _measure_with_retry(
    dmm_v,
    dmm_i,
    siglent_module,
    temperature_interp,
    *,
    config,
    previous_resistance=None,
):
    measured_voltage, measured_current, temperature = measure_resistivity(
        dmm_v,
        dmm_i,
        siglent_module,
        temperature_interp,
        config=config,
    )
    resistance = _calculate_resistance(measured_voltage, measured_current, config=config)
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
        retry_voltage, retry_current, retry_temperature = measure_resistivity(
            dmm_v,
            dmm_i,
            siglent_module,
            temperature_interp,
            config=config,
        )
        retry_resistance = _calculate_resistance(retry_voltage, retry_current, config=config)
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


def _set_voltage_if_needed(power_supply, voltage, previous_voltage, config):
    if previous_voltage is None or abs(voltage - previous_voltage) >= config["minimum_voltage_change"]:
        siglent.set_voltage(power_supply, voltage=voltage)
        return voltage
    return previous_voltage


def _curve_ordered_temperature_profile(r_vs_t):
    curve = np.asarray(r_vs_t, dtype=float)
    temperature_order = np.argsort(curve[1, :])
    ordered = curve[:, temperature_order]
    _, unique_indices = np.unique(ordered[1, :], return_index=True)
    ordered = ordered[:, np.sort(unique_indices)]
    return ordered[0, :], ordered[1, :]


def build_curve_shaped_voltage_schedule(r_vs_t, start_voltage, end_voltage, steps):
    if steps < 2:
        raise ValueError("Curve sweep requires at least two voltage points.")

    resistance_axis, temperature_axis = _curve_ordered_temperature_profile(r_vs_t)
    if temperature_axis.size < 2:
        raise ValueError("R vs. T data must contain at least two unique temperature points.")

    target_temperatures = np.linspace(float(temperature_axis[0]), float(temperature_axis[-1]), steps)
    target_resistances = np.interp(target_temperatures, temperature_axis, resistance_axis)

    resistance_span = float(np.max(target_resistances) - np.min(target_resistances))
    if resistance_span <= 1e-12:
        voltage_fractions = np.linspace(0.0, 1.0, steps)
    else:
        if target_resistances[-1] >= target_resistances[0]:
            voltage_fractions = (target_resistances - float(np.min(target_resistances))) / resistance_span
        else:
            voltage_fractions = (float(np.max(target_resistances)) - target_resistances) / resistance_span
        voltage_fractions = np.maximum.accumulate(np.clip(voltage_fractions, 0.0, 1.0))
        voltage_fractions[0] = 0.0
        voltage_fractions[-1] = 1.0

    voltages = start_voltage + voltage_fractions * (end_voltage - start_voltage)
    voltages = np.maximum.accumulate(np.asarray(voltages, dtype=float))
    voltages[-1] = end_voltage
    return voltages, target_temperatures


def _compute_next_voltage(
    pid_controller,
    temperature,
    setpoint,
    current_voltage,
    measured_current,
    target_temperature,
    temp_rate_c_min,
    ramp_speed_min,
    config,
    loop_time,
):
    control_min_voltage = _measurement_voltage_floor(config)
    if abs(measured_current) > config["max_current"]:
        raise ExperimentSafetyError(
            f"Measured current {measured_current:.4e} A exceeded max_current {config['max_current']:.4e} A."
        )

    if temperature > target_temperature + config["safety_temp_margin_c"]:
        raise ExperimentSafetyError(
            f"Measured temperature {temperature:.2f} C exceeded the safety limit near target {target_temperature:.2f} C."
        )

    delta_voltage = pid_controller.compute(temperature, dt=loop_time, setpoint=setpoint)
    if not np.isfinite(delta_voltage):
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

    if temperature <= setpoint - under_target_band and delta_voltage < 0.0:
        delta_voltage = 0.0

    aggressive_step = float(config.get("max_voltage_step_up_far", config["max_voltage_step_up"]))
    far_below_setpoint = temperature <= setpoint - config.get("aggressive_step_band_c", 4.0)
    significantly_below_setpoint = temperature <= setpoint - rate_limit_band
    catchup_rate_c_min = max(ramp_speed_min * 0.6, ramp_speed_min - 3.0, 1.0)
    if far_below_setpoint and not current_limited:
        if temp_rate_c_min is None or not np.isfinite(temp_rate_c_min) or temp_rate_c_min < catchup_rate_c_min:
            delta_voltage = max(delta_voltage, aggressive_step)
        else:
            delta_voltage = max(delta_voltage, config["max_voltage_step_up"])
    elif significantly_below_setpoint and not current_limited:
        delta_voltage = max(delta_voltage, config["max_voltage_step_up"])

    if temperature >= setpoint + config["temperature_tolerance_c"]:
        delta_voltage = min(delta_voltage, 0.0)

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
        if near_setpoint and temp_rate_c_min > soft_rate_limit and delta_voltage > 0.0:
            delta_voltage = 0.0
        if (
            near_setpoint
            and temp_rate_c_min > soft_rate_limit
            and temperature >= setpoint - config["temperature_tolerance_c"]
        ):
            delta_voltage = min(delta_voltage, -config["max_voltage_step_down"] / 2.0)
        if near_setpoint and temp_rate_c_min > hard_rate_limit:
            pid_controller.reset(measurement=temperature)
            delta_voltage = -config["max_voltage_step_down"]

    if current_limited and delta_voltage > 0.0:
        delta_voltage = 0.0

    if temperature >= target_temperature and setpoint >= target_temperature:
        delta_voltage = min(delta_voltage, 0.0)

    requested_voltage = _clamp(current_voltage + delta_voltage, control_min_voltage, config["max_voltage"])
    new_voltage = _limit_voltage_slew(
        requested_voltage,
        current_voltage,
        control_min_voltage,
        config["max_voltage"],
        config,
    )
    return new_voltage


def _confirmed_upward_temperature_jump(
    temperature,
    previous_temperature,
    measured_resistance,
    previous_resistance,
    measured_current,
    applied_voltage,
    resistance_confirmed,
    setpoint,
    config,
):
    if not resistance_confirmed:
        return False
    if not all(
        np.isfinite(value)
        for value in (
            temperature,
            previous_temperature,
            measured_resistance,
            previous_resistance,
            measured_current,
            applied_voltage,
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
    minimum_confirm_voltage = max(
        config.get("ignore_invalid_below_voltage", 0.05) * 2.0,
        float(config.get("measurement_jump_confirm_min_voltage", 0.1)),
    )
    if abs(measured_current) < minimum_confirm_current or applied_voltage < minimum_confirm_voltage:
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
    applied_voltage,
    resistance_confirmed,
    setpoint,
    config,
):
    if not all(
        np.isfinite(value)
        for value in (
            temperature,
            previous_temperature,
            measured_resistance,
            previous_resistance,
            measured_current,
            applied_voltage,
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
    minimum_confirm_voltage = max(
        config.get("ignore_invalid_below_voltage", 0.05) * 2.0,
        float(config.get("measurement_jump_confirm_min_voltage", 0.1)),
    )
    return abs(measured_current) >= minimum_confirm_current and applied_voltage >= minimum_confirm_voltage


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
    applied_voltage,
    resistance_confirmed,
    config,
):
    if direction not in {"up", "down"} or not resistance_confirmed:
        return False
    if not all(
        np.isfinite(value)
        for value in (
            temperature,
            previous_temperature,
            measured_resistance,
            previous_resistance,
            measured_current,
            applied_voltage,
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
    minimum_confirm_voltage = max(
        config.get("ignore_invalid_below_voltage", 0.05) * 2.0,
        float(config.get("measurement_jump_confirm_min_voltage", 0.1)),
    )
    return abs(measured_current) >= minimum_confirm_current and applied_voltage >= minimum_confirm_voltage


def _temperature_jump_probe_voltage(direction, applied_voltage, measured_current, config):
    step = max(
        float(config.get("measurement_jump_probe_voltage_step", 0.002)),
        float(config.get("minimum_voltage_change", 1e-4)),
    )
    lower_bound = max(
        float(config["min_voltage"]),
        float(config.get("measurement_voltage_floor", config["min_voltage"])),
    )
    if direction == "up":
        return _clamp(applied_voltage - step, lower_bound, float(config["max_voltage"]))

    if not np.isfinite(measured_current) or abs(measured_current) >= 0.95 * float(config["max_current"]):
        return float(applied_voltage)
    return _clamp(applied_voltage + step, lower_bound, float(config["max_voltage"]))


def _advance_temperature_jump_probe(
    probe,
    direction,
    temperature,
    resistance,
    applied_voltage,
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
        probe.origin_voltage = float(applied_voltage)
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
        voltage_step = max(
            float(config.get("measurement_jump_probe_voltage_step", 0.002)),
            float(config.get("minimum_voltage_change", 1e-4)),
        )
        minimum_probe_change = max(
            float(config.get("minimum_voltage_change", 1e-4)),
            voltage_step * 0.25,
        )
        voltage_was_probed = (
            applied_voltage <= probe.origin_voltage - minimum_probe_change
            if direction == "up"
            else applied_voltage >= probe.origin_voltage + minimum_probe_change
        )
        candidate_is_consistent = (
            abs(temperature - probe.candidate_temperature) <= temperature_tolerance
            and abs(resistance - probe.candidate_resistance) <= resistance_tolerance
        )

        if candidate_is_consistent and voltage_was_probed:
            probe.confirmations += 1
        else:
            # Keep the initial candidate and voltage as the fixed probe reference.
            # An inconsistent reading does not slide the trusted temperature window.
            probe.confirmations = 0

    if probe.confirmations >= required_confirmations:
        attempts = probe.attempts
        probe.reset()
        return True, None, attempts

    if probe.attempts >= maximum_attempts:
        raise ExperimentSafetyError(
            "Large temperature/resistance jump did not stabilize during the controlled voltage probe "
            f"after {probe.attempts} samples. Stopping instead of controlling from an uncertain temperature."
        )

    requested_voltage = _temperature_jump_probe_voltage(
        direction,
        float(applied_voltage),
        float(measured_current),
        config,
    )
    return False, requested_voltage, probe.attempts


def _psu_keepalive_voltage(config):
    try:
        keepalive_voltage = float(config.get("psu_keepalive_voltage", 0.001))
    except (TypeError, ValueError) as exc:
        raise ValueError("psu_keepalive_voltage must be a positive finite voltage.") from exc

    max_voltage = float(config["max_voltage"])
    if not np.isfinite(keepalive_voltage) or keepalive_voltage <= 0 or keepalive_voltage > max_voltage:
        raise ValueError(
            "psu_keepalive_voltage must be positive, finite, and no greater than max_voltage."
        )
    return keepalive_voltage


def prepare_power_supply_output(power_supply, config):
    """Keep CH1 enabled at a negligible voltage while a run is prepared."""
    keepalive_voltage = _psu_keepalive_voltage(config)
    siglent.set_voltage(power_supply, voltage=keepalive_voltage)
    siglent.set_output(power_supply, state="ON")
    print(f"Power supply output enabled at keep-alive voltage {keepalive_voltage:.6f} V.")


def _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager, config=None):
    if power_supply is not None:
        config = config or CONTROL_DEFAULTS
        keepalive_voltage = None
        try:
            keepalive_voltage = _psu_keepalive_voltage(config)
            siglent.set_voltage(power_supply, voltage=keepalive_voltage)
        except Exception as exc:
            print(f"An error occurred in setting the PSU keep-alive voltage: {exc}")
        if keepalive_voltage is not None:
            time.sleep(0.1)
            print(f"Power supply output left ON at {keepalive_voltage:.6f} V.")
        else:
            try:
                siglent.set_output(power_supply, state="OFF")
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
    loop_time = 1.0 / config["experiment_frequency"]
    max_voltage = min(float(config["max_voltage"]), float(sweep_params.get("max_voltage", config["max_voltage"])))
    start_voltage = max(
        float(config.get("curve_sweep_start_voltage", 0.01)),
        float(config.get("measurement_voltage_floor", 0.01)),
        0.01,
    )
    start_voltage = _clamp(start_voltage, float(config["min_voltage"]), max_voltage)
    voltage_step = max(float(config.get("curve_sweep_voltage_step", 0.005)), 1e-6)
    requested_steps = max(2, int(np.ceil(max_voltage / voltage_step)))

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

        schedule_voltages, schedule_temperatures = build_curve_shaped_voltage_schedule(
            r_vs_t,
            start_voltage=start_voltage,
            end_voltage=max_voltage,
            steps=requested_steps,
        )
        print(
            f"Curve sweep: start={start_voltage:.4f} V, end={max_voltage:.4f} V, "
            f"steps={requested_steps}, step_basis={voltage_step:.4f} V"
        )

        previous_voltage = None
        previous_resistance = None
        for target_temperature, target_voltage in zip(schedule_temperatures, schedule_voltages):
            if emitter.stopped:
                print("Stop signal received.")
                break

            loop_started = time.time()
            previous_voltage = _set_voltage_if_needed(power_supply, float(target_voltage), previous_voltage, config)
            time.sleep(max(loop_time, 0.2))

            measured_voltage, measured_current, temperature, measured_resistance, _ = _measure_with_retry(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                config=config,
                previous_resistance=previous_resistance,
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
                f"PSU command: {previous_voltage:.4f} V"
            )
            _persist_measurement(
                data_saver,
                float(target_temperature),
                temperature,
                measured_voltage,
                measured_current,
                float(previous_voltage),
                measured_resistance,
            )
            _emit_measurement(
                emitter,
                float(target_temperature),
                temperature,
                measured_voltage,
                measured_current,
                float(previous_voltage),
                measured_resistance,
            )

            elapsed = time.time() - loop_started
            if elapsed < loop_time:
                time.sleep(loop_time - elapsed)

    finally:
        _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager, config=config)
        if data_saver is not None:
            data_saver.finalize()
        print("Curve sweep thread finished.")


def voltage_ramp(emitter, ramp_params, r_vs_t, config, data_saver=None):
    """Ramp the PSU command by elapsed time while retaining current, power, and voltage safety limits."""
    if r_vs_t is None:
        raise ValueError("An R-vs-T table must be loaded before starting a voltage ramp.")

    config = build_control_config(config)
    temperature_interp = build_temperature_interpolator(r_vs_t, config=config)
    loop_time = 1.0 / float(config["experiment_frequency"])
    ramp_speed_v_min = float(ramp_params["ramp_speed_min"])
    if not np.isfinite(ramp_speed_v_min) or ramp_speed_v_min <= 0:
        raise ValueError("Voltage-mode ramp_speed_min must be positive and finite.")

    measurement_voltage_floor = _measurement_voltage_floor(config)
    maximum_voltage = float(config["max_voltage"])
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

        commanded_voltage, previous_voltage = _start_control_at_initial_voltage(
            power_supply,
            config,
            previous_voltage=None,
            loop_time=loop_time,
        )
        ramp_started = time.monotonic()
        previous_resistance = None
        print(
            f"Voltage mode: start={commanded_voltage:.6f} V, "
            f"ramp_speed={ramp_speed_v_min:.6f} V/min, max_power={max_power_w:.6f} W, "
            f"absolute_voltage_ceiling={maximum_voltage:.6f} V."
        )

        while not emitter.stopped:
            loop_started = time.monotonic()
            applied_voltage = float(commanded_voltage)
            measured_voltage, measured_current, temperature, measured_resistance, _ = _measure_with_retry(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                config=config,
                previous_resistance=previous_resistance,
            )
            _enforce_electrical_safety(measured_voltage, measured_current, config)
            measured_power_w = _sample_power_w(measured_voltage, measured_current)
            if np.isfinite(measured_resistance):
                previous_resistance = measured_resistance

            print(
                f"Voltage mode: T={temperature if np.isfinite(temperature) else float('nan'):.2f} C, "
                f"Vsample={measured_voltage:.6f} V, Current={measured_current:.6e} A, "
                f"Power={measured_power_w:.6f} W, PSU command={applied_voltage:.6f} V."
            )
            _persist_measurement(
                data_saver,
                np.nan,
                temperature,
                measured_voltage,
                measured_current,
                applied_voltage,
                measured_resistance,
            )
            _emit_measurement(
                emitter,
                np.nan,
                temperature,
                measured_voltage,
                measured_current,
                applied_voltage,
                measured_resistance,
            )

            elapsed_ramp_s = time.monotonic() - ramp_started
            commanded_voltage = _voltage_ramp_command(
                measurement_voltage_floor,
                ramp_speed_v_min,
                elapsed_ramp_s,
                applied_voltage,
                config,
            )
            if (
                applied_voltage >= maximum_voltage - float(config.get("minimum_voltage_change", 1e-4))
                and commanded_voltage <= applied_voltage + 1e-12
            ):
                raise ExperimentSafetyError(
                    f"Voltage mode reached the absolute software voltage ceiling {maximum_voltage:.6f} V "
                    f"before reaching max_power_w {max_power_w:.6f} W."
                )
            previous_voltage = _set_voltage_if_needed(
                power_supply,
                commanded_voltage,
                previous_voltage,
                config,
            )

            elapsed_loop_s = time.monotonic() - loop_started
            if elapsed_loop_s < loop_time:
                time.sleep(loop_time - elapsed_loop_s)

        if emitter.stopped:
            print("Stop signal received.")
    finally:
        _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager, config=config)
        if data_saver is not None:
            data_saver.finalize()
        print("Voltage-ramp thread finished.")


def tds(emitter, experiment_params, r_vs_t, config, t_zero, data_saver=None):
    if r_vs_t is None:
        raise ValueError("A resistivity-versus-temperature table must be loaded before starting an experiment.")

    config = build_control_config(config)
    temperature_interp = build_temperature_interpolator(r_vs_t, config=config)
    _validate_temperature_program_bounds(experiment_params, temperature_interp)
    loop_time = 1.0 / config["experiment_frequency"]

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

        previous_voltage = None
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
                output_limits=(-config["max_voltage_step_down"], config["max_voltage_step_up"]),
                integral_limits=(-config["pid_integral_limit"], config["pid_integral_limit"]),
                derivative_filter=config["pid_derivative_filter"],
            )

            measurement_voltage_floor = _measurement_voltage_floor(config)
            pid_voltage, previous_voltage = _start_control_at_initial_voltage(
                power_supply=power_supply,
                config=config,
                previous_voltage=previous_voltage,
                loop_time=loop_time,
            )
            print(
                f"Using calibrated T0 {float(t_zero):.2f} C as the initial trusted temperature; "
                f"the first live reading will control the next voltage from {pid_voltage:.4f} V."
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
            low_signal_voltage_recovery = LowSignalVoltageRecovery()
            last_program_update_time = time.monotonic()

            while not emitter.stopped:
                loop_started = time.time()
                program_update_time = time.monotonic()
                program_dt = max(program_update_time - last_program_update_time, 0.0)
                last_program_update_time = program_update_time
                applied_voltage = pid_voltage
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
                )
                raw_temperature = temperature
                low_signal_state = _is_low_signal_state(applied_voltage, config)
                low_signal_jump_pending = False
                low_signal_jump_confirmed = False
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
                    applied_voltage=applied_voltage,
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
                    applied_voltage=applied_voltage,
                    resistance_confirmed=resistance_confirmed,
                    setpoint=float(program.scheduled_target),
                    config=config,
                )
                reset_temperature_reference = low_signal_jump_confirmed
                jump_probe_voltage_request = None

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
                        probe_eligible = _temperature_jump_probe_eligible(
                            "down",
                            temperature,
                            previous_temperature,
                            measured_resistance,
                            previous_resistance,
                            measured_current,
                            applied_voltage,
                            resistance_confirmed,
                            config,
                        )
                        if large_jump:
                            pending_cooldown_jump_count = 0
                            pending_heatup_jump_count = 0
                            if probe_eligible:
                                probe_confirmed, jump_probe_voltage_request, probe_attempt = (
                                    _advance_temperature_jump_probe(
                                        temperature_jump_probe,
                                        "down",
                                        temperature,
                                        measured_resistance,
                                        applied_voltage,
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
                                else:
                                    probe_action = (
                                        "increasing"
                                        if jump_probe_voltage_request > applied_voltage + 1e-9
                                        else "holding"
                                    )
                                    print(
                                        f"Large downward temperature jump detected: previous={previous_temperature:.2f} C, "
                                        f"candidate={temperature:.2f} C, R={measured_resistance:.4f} Ohm. "
                                        f"Probe sample {probe_attempt}: {probe_action} PSU slightly from "
                                        f"{applied_voltage:.4f} to {jump_probe_voltage_request:.4f} V before deciding."
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
                        probe_eligible = _temperature_jump_probe_eligible(
                            "up",
                            temperature,
                            previous_temperature,
                            measured_resistance,
                            previous_resistance,
                            measured_current,
                            applied_voltage,
                            resistance_confirmed,
                            config,
                        )
                        if large_jump:
                            pending_cooldown_jump_count = 0
                            pending_heatup_jump_count = 0
                            if probe_eligible:
                                probe_confirmed, jump_probe_voltage_request, probe_attempt = (
                                    _advance_temperature_jump_probe(
                                        temperature_jump_probe,
                                        "up",
                                        temperature,
                                        measured_resistance,
                                        applied_voltage,
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
                                else:
                                    probe_action = (
                                        "decreasing"
                                        if jump_probe_voltage_request < applied_voltage - 1e-9
                                        else "holding"
                                    )
                                    print(
                                        f"Large upward temperature jump detected: previous={previous_temperature:.2f} C, "
                                        f"candidate={temperature:.2f} C, R={measured_resistance:.4f} Ohm. "
                                        f"Probe sample {probe_attempt}: {probe_action} PSU slightly from "
                                        f"{applied_voltage:.4f} to {jump_probe_voltage_request:.4f} V before deciding."
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
                            raise ExperimentSafetyError(
                                "Large temperature/resistance jump probe could not obtain stable readings after "
                                f"{temperature_jump_probe.attempts} samples. Stopping instead of controlling "
                                "from an uncertain temperature."
                            )
                        jump_probe_voltage_request = _temperature_jump_probe_voltage(
                            temperature_jump_probe.direction,
                            applied_voltage,
                            measured_current,
                            config,
                        )
                        print(
                            "Temperature-jump probe received an unusable measurement; repeating the small "
                            f"voltage probe at {jump_probe_voltage_request:.4f} V."
                        )
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
                            invalid_recovery_peak_voltage = float(applied_voltage)
                        else:
                            invalid_recovery_peak_voltage = max(float(invalid_recovery_peak_voltage), float(applied_voltage))
                        recovery_temperature = previous_temperature

                        if phase != previous_phase:
                            pid_controller.reset(measurement=recovery_temperature)
                            previous_phase = phase
                        else:
                            pid_controller.reset(measurement=recovery_temperature)

                        pid_voltage = _compute_next_voltage(
                            pid_controller=pid_controller,
                            temperature=recovery_temperature,
                            setpoint=setpoint,
                            current_voltage=pid_voltage,
                            measured_current=measured_current,
                            target_temperature=program.target_T,
                            temp_rate_c_min=0.0,
                            ramp_speed_min=program.ramp_speed_min,
                            config=config,
                            loop_time=loop_time,
                        )
                        if low_signal_jump_pending:
                            pid_voltage = min(pid_voltage, applied_voltage)
                        if invalid_reuse_streak >= max(int(config.get("invalid_reuse_hold_after", 8)), 1):
                            # Prevent runaway voltage escalation when we are reusing stale temperature for too long.
                            pid_voltage = min(
                                pid_voltage,
                                applied_voltage - 0.5 * config.get("invalid_voltage_step_down", config["max_voltage_step_up"]),
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
                            pid_voltage = min(
                                pid_voltage,
                                applied_voltage
                                - max(
                                    float(config.get("invalid_voltage_step_down", config["max_voltage_step_up"])),
                                    config["max_voltage_step_up"],
                                ),
                            )
                        elif (
                            not low_signal_state
                            and pid_voltage >= applied_voltage
                            and invalid_reuse_streak < max(int(config.get("invalid_reuse_hold_after", 8)), 1)
                            and (
                                recovery_current_limited
                                or recovery_temperature >= setpoint - recovery_under_target_band
                            )
                        ):
                            pid_voltage = applied_voltage - config["max_voltage_step_up"]
                        pid_voltage = _limit_voltage_slew(
                            pid_voltage,
                            applied_voltage,
                            measurement_voltage_floor,
                            config["max_voltage"],
                            config,
                        )
                        max_invalid_drop = max(float(config.get("invalid_max_drop_from_recent_peak_v", 0.1)), 0.0)
                        invalid_recovery_floor = max(
                            measurement_voltage_floor,
                            float(invalid_recovery_peak_voltage) - max_invalid_drop,
                        )
                        pid_voltage = max(pid_voltage, invalid_recovery_floor)
                        if jump_probe_voltage_request is not None:
                            pid_voltage = _limit_voltage_slew(
                                jump_probe_voltage_request,
                                applied_voltage,
                                measurement_voltage_floor,
                                config["max_voltage"],
                                config,
                            )
                        low_signal_recovery_voltage, low_signal_recovery_stepped = (
                            _advance_low_signal_voltage_recovery(
                                recovery=low_signal_voltage_recovery,
                                invalid_reuse_streak=invalid_reuse_streak,
                                low_signal_state=low_signal_state,
                                applied_voltage=applied_voltage,
                                measured_current=measured_current,
                                config=config,
                            )
                        )
                        if low_signal_recovery_voltage is not None:
                            pid_voltage = low_signal_recovery_voltage
                        if low_signal_recovery_stepped:
                            print(
                                "Low-signal recovery probe "
                                f"{low_signal_voltage_recovery.attempts}/"
                                f"{max(int(config.get('low_signal_recovery_max_attempts', 5)), 1)}: "
                                f"increasing commanded PSU from {applied_voltage:.4f} to {pid_voltage:.4f} V "
                                "and observing the next measurements."
                            )
                        previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)
                        invalid_measurements = 0
                        if (
                            resistance_confirmed
                            and np.isfinite(measured_resistance)
                            and not temperature_jump_probe.active
                        ):
                            previous_resistance = measured_resistance
                        if pid_voltage > applied_voltage + 1e-9:
                            recovery_action = "continuing upward"
                        elif pid_voltage < applied_voltage - 1e-9:
                            recovery_action = "gently backing off"
                        else:
                            recovery_action = "holding"
                        print(
                            f"Ignoring {'low-signal' if low_signal_state else 'transient'} invalid measurement. "
                            f"Measured Vsample={measured_voltage}, I={measured_current} while commanded PSU was {applied_voltage:.4f} V. "
                            f"Reusing last trusted temperature {recovery_temperature:.2f} C and "
                            f"{recovery_action} to {pid_voltage:.4f} V."
                        )
                        _persist_measurement(
                            data_saver,
                            setpoint,
                            recovery_temperature,
                            measured_voltage,
                            measured_current,
                            applied_voltage,
                            measured_resistance,
                        )
                        _emit_measurement(
                            emitter,
                            setpoint,
                            recovery_temperature,
                            measured_voltage,
                            measured_current,
                            applied_voltage,
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
                        invalid_recovery_peak_voltage = float(applied_voltage)
                    else:
                        invalid_recovery_peak_voltage = max(float(invalid_recovery_peak_voltage), float(applied_voltage))
                    pid_controller.reset(measurement=previous_temperature)
                    pid_voltage = _clamp(
                        applied_voltage - config.get("invalid_voltage_step_down", config["max_voltage_step_down"]),
                        measurement_voltage_floor,
                        config["max_voltage"],
                    )
                    pid_voltage = _limit_voltage_slew(
                        pid_voltage,
                        applied_voltage,
                        measurement_voltage_floor,
                        config["max_voltage"],
                        config,
                    )
                    max_invalid_drop = max(float(config.get("invalid_max_drop_from_recent_peak_v", 0.1)), 0.0)
                    invalid_recovery_floor = max(
                        measurement_voltage_floor,
                        float(invalid_recovery_peak_voltage) - max_invalid_drop,
                    )
                    pid_voltage = max(pid_voltage, invalid_recovery_floor)
                    previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)
                    print(
                        "Invalid measurement received. "
                        f"Measured Vsample={measured_voltage}, I={measured_current} while commanded PSU was {applied_voltage:.4f} V. "
                        f"Reducing PSU to {pid_voltage:.4f} V (attempt {invalid_measurements})."
                    )
                    if invalid_measurements >= config["measurement_fail_limit"]:
                        raise ExperimentSafetyError("Too many invalid measurements in a row.")
                    _persist_measurement(
                        data_saver,
                        setpoint,
                        np.nan,
                        measured_voltage,
                        measured_current,
                        applied_voltage,
                        measured_resistance,
                    )
                    _emit_measurement(
                        emitter,
                        setpoint,
                        np.nan,
                        measured_voltage,
                        measured_current,
                        applied_voltage,
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

                pid_voltage = _compute_next_voltage(
                    pid_controller=pid_controller,
                    temperature=filtered_temperature,
                    setpoint=setpoint,
                    current_voltage=pid_voltage,
                    measured_current=measured_current,
                    target_temperature=program.target_T,
                    temp_rate_c_min=temp_rate_c_min,
                    ramp_speed_min=program.ramp_speed_min,
                    config=config,
                    loop_time=loop_time,
                )
                previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)

                print(
                    f"Phase: {phase}, T: {filtered_temperature:.2f} C, Setpoint: {setpoint:.2f} C, "
                    f"Vsample: {measured_voltage:.6f} V, Current: {measured_current:.4e} A, "
                    f"PSU command: {applied_voltage:.4f} -> {pid_voltage:.4f} V, "
                    f"Rate: {temp_rate_c_min if temp_rate_c_min is not None else 0.0:.2f} C/min"
                )
                _persist_measurement(
                    data_saver,
                    setpoint,
                    filtered_temperature,
                    measured_voltage,
                    measured_current,
                    applied_voltage,
                    measured_resistance,
                )
                _emit_measurement(
                    emitter,
                    setpoint,
                    filtered_temperature,
                    measured_voltage,
                    measured_current,
                    applied_voltage,
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
        _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager, config=config)
        if data_saver is not None:
            data_saver.finalize()
        print("TDS experiment thread finished.")


def measure_resistivity(dmm_v, dmm_i, siglent_module, temperature_interp, calibration=False, config=None):
    def read_pair_once():
        synchronized_reader = getattr(siglent_module, "read_DMM_pair", None)
        use_synchronized_reading = config is None or bool(config.get("dmm_synchronized_reading", True))
        if use_synchronized_reading and synchronized_reader is not None:
            try:
                voltage, current = synchronized_reader(dmm_v, dmm_i)
                voltage = float(voltage)
                current = float(current)
            except Exception as exc:
                print(f"Synchronized DMM reading failed; retrying this sample with READ?: {exc}")
                voltage = np.nan
                current = np.nan
        else:
            voltage = np.nan
            current = np.nan

        if not np.isfinite(voltage):
            try:
                voltage = float(siglent_module.read_DMM(dmm_v))
            except Exception as exc:
                print(f"An error occurred reading voltage DMM: {exc}")
                voltage = np.nan

        if not np.isfinite(current):
            try:
                current = float(siglent_module.read_DMM(dmm_i))
            except Exception as exc:
                print(f"An error occurred reading current DMM: {exc}")
                current = np.nan
        return voltage, current

    measured_voltage, measured_current = read_pair_once()

    range_increaser = (
        getattr(siglent_module, "increase_dc_range_if_needed", None)
        if "increase_dc_range_if_needed" in dir(siglent_module)
        else None
    )
    if config is not None and callable(range_increaser):
        voltage_range_change = range_increaser(
            dmm_v, "VOLT", measured_voltage, config
        )
        current_range_change = range_increaser(
            dmm_i, "CURR", measured_current, config
        )
        if voltage_range_change is not None or current_range_change is not None:
            settle_time_s = float(config.get("dmm_range_settle_time_s", 0.3))
            discard_readings = int(config.get("dmm_range_discard_readings", 2))
            if not np.isfinite(settle_time_s) or settle_time_s < 0:
                raise ValueError("dmm_range_settle_time_s must be finite and non-negative.")
            if discard_readings < 0:
                raise ValueError("dmm_range_discard_readings must be non-negative.")
            if settle_time_s:
                time.sleep(settle_time_s)
            for _ in range(discard_readings):
                read_pair_once()
            measured_voltage, measured_current = read_pair_once()

    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current) or abs(measured_current) < 1e-12:
        return measured_voltage, measured_current, np.nan

    if config is not None:
        _enforce_electrical_safety(measured_voltage, measured_current, config)

    resistance = _calculate_resistance(measured_voltage, measured_current, config=config)
    if not np.isfinite(resistance) or resistance <= 0:
        print(f"Invalid resistance calculated from V={measured_voltage}, I={measured_current}")
        return measured_voltage, measured_current, np.nan
    try:
        temperature = float(temperature_interp(resistance))
    except Exception as exc:
        print(f"An error occurred interpolating temperature: {exc}")
        temperature = np.nan

    if config is not None and not _resistance_in_curve_bounds(resistance, temperature_interp, config):
        print(
            f"Measured resistance {resistance:.6f} Ohm is outside the configured R vs. T range; "
            "treating it as invalid."
        )
        return measured_voltage, measured_current, np.nan

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

    return measured_voltage, measured_current, temperature
