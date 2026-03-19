import time
from dataclasses import dataclass

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

from . import pid
from . import siglent


CONTROL_DEFAULTS = {
    "controller_mode": "PI",
    "pid_kp": 0.008,
    "pid_ki": 0.0004,
    "pid_kd": 0.0,
    "pid_integral_limit": 400.0,
    "pid_derivative_filter": 0.6,
    "startup_voltage": 0.01,
    "min_voltage": 0.0,
    "max_voltage_step_up": 0.01,
    "max_voltage_step_down": 0.04,
    "temperature_tolerance_c": 2.0,
    "hold_entry_tolerance_c": 3.0,
    "safety_temp_margin_c": 15.0,
    "soft_temp_rate_margin_c_min": 1.0,
    "hard_temp_rate_margin_c_min": 4.0,
    "measurement_fail_limit": 20,
    "minimum_current_a": 5e-4,
    "minimum_voltage_change": 1e-4,
    "measurement_voltage_floor": 0.01,
    "measurement_filter_samples": 3,
    "resistance_range_margin_ratio": 0.0,
    "resistance_range_margin_ohm": 0.01,
    "warmup_stable_samples": 3,
    "resistance_glitch_jump_ohm": 0.03,
    "resistance_glitch_jump_ratio": 0.015,
    "measurement_retry_attempts": 2,
    "measurement_retry_delay_s": 0.15,
    "measurement_retry_consensus_ohm": 0.015,
    "measurement_temp_jump_c": 8.0,
    "measurement_temp_jump_up_c": 20.0,
    "measurement_temp_jump_down_c": 8.0,
    "measurement_temp_jump_accept_up_c": 35.0,
    "measurement_temp_jump_accept_setpoint_margin_c": 15.0,
    "measurement_cooldown_confirm_samples": 1,
    "ignore_invalid_below_voltage": 0.05,
    "invalid_voltage_step_down": 0.02,
    "rate_limit_activation_band_c": 2.0,
    "under_target_no_decrease_band_c": 1.5,
    "autosave_flush_interval_s": 5.0,
    "autosave_batch_size": 10,
    "tuning_voltage_step": 0.01,
    "tuning_start_voltage": 0.1,
    "tuning_search_max_voltage": 0.5,
    "tuning_settle_time_s": 0.3,
    "tuning_response_voltage_step": 0.05,
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
    "max_voltage_step_up_far": 0.04,
    "aggressive_step_band_c": 4.0,
    "tuning_plateau_growth_c": 0.08,
    "t0_calibration_voltage": 0.1,
    "t0_voltage_search_start": 0.01,
    "t0_voltage_step": 0.01,
    "t0_settle_time_s": 3.0,
    "t0_calibration_samples": 5,
    "t0_warmup_samples": 1,
    "t0_stable_current_samples": 3,
    "t0_stable_current_a": 1e-4,
    "t0_max_temp_error_c": 80.0,
}


class ExperimentSafetyError(RuntimeError):
    """Raised when the experiment should stop to protect the sample or setup."""


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


def get_controller_mode(config):
    mode = str(config.get("controller_mode", CONTROL_DEFAULTS["controller_mode"])).strip().upper()
    return mode if mode in {"PI", "PID"} else CONTROL_DEFAULTS["controller_mode"]


def build_control_config(config):
    merged = dict(config)
    for key, value in CONTROL_DEFAULTS.items():
        merged.setdefault(key, value)
    merged["controller_mode"] = get_controller_mode(merged)
    return merged


def build_temperature_interpolator(r_vs_t):
    curve = np.asarray(r_vs_t, dtype=float)
    if curve.shape[0] != 2 or curve.shape[1] < 2:
        raise ValueError("R vs. T data must have shape 2 x N with at least two points.")

    resistance_order = np.argsort(curve[0, :])
    resistance_curve = curve[:, resistance_order]
    _, unique_indices = np.unique(resistance_curve[0, :], return_index=True)
    resistance_curve = resistance_curve[:, np.sort(unique_indices)]
    return interp1d(
        resistance_curve[0, :],
        resistance_curve[1, :],
        kind="linear",
        fill_value="extrapolate",
    )


@dataclass
class TemperatureProgram:
    start_T: float
    step_T: float
    target_T: float
    ramp_speed_c_min: float
    hold_step_time_min: float
    temperature_tolerance_c: float
    hold_entry_tolerance_c: float
    warmup_stable_samples: int

    def __post_init__(self):
        if self.target_T < self.start_T:
            raise ValueError("target_T must be greater than or equal to start_T.")
        if self.ramp_speed_c_min <= 0:
            raise ValueError("ramp_speed_c_min must be greater than zero.")
        if self.hold_step_time_min < 0:
            raise ValueError("hold_step_time_min must be non-negative.")

        self.simple_ramp = self.step_T <= 0 or self.step_T >= (self.target_T - self.start_T)
        self.ramp_speed_c_s = self.ramp_speed_c_min / 60.0
        self.hold_step_time_s = self.hold_step_time_min * 60.0
        self.phase = "warmup"
        self.scheduled_target = self.start_T
        self.current_plateau = self.start_T
        self.hold_elapsed_s = 0.0
        self.warmup_stable_count = 0

    def initialize(self, initial_temperature):
        self.scheduled_target = min(initial_temperature, self.start_T)
        self.current_plateau = self.start_T
        self.hold_elapsed_s = 0.0
        self.phase = "warmup"
        self.warmup_stable_count = 0

    def _advance_target(self, target_limit, dt):
        self.scheduled_target = min(target_limit, self.scheduled_target + self.ramp_speed_c_s * dt)
        return self.scheduled_target

    def update(self, measured_temperature, dt):
        while True:
            if self.phase == "warmup":
                if measured_temperature > self.start_T + self.temperature_tolerance_c:
                    self.scheduled_target = self.start_T
                    self.warmup_stable_count = 0
                    return self.start_T, self.phase, False
                target = self._advance_target(self.start_T, dt)
                if abs(measured_temperature - self.start_T) <= self.temperature_tolerance_c:
                    self.warmup_stable_count += 1
                else:
                    self.warmup_stable_count = 0
                if target >= self.start_T and self.warmup_stable_count >= self.warmup_stable_samples:
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


def _emit_measurement(emitter, target_temperature, temperature, measured_voltage, measured_current, pid_voltage):
    emitter.experiment_signal.emit(
        [time.time(), target_temperature, temperature, 0, measured_voltage, measured_current, pid_voltage]
    )


def _persist_measurement(data_saver, target_temperature, temperature, measured_voltage, measured_current, pid_voltage):
    if data_saver is None:
        return
    data_saver.enqueue([time.time(), target_temperature, temperature, 0, measured_voltage, measured_current, pid_voltage])


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


def _calculate_resistance(measured_voltage, measured_current):
    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current):
        return np.nan
    if abs(measured_current) < 1e-12:
        return np.nan
    resistance = measured_voltage / measured_current
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
    resistance = _calculate_resistance(measured_voltage, measured_current)
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
        retry_resistance = _calculate_resistance(retry_voltage, retry_current)
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


def _compute_next_voltage(
    pid_controller,
    temperature,
    setpoint,
    current_voltage,
    measured_current,
    target_temperature,
    temp_rate_c_min,
    ramp_speed_c_min,
    config,
    loop_time,
):
    control_min_voltage = _clamp(
        max(config["min_voltage"], config.get("measurement_voltage_floor", config["min_voltage"])),
        config["min_voltage"],
        config["max_voltage"],
    )
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
    catchup_rate_c_min = max(ramp_speed_c_min * 0.6, ramp_speed_c_min - 3.0, 1.0)
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
        ramp_speed_c_min + config["soft_temp_rate_margin_c_min"],
        config["soft_temp_rate_margin_c_min"],
    )
    hard_rate_limit = max(
        ramp_speed_c_min + config["hard_temp_rate_margin_c_min"],
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

    new_voltage = _clamp(current_voltage + delta_voltage, control_min_voltage, config["max_voltage"])
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

    minimum_confirm_current = max(config["minimum_current_a"] * 20.0, 0.05)
    minimum_confirm_voltage = max(
        config.get("ignore_invalid_below_voltage", 0.05) * 4.0,
        0.5,
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
        )
    ):
        return False
    if temperature >= previous_temperature or measured_resistance >= previous_resistance:
        return False

    minimum_confirm_current = max(config["minimum_current_a"] * 20.0, 0.05)
    minimum_confirm_voltage = max(
        config.get("ignore_invalid_below_voltage", 0.05) * 4.0,
        0.5,
    )
    return abs(measured_current) >= minimum_confirm_current and applied_voltage >= minimum_confirm_voltage


def _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager):
    if power_supply is not None:
        try:
            siglent.set_voltage(power_supply, voltage=0.0)
        except Exception as exc:
            print(f"An error occurred in setting voltage to zero: {exc}")
        try:
            time.sleep(0.1)
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


def tds(emitter, experiment_params, r_vs_t, config, t_zero, data_saver=None):
    if r_vs_t is None:
        raise ValueError("A resistivity-versus-temperature table must be loaded before starting an experiment.")

    config = build_control_config(config)
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

        siglent.set_output(power_supply, state="ON")
        time.sleep(0.04)
        siglent.set_voltage(power_supply, voltage=0.0)
        time.sleep(1.0)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])

        temperature_interp = build_temperature_interpolator(r_vs_t)

        previous_voltage = None
        for ex_param in experiment_params:
            print("Experiment parameters:", ex_param)
            program = TemperatureProgram(
                start_T=ex_param["start_T"],
                step_T=ex_param["step_T"],
                target_T=ex_param["target_T"],
                ramp_speed_c_min=ex_param["ramp_speed_c_min"],
                hold_step_time_min=ex_param["hold_step_time_min"],
                temperature_tolerance_c=config["temperature_tolerance_c"],
                hold_entry_tolerance_c=config["hold_entry_tolerance_c"],
                warmup_stable_samples=int(config.get("warmup_stable_samples", 3)),
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

            measurement_voltage_floor = _clamp(
                max(config["startup_voltage"], config.get("measurement_voltage_floor", config["startup_voltage"])),
                config["min_voltage"],
                config["max_voltage"],
            )
            pid_voltage = measurement_voltage_floor
            previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)
            time.sleep(max(2.0, loop_time))

            measured_voltage = np.nan
            measured_current = np.nan
            temperature = np.nan
            for initial_attempt in range(int(config["measurement_fail_limit"])):
                measured_voltage, measured_current, temperature = measure_resistivity(
                    dmm_v, dmm_i, siglent, temperature_interp, config=config
                )
                if _is_valid_measurement(measured_voltage, measured_current, temperature, config):
                    break
                print(
                    "Invalid initial measurement received. "
                    f"Measured Vsample={measured_voltage}, I={measured_current} at PSU {pid_voltage:.4f} V "
                    f"(attempt {initial_attempt + 1}/{config['measurement_fail_limit']})."
                )
                time.sleep(loop_time)
            initial_resistance = _calculate_resistance(measured_voltage, measured_current)
            if not _is_valid_measurement(measured_voltage, measured_current, temperature, config):
                raise ExperimentSafetyError(
                    "Unable to acquire a valid initial measurement after repeated attempts."
                )

            print(
                f"Initial temperature {temperature:.2f} C, start temperature {program.start_T:.2f} C, "
                f"resistance {measured_voltage / measured_current:.6e} Ohm"
            )

            program.initialize(max(temperature, t_zero))
            pid_controller.reset(measurement=temperature)
            invalid_measurements = 0
            temperature_history = [float(temperature)]
            filtered_temperature = _temperature_filter(
                temperature_history,
                temperature,
                config.get("measurement_filter_samples", 3),
            )
            previous_temperature = filtered_temperature
            previous_resistance = initial_resistance
            previous_phase = None
            pending_cooldown_jump_count = 0

            while not emitter.stopped:
                loop_started = time.time()
                applied_voltage = pid_voltage
                measured_voltage, measured_current, temperature, measured_resistance, resistance_confirmed = _measure_with_retry(
                    dmm_v,
                    dmm_i,
                    siglent,
                    temperature_interp,
                    config=config,
                    previous_resistance=previous_resistance,
                )
                raw_temperature = temperature
                low_signal_state = _is_low_signal_state(applied_voltage, config)
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
                    config=config,
                )
                reset_temperature_reference = False

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
                    if temperature_delta < -jump_down_limit:
                        if confirmed_downward_jump:
                            pending_cooldown_jump_count += 1
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
                            pending_cooldown_jump_count = 0
                            print(
                                f"Temperature jump detected: previous={previous_temperature:.2f} C, "
                                f"new={temperature:.2f} C. Treating this reading as invalid."
                            )
                            temperature = np.nan
                    elif temperature_delta > jump_up_limit:
                        if confirmed_upward_jump:
                            print(
                                f"Confirmed upward temperature jump: previous={previous_temperature:.2f} C, "
                                f"new={temperature:.2f} C. Accepting it and resetting the temperature filter."
                            )
                            temperature_history[:] = [float(temperature)]
                            pending_cooldown_jump_count = 0
                            reset_temperature_reference = True
                        else:
                            pending_cooldown_jump_count = 0
                            print(
                                f"Temperature jump detected: previous={previous_temperature:.2f} C, "
                                f"new={temperature:.2f} C. Treating this reading as invalid."
                            )
                            temperature = np.nan
                    else:
                        pending_cooldown_jump_count = 0
                else:
                    pending_cooldown_jump_count = 0

                if not _is_valid_measurement(measured_voltage, measured_current, temperature, config):
                    can_reuse_last_temperature = (
                        previous_temperature is not None
                        and np.isfinite(previous_temperature)
                        and np.isfinite(measured_voltage)
                        and np.isfinite(measured_current)
                        and abs(measured_current) <= config["max_current"]
                    )
                    if can_reuse_last_temperature:
                        recovery_temperature = previous_temperature
                        setpoint, phase, finished = program.update(recovery_temperature, loop_time)

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
                            ramp_speed_c_min=program.ramp_speed_c_min,
                            config=config,
                            loop_time=loop_time,
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
                            and (
                                recovery_current_limited
                                or recovery_temperature >= setpoint - recovery_under_target_band
                            )
                        ):
                            pid_voltage = applied_voltage - config["max_voltage_step_up"]
                        pid_voltage = _clamp(pid_voltage, measurement_voltage_floor, config["max_voltage"])
                        previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)
                        invalid_measurements = 0
                        if resistance_confirmed and np.isfinite(measured_resistance):
                            previous_resistance = measured_resistance
                        if pid_voltage > applied_voltage + 1e-9:
                            recovery_action = "continuing upward"
                        elif pid_voltage < applied_voltage - 1e-9:
                            recovery_action = "gently backing off"
                        else:
                            recovery_action = "holding"
                        print(
                            f"Ignoring {'low-signal' if low_signal_state else 'transient'} invalid measurement. "
                            f"Measured Vsample={measured_voltage}, I={measured_current} while PSU was {applied_voltage:.4f} V. "
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
                        )
                        _emit_measurement(
                            emitter,
                            setpoint,
                            recovery_temperature,
                            measured_voltage,
                            measured_current,
                            applied_voltage,
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
                    pid_controller.reset(measurement=previous_temperature)
                    pid_voltage = _clamp(
                        applied_voltage - config.get("invalid_voltage_step_down", config["max_voltage_step_down"]),
                        measurement_voltage_floor,
                        config["max_voltage"],
                    )
                    previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)
                    print(
                        "Invalid measurement received. "
                        f"Measured Vsample={measured_voltage}, I={measured_current} while PSU was {applied_voltage:.4f} V. "
                        f"Reducing PSU to {pid_voltage:.4f} V (attempt {invalid_measurements})."
                    )
                    if invalid_measurements >= config["measurement_fail_limit"]:
                        raise ExperimentSafetyError("Too many invalid measurements in a row.")
                    _persist_measurement(
                        data_saver,
                        program.scheduled_target,
                        np.nan,
                        measured_voltage,
                        measured_current,
                        applied_voltage,
                    )
                    _emit_measurement(
                        emitter,
                        program.scheduled_target,
                        np.nan,
                        measured_voltage,
                        measured_current,
                        applied_voltage,
                    )
                    elapsed = time.time() - loop_started
                    if elapsed < loop_time:
                        time.sleep(loop_time - elapsed)
                    continue

                invalid_measurements = 0
                filtered_temperature = _temperature_filter(
                    temperature_history,
                    temperature,
                    config.get("measurement_filter_samples", 3),
                )
                rate_reference_temperature = (
                    filtered_temperature if reset_temperature_reference else previous_temperature
                )
                temp_rate_c_min = _temperature_rate_c_min(filtered_temperature, rate_reference_temperature, loop_time)
                setpoint, phase, finished = program.update(filtered_temperature, loop_time)

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
                    ramp_speed_c_min=program.ramp_speed_c_min,
                    config=config,
                    loop_time=loop_time,
                )
                previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)

                print(
                    f"Phase: {phase}, T: {filtered_temperature:.2f} C, Setpoint: {setpoint:.2f} C, "
                    f"Vsample: {measured_voltage:.6f} V, Current: {measured_current:.4e} A, "
                    f"PSU: {applied_voltage:.4f} -> {pid_voltage:.4f} V, "
                    f"Rate: {temp_rate_c_min if temp_rate_c_min is not None else 0.0:.2f} C/min"
                )
                _persist_measurement(
                    data_saver,
                    setpoint,
                    filtered_temperature,
                    measured_voltage,
                    measured_current,
                    applied_voltage,
                )
                _emit_measurement(
                    emitter,
                    setpoint,
                    filtered_temperature,
                    measured_voltage,
                    measured_current,
                    applied_voltage,
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


def measure_resistivity(dmm_v, dmm_i, siglent_module, temperature_interp, calibration=False, config=None):
    try:
        measured_voltage = float(siglent_module.read_DMM(dmm_v))
    except Exception as exc:
        print(f"An error occurred reading voltage DMM: {exc}")
        measured_voltage = np.nan

    try:
        measured_current = float(siglent_module.read_DMM(dmm_i))
    except Exception as exc:
        print(f"An error occurred reading current DMM: {exc}")
        measured_current = np.nan

    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current) or abs(measured_current) < 1e-12:
        return measured_voltage, measured_current, np.nan

    resistance = measured_voltage / measured_current
    if not np.isfinite(resistance) or resistance <= 0:
        print(f"Invalid resistance calculated from V={measured_voltage}, I={measured_current}")
        return measured_voltage, measured_current, np.nan
    if config is not None and not _resistance_in_curve_bounds(resistance, temperature_interp, config):
        print(
            f"Measured resistance {resistance:.6f} Ohm is outside the loaded R vs. T range; "
            "treating it as invalid."
        )
        return measured_voltage, measured_current, np.nan

    try:
        temperature = float(temperature_interp(resistance))
    except Exception as exc:
        print(f"An error occurred interpolating temperature: {exc}")
        temperature = np.nan

    if np.isfinite(temperature) and temperature < 0 and not calibration:
        print(f"Calculated temperature is {temperature}; clamping it to 0 C.")
        temperature = 0.0

    return measured_voltage, measured_current, temperature
