import time

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

from . import siglent
from . import tds_experiment


class CalibrationCancelled(RuntimeError):
    """Raised when the user stops calibration or controller tuning from the GUI."""


def _prepare_curve_interpolators(r_vs_t):
    curve = np.asarray(r_vs_t, dtype=float)
    if curve.shape[0] != 2 or curve.shape[1] < 2:
        raise ValueError("R vs. T data must have shape 2 x N with at least two points.")

    temperature_order = np.argsort(curve[1, :])
    temperature_curve = curve[:, temperature_order]
    _, unique_temperature_indices = np.unique(temperature_curve[1, :], return_index=True)
    temperature_curve = temperature_curve[:, np.sort(unique_temperature_indices)]

    resistance_order = np.argsort(curve[0, :])
    resistance_curve = curve[:, resistance_order]
    _, unique_resistance_indices = np.unique(resistance_curve[0, :], return_index=True)
    resistance_curve = resistance_curve[:, np.sort(unique_resistance_indices)]

    resistivity_interp = interp1d(
        temperature_curve[1, :],
        temperature_curve[0, :],
        kind="linear",
        fill_value="extrapolate",
    )
    temperature_interp = interp1d(
        resistance_curve[0, :],
        resistance_curve[1, :],
        kind="linear",
        fill_value="extrapolate",
    )
    return curve, resistivity_interp, temperature_interp


def _calculate_resistance(measured_voltage, measured_current):
    if not np.isfinite(measured_voltage) or not np.isfinite(measured_current):
        return np.nan
    if abs(measured_current) < 1e-12:
        return np.nan
    resistance = measured_voltage / measured_current
    if not np.isfinite(resistance) or resistance <= 0:
        return np.nan
    return float(resistance)


def _filter_room_temperature_samples(samples):
    if len(samples) < 3:
        return samples

    resistances = np.array([sample["resistance"] for sample in samples], dtype=float)
    median_resistance = float(np.median(resistances))
    deviations = np.abs(resistances - median_resistance)
    mad = float(np.median(deviations))
    resistance_window = max(6.0 * mad, 0.02 * median_resistance, 0.05)

    filtered = [
        sample
        for sample in samples
        if abs(sample["resistance"] - median_resistance) <= resistance_window
    ]
    if len(filtered) < 3:
        return samples
    return filtered


def _check_stop(emitter):
    if emitter is not None and getattr(emitter, "stopped", False):
        print("Calibration or controller tuning stop requested by user.")
        raise CalibrationCancelled("Stopped by user.")


def _emit_live_measurement(
    emitter,
    *,
    target_temperature,
    temperature,
    measured_voltage,
    measured_current,
    applied_voltage,
):
    if emitter is None or not hasattr(emitter, "live_measurement_signal"):
        return

    emitter.live_measurement_signal.emit(
        {
            "target_temperature": target_temperature,
            "temperature": temperature,
            "measured_voltage": measured_voltage,
            "measured_current": measured_current,
            "applied_voltage": applied_voltage,
        }
    )


def _sleep_with_stop(duration_s, emitter):
    remaining = max(float(duration_s), 0.0)
    while remaining > 0:
        _check_stop(emitter)
        sleep_chunk = min(0.1, remaining)
        time.sleep(sleep_chunk)
        remaining -= sleep_chunk


def _temperature_is_in_window(temperature, lower_bound=None, upper_bound=None):
    if not np.isfinite(temperature):
        return False
    if lower_bound is not None and temperature < lower_bound:
        return False
    if upper_bound is not None and temperature > upper_bound:
        return False
    return True


def _current_series_is_stable(currents, minimum_current):
    if not currents:
        return False

    current_array = np.asarray(currents, dtype=float)
    if not np.all(np.isfinite(current_array)):
        return False
    if np.any(current_array <= minimum_current):
        return False

    median_current = float(np.median(current_array))
    allowed_spread = max(0.15 * median_current, 5.0 * minimum_current)
    return float(np.max(np.abs(current_array - median_current))) <= allowed_spread


def _find_stable_current_voltage(
    *,
    dmm_v,
    dmm_i,
    power_supply,
    temperature_interp,
    config,
    start_voltage,
    max_voltage,
    step_voltage,
    settle_time_s,
    stable_samples,
    minimum_current,
    emitter,
    label,
    temperature_lower_bound=None,
    temperature_upper_bound=None,
    display_target_temperature=None,
    allow_current_only_fallback=False,
):
    sample_interval_s = max(0.5, 1.0 / config["experiment_frequency"])
    voltage = max(start_voltage, config["min_voltage"], 0.005)
    search_upper_bound = min(max_voltage, config["max_voltage"])
    voltage_step = max(step_voltage, config["minimum_voltage_change"])

    while voltage <= search_upper_bound + 1e-12:
        _check_stop(emitter)
        siglent.set_voltage(power_supply, voltage=voltage)
        print(f"{label}: trying {voltage:.4f} V")
        _sleep_with_stop(settle_time_s, emitter)

        samples = []
        consecutive_invalid_samples = 0
        invalid_advance_count = max(int(config.get("stable_current_invalid_advance_count", 5)), 1)
        announced_current_only_fallback = False
        attempts = 0
        max_attempts = max(int(stable_samples) * 3, int(stable_samples) + config["measurement_fail_limit"] * 3)
        while len(samples) < int(stable_samples) and attempts < max_attempts:
            _check_stop(emitter)
            attempts += 1
            measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                calibration=True,
                config=config,
            )
            resistance = _calculate_resistance(measured_voltage, measured_current)
            _emit_live_measurement(
                emitter,
                target_temperature=display_target_temperature,
                temperature=temperature,
                measured_voltage=measured_voltage,
                measured_current=measured_current,
                applied_voltage=voltage,
            )
            print(
                f"{label} sample: T={temperature}, V={measured_voltage}, "
                f"I={measured_current}, R={resistance}"
            )

            if abs(measured_current) > config["max_current"]:
                raise tds_experiment.ExperimentSafetyError(
                    f"{label}: measured current {measured_current:.4e} A exceeded max_current."
                )

            stable_current_sample = (
                np.isfinite(measured_voltage)
                and np.isfinite(measured_current)
                and measured_current > minimum_current
                and np.isfinite(resistance)
            )
            valid_sample = (
                tds_experiment._is_valid_measurement(measured_voltage, measured_current, temperature, config)
                and stable_current_sample
                and _temperature_is_in_window(
                    temperature,
                    lower_bound=temperature_lower_bound,
                    upper_bound=temperature_upper_bound,
                )
            )
            if (
                not valid_sample
                and allow_current_only_fallback
                and stable_current_sample
                and not np.isfinite(temperature)
            ):
                valid_sample = True
                if not announced_current_only_fallback:
                    print(
                        f"{label}: resistance is outside the loaded R vs. T range, "
                        "so calibration will use stable current and room-temperature scaling."
                    )
                    announced_current_only_fallback = True

            if valid_sample:
                consecutive_invalid_samples = 0
                samples.append(
                    {
                        "voltage": float(measured_voltage),
                        "current": float(measured_current),
                        "temperature": float(temperature),
                        "resistance": float(resistance),
                    }
                )
            else:
                consecutive_invalid_samples += 1
                if samples:
                    print(f"{label}: unstable sample detected, restarting stability check at {voltage:.4f} V.")
                samples = []
                if consecutive_invalid_samples >= invalid_advance_count:
                    print(
                        f"{label}: received {consecutive_invalid_samples} invalid samples at {voltage:.4f} V, "
                        "increasing search voltage."
                    )
                    break

            _sleep_with_stop(sample_interval_s, emitter)

        currents = [sample["current"] for sample in samples]
        if len(samples) >= int(stable_samples) and _current_series_is_stable(currents, minimum_current):
            print(f"{label}: stable positive current found at {voltage:.4f} V")
            return float(voltage), samples

        voltage += voltage_step

    raise ValueError(
        f"{label}: could not find a stable positive current between {start_voltage:.4f} V "
        f"and {search_upper_bound:.4f} V."
    )


def calibrate_temperature_curve(r_vs_t, room_temp, config=None, emitter=None):
    """
    Shift the resistivity curve so the measured room-temperature resistance lines
    up with the loaded calibration table.
    """
    config = tds_experiment.build_control_config(config or {})
    curve, resistivity_interp, temperature_interp = _prepare_curve_interpolators(r_vs_t)

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
        _sleep_with_stop(0.04, emitter)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        _sleep_with_stop(1.0, emitter)

        calibration_voltage, _ = _find_stable_current_voltage(
            dmm_v=dmm_v,
            dmm_i=dmm_i,
            power_supply=power_supply,
            temperature_interp=temperature_interp,
            config=config,
            start_voltage=config["t0_voltage_search_start"],
            max_voltage=max(config["t0_calibration_voltage"], config["t0_voltage_search_start"]),
            step_voltage=config["t0_voltage_step"],
            settle_time_s=config["t0_settle_time_s"],
            stable_samples=config["t0_stable_current_samples"],
            minimum_current=config["t0_stable_current_a"],
            emitter=emitter,
            label="T0 search",
            temperature_lower_bound=room_temp - config["t0_max_temp_error_c"],
            temperature_upper_bound=room_temp + config["t0_max_temp_error_c"],
            display_target_temperature=room_temp,
            allow_current_only_fallback=True,
        )
        print(f"Using T0 calibration voltage: {calibration_voltage:.4f} V")

        sample_interval_s = max(0.5, 1.0 / config["experiment_frequency"])

        accepted_samples = []
        warmup_remaining = max(int(config["t0_warmup_samples"]), 0)
        target_samples = max(int(config["t0_calibration_samples"]), 3)
        attempts = 0
        room_temp_scaling_fallback_used = False
        max_attempts = max(
            12,
            target_samples + warmup_remaining + config["measurement_fail_limit"] * 6,
        )
        while len(accepted_samples) < target_samples and attempts < max_attempts:
            _check_stop(emitter)
            attempts += 1
            measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                calibration=True,
                config=config,
            )
            _emit_live_measurement(
                emitter,
                target_temperature=room_temp,
                temperature=temperature,
                measured_voltage=measured_voltage,
                measured_current=measured_current,
                applied_voltage=calibration_voltage,
            )
            print(
                f"Room-temperature calibration sample: T={temperature}, V={measured_voltage}, I={measured_current}"
            )
            if abs(measured_current) > config["max_current"]:
                raise tds_experiment.ExperimentSafetyError(
                    f"Measured current {measured_current:.4e} A exceeded max_current during T0 calibration."
                )

            if not np.isfinite(measured_voltage) or not np.isfinite(measured_current):
                print("Rejected room-temperature calibration sample: invalid measurement.")
                _sleep_with_stop(sample_interval_s, emitter)
                continue

            resistance = _calculate_resistance(measured_voltage, measured_current)
            if not np.isfinite(resistance):
                print("Rejected room-temperature calibration sample: invalid resistance.")
                _sleep_with_stop(sample_interval_s, emitter)
                continue
            sample_temperature = float(temperature)
            if np.isfinite(temperature):
                if abs(temperature - room_temp) > config["t0_max_temp_error_c"]:
                    print(
                        "Rejected room-temperature calibration sample: "
                        f"|T-room_temp|={abs(temperature - room_temp):.2f} C exceeds "
                        f"{config['t0_max_temp_error_c']:.2f} C."
                    )
                    _sleep_with_stop(sample_interval_s, emitter)
                    continue
            else:
                if abs(measured_current) <= config["t0_stable_current_a"]:
                    print("Rejected room-temperature calibration sample: current is too small.")
                    _sleep_with_stop(sample_interval_s, emitter)
                    continue
                sample_temperature = float(room_temp)
                if not room_temp_scaling_fallback_used:
                    print(
                        "Room-temperature calibration is using resistance-only scaling because the measured "
                        "wire resistance is outside the loaded R vs. T range."
                    )
                    room_temp_scaling_fallback_used = True

            if warmup_remaining > 0:
                print(
                    "Discarding room-temperature calibration warmup sample: "
                    f"R={resistance:.4f} Ohm, T={sample_temperature:.2f} C"
                )
                warmup_remaining -= 1
                _sleep_with_stop(sample_interval_s, emitter)
                continue

            accepted_samples.append(
                {
                    "voltage": float(measured_voltage),
                    "current": float(measured_current),
                    "temperature": sample_temperature,
                    "resistance": resistance,
                }
            )
            _sleep_with_stop(sample_interval_s, emitter)

        if len(accepted_samples) < 3:
            raise ValueError("Could not collect enough stable room-temperature calibration samples.")

        filtered_samples = _filter_room_temperature_samples(accepted_samples)
        if len(filtered_samples) != len(accepted_samples):
            print(
                f"Using {len(filtered_samples)} of {len(accepted_samples)} room-temperature samples "
                "after resistance outlier filtering."
            )

        measured_current = float(
            np.median(np.array([sample["current"] for sample in filtered_samples], dtype=float))
        )
        measured_voltage = float(
            np.median(np.array([sample["voltage"] for sample in filtered_samples], dtype=float))
        )
        temperature = float(
            np.median(np.array([sample["temperature"] for sample in filtered_samples], dtype=float))
        )
        measured_resistivity = float(
            np.median(np.array([sample["resistance"] for sample in filtered_samples], dtype=float))
        )
        print(f"Final room-temperature sample: T={temperature}, V={measured_voltage}, I={measured_current}")

        if not np.isfinite(measured_resistivity) or measured_resistivity <= 0:
            print(f"Measured resistivity {measured_resistivity:.4f} Ohm is invalid.")
            return None

        resistivity_room_temp = float(resistivity_interp(room_temp))
        scale = measured_resistivity / resistivity_room_temp
        print(f"Measured resistivity: {measured_resistivity:.4f} Ohm")
        print(f"Reference resistivity: {resistivity_room_temp:.4f} Ohm")
        print(f"Calibration scale: {scale:.4f}")
        if room_temp_scaling_fallback_used:
            print(
                "Applied room-temperature scaling fallback so a different wire geometry can reuse the "
                "loaded material curve."
            )

        calibrated = curve.copy()
        calibrated[0, :] *= scale
        return calibrated

    finally:
        tds_experiment._shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)


def _estimate_pid_from_step(response, base_temperature, step_voltage, loop_time, min_temp_rise, controller_mode="PI"):
    if not response:
        raise ValueError("Controller tuning did not collect any valid samples.")

    times = np.array([sample["elapsed_s"] for sample in response], dtype=float)
    temperatures = np.array([sample["temperature"] for sample in response], dtype=float)
    temperature_rise = temperatures - base_temperature
    peak_rise = float(np.max(temperature_rise))

    if peak_rise < min_temp_rise:
        raise ValueError(
            "Controller tuning did not produce enough temperature change. Increase tuning_search_max_voltage "
            "or tuning_voltage_step carefully."
        )

    # Small safe tuning steps can still produce a usable response, so keep the
    # dead-time threshold low enough to identify them instead of demanding a
    # large temperature excursion.
    threshold = max(0.1 * peak_rise, 0.15)
    threshold_indices = np.where(temperature_rise >= threshold)[0]
    dead_time_s = float(times[threshold_indices[0]]) if threshold_indices.size else 0.0

    target_63 = 0.632 * peak_rise
    tau_indices = np.where(temperature_rise >= target_63)[0]
    if tau_indices.size:
        time_constant_s = max(float(times[tau_indices[0]]) - dead_time_s, loop_time)
    else:
        time_constant_s = max(float(times[-1]) - dead_time_s, loop_time)

    process_gain = peak_rise / max(step_voltage, 1e-6)
    lambda_time_s = max(3.0 * dead_time_s, time_constant_s, 30.0)

    kp = time_constant_s / (process_gain * (lambda_time_s + dead_time_s))
    ti = max(time_constant_s + dead_time_s / 2.0, loop_time)
    ki = kp / ti

    controller_mode = str(controller_mode).strip().upper()
    if controller_mode == "PID":
        derivative_time_s = 0.0
        if dead_time_s > 0.0:
            derivative_time_s = (time_constant_s * dead_time_s) / max(
                2.0 * time_constant_s + dead_time_s,
                loop_time,
            )
        kd = float(np.clip(kp * derivative_time_s, 0.0, 0.02))
    else:
        kd = 0.0

    return {
        "Kp": float(np.clip(kp, 0.001, 0.05)),
        "Ki": float(np.clip(ki, 1e-5, 0.01)),
        "Kd": kd,
        "base_temperature": base_temperature,
        "step_voltage": step_voltage,
        "peak_rise_c": peak_rise,
        "dead_time_s": dead_time_s,
        "time_constant_s": time_constant_s,
    }


def _collect_pid_baseline(
    *,
    dmm_v,
    dmm_i,
    temperature_interp,
    config,
    emitter,
    baseline_voltage,
    target_temperature,
    temperature_lower_bound,
    temperature_upper_bound,
    loop_time,
    initial_samples=None,
):
    baseline_temperatures = []
    for sample in initial_samples or []:
        if _temperature_is_in_window(
            sample["temperature"],
            lower_bound=temperature_lower_bound,
            upper_bound=temperature_upper_bound,
        ):
            baseline_temperatures.append(float(sample["temperature"]))

    sample_interval_s = max(loop_time, 0.5)
    invalid_measurements = 0
    while len(baseline_temperatures) < int(config["tuning_baseline_samples"]):
        _check_stop(emitter)
        measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
            dmm_v,
            dmm_i,
            siglent,
            temperature_interp,
            calibration=True,
            config=config,
        )
        resistance = _calculate_resistance(measured_voltage, measured_current)
        _emit_live_measurement(
            emitter,
            target_temperature=target_temperature,
            temperature=temperature,
            measured_voltage=measured_voltage,
            measured_current=measured_current,
            applied_voltage=baseline_voltage,
        )
        valid_baseline = (
            tds_experiment._is_valid_measurement(measured_voltage, measured_current, temperature, config)
            and measured_current > config["tuning_stable_current_a"]
            and _temperature_is_in_window(
                temperature,
                lower_bound=temperature_lower_bound,
                upper_bound=temperature_upper_bound,
            )
        )
        print(
            f"PID baseline sample: T={temperature}, R={resistance}, "
            f"V={measured_voltage}, I={measured_current}, Vps={baseline_voltage:.4f}"
        )
        if valid_baseline:
            baseline_temperatures.append(float(temperature))
            invalid_measurements = 0
        else:
            invalid_measurements += 1
            if invalid_measurements >= config["measurement_fail_limit"]:
                raise ValueError("Could not get a stable baseline temperature for controller tuning.")
        _sleep_with_stop(sample_interval_s, emitter)

    return float(np.median(np.array(baseline_temperatures, dtype=float)))


def _run_pid_tuning_attempt(
    *,
    dmm_v,
    dmm_i,
    power_supply,
    temperature_interp,
    config,
    emitter,
    baseline_voltage,
    response_voltage,
    base_temperature,
    desired_rise,
    required_rise,
    smoothed_required_rise,
    safe_temperature_limit,
    temperature_lower_bound,
    loop_time,
):
    siglent.set_voltage(power_supply, voltage=response_voltage)
    response = []
    invalid_measurements = 0
    start_time = time.time()
    best_smoothed_rise_so_far = float("-inf")
    last_growth_time_s = 0.0

    while time.time() - start_time < config["tuning_max_duration_s"]:
        _check_stop(emitter)
        loop_started = time.time()
        measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
            dmm_v,
            dmm_i,
            siglent,
            temperature_interp,
            calibration=True,
            config=config,
        )
        resistance = _calculate_resistance(measured_voltage, measured_current)
        _emit_live_measurement(
            emitter,
            target_temperature=base_temperature + desired_rise,
            temperature=temperature,
            measured_voltage=measured_voltage,
            measured_current=measured_current,
            applied_voltage=response_voltage,
        )

        if np.isfinite(temperature) and temperature > safe_temperature_limit:
            print(
                f"Controller tuning stopped at the safety temperature limit: "
                f"T={temperature:.2f} C, limit={safe_temperature_limit:.2f} C"
            )
            break

        valid_response = (
            tds_experiment._is_valid_measurement(measured_voltage, measured_current, temperature, config)
            and measured_current > config["tuning_stable_current_a"]
            and _temperature_is_in_window(
                temperature,
                lower_bound=temperature_lower_bound,
                upper_bound=safe_temperature_limit,
            )
        )
        if not valid_response:
            invalid_measurements += 1
            if invalid_measurements >= config["measurement_fail_limit"]:
                raise ValueError("Too many invalid measurements during controller tuning.")
            elapsed = time.time() - loop_started
            if elapsed < loop_time:
                _sleep_with_stop(loop_time - elapsed, emitter)
            continue

        invalid_measurements = 0
        if abs(measured_current) > config["max_current"]:
            raise tds_experiment.ExperimentSafetyError(
                f"Measured current {measured_current:.4e} A exceeded max_current during tuning."
            )

        elapsed_s = time.time() - start_time
        response.append(
            {
                "elapsed_s": elapsed_s,
                "temperature": temperature,
                "current": measured_current,
                "measured_voltage": measured_voltage,
                "resistance": resistance,
            }
        )
        peak_rise_so_far = max(sample["temperature"] for sample in response) - base_temperature
        recent_temperatures = np.array(
            [sample["temperature"] for sample in response[-min(5, len(response)):]],
            dtype=float,
        )
        smoothed_rise_so_far = float(np.median(recent_temperatures) - base_temperature)
        print(
            f"Tuning sample: t={elapsed_s:.1f} s, T={temperature:.2f} C, "
            f"R={resistance:.4f} Ohm, V={measured_voltage:.6f} V, "
            f"I={measured_current:.4e} A, Vps={response_voltage:.4f} V"
        )

        if smoothed_rise_so_far > best_smoothed_rise_so_far + config["tuning_plateau_growth_c"]:
            best_smoothed_rise_so_far = smoothed_rise_so_far
            last_growth_time_s = elapsed_s

        if peak_rise_so_far >= required_rise and smoothed_rise_so_far >= smoothed_required_rise:
            return {
                "status": "usable_response",
                "response": response,
                "peak_rise_c": peak_rise_so_far,
                "smoothed_rise_c": smoothed_rise_so_far,
                "elapsed_s": elapsed_s,
            }

        if temperature >= base_temperature + desired_rise:
            return {
                "status": "target_reached",
                "response": response,
                "peak_rise_c": peak_rise_so_far,
                "smoothed_rise_c": smoothed_rise_so_far,
                "elapsed_s": elapsed_s,
            }

        if (
            elapsed_s >= config["tuning_no_response_timeout_s"]
            and smoothed_rise_so_far < config["tuning_min_observable_rise_c"]
        ):
            return {
                "status": "no_response",
                "response": response,
                "peak_rise_c": peak_rise_so_far,
                "smoothed_rise_c": smoothed_rise_so_far,
                "elapsed_s": elapsed_s,
            }

        if (
            elapsed_s >= config["tuning_plateau_timeout_s"]
            and peak_rise_so_far < required_rise
            and elapsed_s - last_growth_time_s >= config["tuning_plateau_idle_timeout_s"]
        ):
            return {
                "status": "plateau",
                "response": response,
                "peak_rise_c": peak_rise_so_far,
                "smoothed_rise_c": smoothed_rise_so_far,
                "elapsed_s": elapsed_s,
            }

        elapsed = time.time() - loop_started
        if elapsed < loop_time:
            _sleep_with_stop(loop_time - elapsed, emitter)

    peak_rise_c = 0.0
    smoothed_rise_c = 0.0
    elapsed_s = time.time() - start_time
    if response:
        peak_rise_c = max(sample["temperature"] for sample in response) - base_temperature
        recent_temperatures = np.array(
            [sample["temperature"] for sample in response[-min(5, len(response)):]],
            dtype=float,
        )
        smoothed_rise_c = float(np.median(recent_temperatures) - base_temperature)

    return {
        "status": "duration_complete",
        "response": response,
        "peak_rise_c": peak_rise_c,
        "smoothed_rise_c": smoothed_rise_c,
        "elapsed_s": elapsed_s,
    }


def tune_pid(experiment_params, config, r_vs_t, base_temperature_hint=None, emitter=None):
    """
    Tune conservative gains from a small guarded voltage step on the real setup.
    """
    config = tds_experiment.build_control_config(config)
    controller_mode = tds_experiment.get_controller_mode(config)
    loop_time = 1.0 / config["experiment_frequency"]
    _, _, temperature_interp = _prepare_curve_interpolators(r_vs_t)

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
        _sleep_with_stop(0.04, emitter)
        siglent.set_voltage(power_supply, voltage=0.0)
        _sleep_with_stop(1.0, emitter)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])

        stable_temperature_window = config["tuning_temperature_window_c"]
        temperature_lower_bound = None
        temperature_upper_bound = None
        if base_temperature_hint is not None:
            temperature_lower_bound = base_temperature_hint - stable_temperature_window
            temperature_upper_bound = base_temperature_hint + stable_temperature_window

        step_voltage, stable_samples = _find_stable_current_voltage(
            dmm_v=dmm_v,
            dmm_i=dmm_i,
            power_supply=power_supply,
            temperature_interp=temperature_interp,
            config=config,
            start_voltage=config["tuning_start_voltage"],
            max_voltage=max(config["tuning_start_voltage"], config["tuning_search_max_voltage"]),
            step_voltage=config["tuning_voltage_step"],
            settle_time_s=config["tuning_settle_time_s"],
            stable_samples=config["tuning_stable_current_samples"],
            minimum_current=config["tuning_stable_current_a"],
            emitter=emitter,
            label=f"{controller_mode} tuning search",
            temperature_lower_bound=temperature_lower_bound,
            temperature_upper_bound=temperature_upper_bound,
            display_target_temperature=base_temperature_hint,
        )
        baseline_voltage = step_voltage
        print(f"Using {controller_mode} baseline voltage: {baseline_voltage:.4f} V")

        response_step = max(config["tuning_response_voltage_step"], config["minimum_voltage_change"])
        max_response_voltage = min(config["tuning_search_max_voltage"], config["max_voltage"])
        candidate_voltage = max(
            baseline_voltage + response_step,
            baseline_voltage + config["minimum_voltage_change"],
        )
        if candidate_voltage > max_response_voltage + 1e-12:
            raise ValueError(
                "Controller tuning could not create a voltage step above the stable-current baseline. "
                "Increase tuning_search_max_voltage carefully."
            )

        seeded_baseline_samples = stable_samples
        last_failure = None
        attempt_number = 0
        while candidate_voltage <= max_response_voltage + 1e-12:
            attempt_number += 1
            print(
                f"{controller_mode} tuning attempt {attempt_number}: baseline={baseline_voltage:.4f} V, "
                f"response={candidate_voltage:.4f} V"
            )
            siglent.set_voltage(power_supply, voltage=baseline_voltage)
            _sleep_with_stop(config["tuning_between_attempts_s"], emitter)

            base_temperature = _collect_pid_baseline(
                dmm_v=dmm_v,
                dmm_i=dmm_i,
                temperature_interp=temperature_interp,
                config=config,
                emitter=emitter,
                baseline_voltage=baseline_voltage,
                target_temperature=base_temperature_hint,
                temperature_lower_bound=temperature_lower_bound,
                temperature_upper_bound=temperature_upper_bound,
                loop_time=loop_time,
                initial_samples=seeded_baseline_samples,
            )
            seeded_baseline_samples = None

            available_rise = max(0.0, experiment_params["target_T"] - base_temperature)
            desired_rise = min(config["tuning_target_rise_c"], available_rise)
            if desired_rise < config["tuning_min_temperature_rise_c"] and available_rise > 0:
                desired_rise = available_rise
            if desired_rise <= 0:
                raise ValueError(
                    "Target temperature is not above the current temperature, so controller tuning cannot proceed."
                )

            required_rise = min(config["tuning_min_temperature_rise_c"], desired_rise)
            smoothed_required_rise = max(
                config["tuning_min_observable_rise_c"],
                0.65 * required_rise,
            )
            safe_temperature_limit = min(
                experiment_params["target_T"],
                base_temperature + desired_rise + config["temperature_tolerance_c"],
            )
            attempt = _run_pid_tuning_attempt(
                dmm_v=dmm_v,
                dmm_i=dmm_i,
                power_supply=power_supply,
                temperature_interp=temperature_interp,
                config=config,
                emitter=emitter,
                baseline_voltage=baseline_voltage,
                response_voltage=candidate_voltage,
                base_temperature=base_temperature,
                desired_rise=desired_rise,
                required_rise=required_rise,
                smoothed_required_rise=smoothed_required_rise,
                safe_temperature_limit=safe_temperature_limit,
                temperature_lower_bound=temperature_lower_bound,
                loop_time=loop_time,
            )

            siglent.set_voltage(power_supply, voltage=baseline_voltage)
            _sleep_with_stop(config["tuning_between_attempts_s"], emitter)

            if (
                attempt["peak_rise_c"] >= required_rise
                and attempt["smoothed_rise_c"] >= smoothed_required_rise
                and attempt["response"]
            ):
                tuned = _estimate_pid_from_step(
                    response=attempt["response"],
                    base_temperature=base_temperature,
                    step_voltage=candidate_voltage - baseline_voltage,
                    loop_time=loop_time,
                    min_temp_rise=required_rise,
                    controller_mode=controller_mode,
                )
                tuned["baseline_voltage"] = baseline_voltage
                tuned["step_voltage"] = candidate_voltage
                tuned["step_delta_voltage"] = candidate_voltage - baseline_voltage
                print(
                    f"Tuned {controller_mode} parameters: Kp={tuned['Kp']:.6f}, Ki={tuned['Ki']:.6f}, "
                    f"Kd={tuned['Kd']:.6f}, baseline={baseline_voltage:.4f} V, "
                    f"response={candidate_voltage:.4f} V, delta={tuned['step_delta_voltage']:.4f} V, "
                    f"peak rise={tuned['peak_rise_c']:.2f} C"
                )
                return tuned

            last_failure = (
                f"Attempt at {candidate_voltage:.4f} V ended with status {attempt['status']} and produced "
                f"{attempt['smoothed_rise_c']:.2f} C smoothed rise "
                f"({attempt['peak_rise_c']:.2f} C peak)."
            )
            print(f"{controller_mode} tuning attempt did not produce enough response. {last_failure}")
            candidate_voltage += response_step

        failure_message = (
            f"{controller_mode} tuning could not find a usable step response up to {max_response_voltage:.4f} V. "
            f"{last_failure or ''}"
        ).strip()
        raise ValueError(failure_message)

    finally:
        tds_experiment._shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)
