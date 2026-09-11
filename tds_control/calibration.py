import time

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

from . import siglent
from . import tds_experiment


class CalibrationCancelled(RuntimeError):
    """Raised when the user stops calibration or controller tuning from the GUI."""


def _prepare_curve_interpolators(r_vs_t, config=None):
    curve = np.asarray(r_vs_t, dtype=float)
    config = tds_experiment.build_control_config(config or {})
    temperature_curve, temperature_bounds, source_temperature_bounds = (
        tds_experiment._extend_curve_for_configured_extrapolation(curve, config)
    )

    resistivity_interp = interp1d(
        temperature_curve[1, :],
        temperature_curve[0, :],
        kind="linear",
        fill_value="extrapolate",
    )
    temperature_interp = tds_experiment._build_temperature_interpolator_from_curve(
        temperature_curve,
        temperature_bounds,
        source_temperature_bounds,
    )
    return curve, resistivity_interp, temperature_interp


def _calibrated_temperature_spread(resistances, scale, reference_temperature_interp):
    """Return T spread after applying R_cal(T) = scale * R_ref(T)."""
    numeric_scale = float(scale)
    if not np.isfinite(numeric_scale) or numeric_scale <= 0:
        return np.nan

    equivalent_temperatures = np.array(
        [float(reference_temperature_interp(float(value) / numeric_scale)) for value in resistances],
        dtype=float,
    )
    if equivalent_temperatures.size == 0 or not np.all(np.isfinite(equivalent_temperatures)):
        return np.nan
    return float(np.ptp(equivalent_temperatures))


def _scale_and_anchor_curve(curve, scale, anchor_temperature, anchor_resistance):
    """Scale an R-vs-T curve and make the measured T0 pair an explicit curve point."""
    calibrated = np.asarray(curve, dtype=float).copy()
    numeric_scale = float(scale)
    temperature = float(anchor_temperature)
    resistance = float(anchor_resistance)
    if calibrated.ndim != 2 or calibrated.shape[0] != 2 or calibrated.shape[1] < 2:
        raise ValueError("R vs. T data must have shape (2, N) with at least two points.")
    if not all(np.isfinite(value) for value in (numeric_scale, temperature, resistance)):
        raise ValueError("T0 calibration scale and anchor values must be finite.")
    if numeric_scale <= 0 or resistance <= 0:
        raise ValueError("T0 calibration scale and resistance must be positive.")

    calibrated[0, :] *= numeric_scale
    matching_temperature = np.isclose(calibrated[1, :], temperature, rtol=0.0, atol=1e-9)
    calibrated = calibrated[:, ~matching_temperature]
    calibrated = np.hstack(
        (
            calibrated,
            np.array([[resistance], [temperature]], dtype=float),
        )
    )
    return calibrated[:, np.argsort(calibrated[1, :])]


def _filter_room_temperature_samples(samples, config=None):
    if len(samples) < 3:
        return samples

    config = config or {}
    resistances = np.array([sample["resistance"] for sample in samples], dtype=float)
    inlier_mask = tds_experiment._robust_resistance_inlier_mask(resistances, config)
    filtered = [sample for sample, is_inlier in zip(samples, inlier_mask) if is_inlier]
    if len(filtered) < 3:
        return samples
    return filtered


def _check_stop(emitter):
    if emitter is not None and getattr(emitter, "stopped", False):
        print("Calibration or controller tuning stop requested by user.")
        raise CalibrationCancelled("Stopped by user.")


def _calibration_sample_interval_s(config):
    """Pause between calibration samples.

    The duty-cycled resistivity modes already spend a whole heat-and-measure
    cycle inside each reading, so an extra pause only slows calibration down.
    """
    if tds_experiment.resistivity_mode_needs_power_supply(config):
        return 0.0
    return max(0.5, 1.0 / config["experiment_frequency"])


def _emit_live_measurement(
    emitter,
    *,
    target_temperature,
    temperature,
    measured_voltage,
    measured_current,
    applied_current,
    resistance=None,
):
    if emitter is None or not hasattr(emitter, "live_measurement_signal"):
        return

    emitter.live_measurement_signal.emit(
        {
            "target_temperature": target_temperature,
            "temperature": temperature,
            "measured_voltage": measured_voltage,
            "measured_current": measured_current,
            "applied_current": applied_current,
            "resistance": resistance,
        }
    )


def _emit_calibration_warning(emitter, message):
    print(f"WARNING: {message}")
    if emitter is None:
        return
    emitter.last_calibration_warning = message
    warning_signal = getattr(emitter, "calibration_warning_signal", None)
    if warning_signal is not None:
        warning_signal.emit(message)


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


def _resistance_series_is_stable(resistances, config):
    if not resistances:
        return False
    resistance_array = np.asarray(resistances, dtype=float)
    if not np.all(np.isfinite(resistance_array)) or np.any(resistance_array <= 0):
        return False
    median_resistance = float(np.median(resistance_array))
    allowed_deviation = max(
        float(config.get("stable_resistance_spread_ohm", 0.03)),
        abs(median_resistance) * float(config.get("stable_resistance_spread_ratio", 0.005)),
    )
    return float(np.max(np.abs(resistance_array - median_resistance))) <= allowed_deviation


def _find_stable_current_setpoint(
    *,
    dmm_v,
    dmm_i,
    power_supply,
    temperature_interp,
    config,
    start_current,
    max_current,
    step_current,
    settle_time_s,
    stable_samples,
    minimum_current,
    emitter,
    label,
    temperature_lower_bound=None,
    temperature_upper_bound=None,
    display_target_temperature=None,
    allow_current_only_fallback=False,
    stop_on_high_temperature=False,
):
    sample_interval_s = _calibration_sample_interval_s(config)
    setpoint_current = max(start_current, config["min_current"], 0.005)
    search_upper_bound = min(max_current, config["max_current"])
    current_step = max(step_current, config["minimum_current_change"])

    while setpoint_current <= search_upper_bound + 1e-12:
        _check_stop(emitter)
        siglent.set_current(power_supply, current=setpoint_current)
        print(f"{label}: trying {setpoint_current:.4f} A")
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
            measured_voltage, measured_current, temperature, resistance = tds_experiment.measure_resistivity(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                calibration=True,
                config=config,
                power_supply=power_supply,
            )
            _emit_live_measurement(
                emitter,
                target_temperature=display_target_temperature,
                temperature=temperature,
                measured_voltage=measured_voltage,
                measured_current=measured_current,
                applied_current=setpoint_current,
                resistance=resistance,
            )
            print(
                f"{label} sample: T={temperature}, V={measured_voltage}, "
                f"I={measured_current}, R={resistance}"
            )

            if (
                stop_on_high_temperature
                and temperature_upper_bound is not None
                and np.isfinite(temperature)
                and temperature > temperature_upper_bound
            ):
                raise tds_experiment.ExperimentSafetyError(
                    f"{label}: inferred temperature {temperature:.2f} C exceeds the allowed "
                    f"{temperature_upper_bound:.2f} C baseline window at {setpoint_current:.4f} A. "
                    "Lower tuning_start_current before tuning."
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
                        "setpoint_current": float(measured_voltage),
                        "current": float(measured_current),
                        "temperature": float(temperature),
                        "resistance": float(resistance),
                    }
                )
            else:
                consecutive_invalid_samples += 1
                if samples:
                    print(f"{label}: unstable sample detected, restarting stability check at {setpoint_current:.4f} A.")
                samples = []
                if consecutive_invalid_samples >= invalid_advance_count:
                    print(
                        f"{label}: received {consecutive_invalid_samples} invalid samples at {setpoint_current:.4f} A, "
                        "increasing search setpoint_current."
                    )
                    break

            _sleep_with_stop(sample_interval_s, emitter)

        currents = [sample["current"] for sample in samples]
        resistances = [sample["resistance"] for sample in samples]
        if (
            len(samples) >= int(stable_samples)
            and _current_series_is_stable(currents, minimum_current)
            and _resistance_series_is_stable(resistances, config)
        ):
            print(f"{label}: stable current and resistance found at {setpoint_current:.4f} A")
            return float(setpoint_current), samples
        if len(samples) >= int(stable_samples) and _current_series_is_stable(currents, minimum_current):
            print(
                f"{label}: current was stable at {setpoint_current:.4f} A, but resistance was too noisy; "
                "increasing by a cautious low-setpoint_current step."
            )

        next_current = tds_experiment._limit_current_slew(
            setpoint_current + current_step,
            setpoint_current,
            max(float(config["min_current"]), 0.005),
            search_upper_bound,
            config,
        )
        if next_current <= setpoint_current + 1e-12:
            break
        setpoint_current = next_current

    raise ValueError(
        f"{label}: could not find a stable positive current between {start_current:.4f} A "
        f"and {search_upper_bound:.4f} V."
    )


def calibrate_temperature_curve(r_vs_t, room_temp, config=None, emitter=None):
    """
    Shift the resistivity curve so the measured room-temperature resistance lines
    up with the loaded calibration table.
    """
    config = tds_experiment.build_control_config(config or {})
    curve, resistivity_interp, temperature_interp = _prepare_curve_interpolators(r_vs_t, config=config)

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

        tds_experiment.prepare_power_supply_output(power_supply, config)
        siglent.configure_dc_range_from_config(dmm_v, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm_i, "CURR", config)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        _sleep_with_stop(1.0, emitter)

        calibration_current, _ = _find_stable_current_setpoint(
            dmm_v=dmm_v,
            dmm_i=dmm_i,
            power_supply=power_supply,
            temperature_interp=temperature_interp,
            config=config,
            start_current=config["t0_current_search_start"],
            max_current=max(config["t0_calibration_current"], config["t0_current_search_start"]),
            step_current=config["t0_current_step"],
            settle_time_s=config["t0_settle_time_s"],
            stable_samples=config["t0_stable_current_samples"],
            minimum_current=config["t0_stable_current_a"],
            emitter=emitter,
            label="T0 search",
            display_target_temperature=room_temp,
            allow_current_only_fallback=True,
        )
        print(f"Using T0 calibration voltage: {calibration_current:.4f} A")

        sample_interval_s = _calibration_sample_interval_s(config)

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
            measured_voltage, measured_current, temperature, resistance = tds_experiment.measure_resistivity(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                calibration=True,
                config=config,
                power_supply=power_supply,
            )
            _emit_live_measurement(
                emitter,
                target_temperature=room_temp,
                temperature=temperature,
                measured_voltage=measured_voltage,
                measured_current=measured_current,
                applied_current=calibration_current,
                resistance=resistance,
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

            if not np.isfinite(resistance):
                print("Rejected room-temperature calibration sample: invalid resistance.")
                _sleep_with_stop(sample_interval_s, emitter)
                continue
            sample_temperature = float(room_temp)
            if np.isfinite(temperature):
                temperature_difference = abs(float(temperature) - room_temp)
                if (
                    temperature_difference > config["t0_max_temp_error_c"]
                    and not room_temp_scaling_fallback_used
                ):
                    _emit_calibration_warning(
                        emitter,
                        "The uncalibrated R-vs-T curve inferred "
                        f"{float(temperature):.2f} C while the entered T0 is {room_temp:.2f} C "
                        f"(difference {temperature_difference:.2f} C; warning threshold "
                        f"{config['t0_max_temp_error_c']:.2f} C). T0 calibration will continue using "
                        "the stable measured resistance and entered T0. Verify the material curve, units, "
                        "Kelvin wiring, and entered T0 before the experiment.",
                    )
                    room_temp_scaling_fallback_used = True
            else:
                if abs(measured_current) <= config["t0_stable_current_a"]:
                    print("Rejected room-temperature calibration sample: current is too small.")
                    _sleep_with_stop(sample_interval_s, emitter)
                    continue
                if not room_temp_scaling_fallback_used:
                    _emit_calibration_warning(
                        emitter,
                        "The uncalibrated R-vs-T curve could not infer a finite temperature for the measured "
                        f"resistance. T0 calibration will continue using the entered T0 of {room_temp:.2f} C. "
                        "Verify the material curve, units, Kelvin wiring, and entered T0 before the experiment.",
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

        filtered_samples = _filter_room_temperature_samples(accepted_samples, config=config)
        if len(filtered_samples) != len(accepted_samples):
            print(
                f"Using {len(filtered_samples)} of {len(accepted_samples)} room-temperature samples "
                "after resistance outlier filtering."
            )

        final_resistances = [sample["resistance"] for sample in filtered_samples]
        if not _resistance_series_is_stable(final_resistances, config):
            raise ValueError(
                "T0 resistance did not remain stable enough for a reliable low-TCR calibration. "
                "Use smaller fixed DMM ranges, allow more settling time, or increase Initial Voltage carefully."
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

        calibrated = _scale_and_anchor_curve(
            curve,
            scale,
            anchor_temperature=room_temp,
            anchor_resistance=measured_resistivity,
        )
        try:
            equivalent_spread = _calibrated_temperature_spread(
                final_resistances,
                scale,
                temperature_interp,
            )
        except Exception as exc:
            equivalent_spread = np.nan
            print(
                "WARNING: Could not calculate the T0 temperature-equivalent resistance spread; "
                f"the completed T0 calibration remains valid. Details: {exc}"
            )

        if np.isfinite(equivalent_spread):
            print(
                "T0 resistance uncertainty corresponds to an inferred temperature spread of "
                f"{equivalent_spread:.2f} C across {len(final_resistances)} samples."
            )
            warning_limit = float(config.get("t0_temperature_spread_warning_c", 5.0))
            if equivalent_spread > warning_limit:
                _emit_calibration_warning(
                    emitter,
                    f"T0 resistance scatter corresponds to {equivalent_spread:.2f} C, above the "
                    f"configured {warning_limit:.2f} C warning limit. The calibration anchor was saved, "
                    "but temperature estimates for this low-TCR wire may fluctuate. Increase Initial "
                    "Voltage carefully or improve the measurement signal before starting the experiment.",
                )
        return calibrated

    finally:
        tds_experiment._shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)


def _estimate_pid_from_step(response, base_temperature, step_current, loop_time, min_temp_rise, controller_mode="PI"):
    if not response:
        raise ValueError("Controller tuning did not collect any valid samples.")

    times = np.array([sample["elapsed_s"] for sample in response], dtype=float)
    temperatures = np.array([sample["temperature"] for sample in response], dtype=float)
    temperature_rise = temperatures - base_temperature
    peak_rise = float(np.max(temperature_rise))

    if peak_rise < min_temp_rise:
        raise ValueError(
            "Controller tuning did not produce enough temperature change. Increase tuning_search_max_current "
            "or tuning_current_step carefully."
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

    process_gain = peak_rise / max(step_current, 1e-6)
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
        "step_current": step_current,
        "peak_rise_c": peak_rise,
        "dead_time_s": dead_time_s,
        "time_constant_s": time_constant_s,
    }


def _collect_pid_baseline(
    *,
    dmm_v,
    dmm_i,
    power_supply,
    temperature_interp,
    config,
    emitter,
    baseline_current,
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

    sample_interval_s = _calibration_sample_interval_s(config)
    invalid_measurements = 0
    while len(baseline_temperatures) < int(config["tuning_baseline_samples"]):
        _check_stop(emitter)
        measured_voltage, measured_current, temperature, resistance = tds_experiment.measure_resistivity(
            dmm_v,
            dmm_i,
            siglent,
            temperature_interp,
            calibration=True,
            config=config,
            power_supply=power_supply,
        )
        _emit_live_measurement(
            emitter,
            target_temperature=target_temperature,
            temperature=temperature,
            measured_voltage=measured_voltage,
            measured_current=measured_current,
            applied_current=baseline_current,
            resistance=resistance,
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
            f"V={measured_voltage}, I={measured_current}, Vps={baseline_current:.4f}"
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
    baseline_current,
    response_current,
    base_temperature,
    desired_rise,
    required_rise,
    smoothed_required_rise,
    safe_temperature_limit,
    temperature_lower_bound,
    loop_time,
):
    siglent.set_current(power_supply, current=response_current)
    response = []
    invalid_measurements = 0
    start_time = time.time()
    best_smoothed_rise_so_far = float("-inf")
    last_growth_time_s = 0.0

    while time.time() - start_time < config["tuning_max_duration_s"]:
        _check_stop(emitter)
        loop_started = time.time()
        measured_voltage, measured_current, temperature, resistance = tds_experiment.measure_resistivity(
            dmm_v,
            dmm_i,
            siglent,
            temperature_interp,
            calibration=True,
            config=config,
            power_supply=power_supply,
        )
        _emit_live_measurement(
            emitter,
            target_temperature=base_temperature + desired_rise,
            temperature=temperature,
            measured_voltage=measured_voltage,
            measured_current=measured_current,
            applied_current=response_current,
            resistance=resistance,
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
            f"I={measured_current:.4e} A, Vps={response_current:.4f} A"
        )

        if smoothed_rise_so_far > best_smoothed_rise_so_far + config["tuning_plateau_growth_c"]:
            best_smoothed_rise_so_far = smoothed_rise_so_far
            last_growth_time_s = elapsed_s

        # A high-gain point can cross the rise threshold on its very first
        # sample, before the response curve has revealed its actual shape.
        # Accepting that immediately starves _estimate_pid_from_step of the
        # samples it needs to identify dead time and time constant, both of
        # which collapse toward loop_time with too few points - producing a
        # falsely tiny tau and an over-aggressive Ki.
        if (
            len(response) >= int(config.get("tuning_min_response_samples", 5))
            and peak_rise_so_far >= required_rise
            and smoothed_rise_so_far >= smoothed_required_rise
        ):
            return {
                "status": "usable_response",
                "response": response,
                "peak_rise_c": peak_rise_so_far,
                "smoothed_rise_c": smoothed_rise_so_far,
                "elapsed_s": elapsed_s,
            }

        if (
            len(response) >= int(config.get("tuning_min_response_samples", 5))
            and temperature >= base_temperature + desired_rise
        ):
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


def _suggest_current_step(process_gain, time_constant_s, loop_time, config):
    """Suggest a per-loop current step from an identified process gain/tau.

    Bounds how much of a step's eventual temperature effect can appear
    within one control loop period to temperature_tolerance_c, using the
    fraction of a first-order step response that manifests in one loop:
    1 - exp(-loop_time / tau). A slower process (large tau relative to the
    loop period) tolerates a bigger step, since only a small fraction of its
    effect can show up before the controller gets another chance to react.
    This is a starting suggestion, not a guarantee - it does not account for
    sustained ramp rate, which the existing rate-limiting logic still governs
    separately.
    """
    try:
        process_gain = float(process_gain)
        time_constant_s = float(time_constant_s)
        loop_time = float(loop_time)
    except (TypeError, ValueError):
        return float(config["minimum_current_change"])
    if not np.isfinite(process_gain) or process_gain <= 0:
        return float(config["minimum_current_change"])

    tolerance = float(config.get("temperature_tolerance_c", 2.0))
    tau = max(time_constant_s, 1e-6)
    dt = max(loop_time, 1e-6)
    fraction_manifested = max(1.0 - np.exp(-dt / tau), 1e-6)
    raw_step = tolerance / (process_gain * fraction_manifested)

    lower_bound = float(config["minimum_current_change"])
    upper_bound = max(0.1 * float(config["max_current"]), lower_bound)
    return float(np.clip(raw_step, lower_bound, upper_bound))


def _tune_one_point(
    *,
    dmm_v,
    dmm_i,
    power_supply,
    temperature_interp,
    config,
    controller_mode,
    loop_time,
    experiment_params,
    base_temperature_hint,
    start_current,
    emitter,
    label,
):
    """Run one full baseline-search-plus-step-response tuning sequence.

    Returns Kp/Ki/Kd plus the identified dead_time_s/time_constant_s and a
    derived process_gain_c_per_a, at the single current this call tests.
    """
    stable_temperature_window = config["tuning_temperature_window_c"]
    temperature_lower_bound = None
    temperature_upper_bound = None
    if base_temperature_hint is not None:
        temperature_lower_bound = base_temperature_hint - stable_temperature_window
        temperature_upper_bound = base_temperature_hint + stable_temperature_window

    stable_setpoint_current, stable_samples = _find_stable_current_setpoint(
        dmm_v=dmm_v,
        dmm_i=dmm_i,
        power_supply=power_supply,
        temperature_interp=temperature_interp,
        config=config,
        start_current=start_current,
        max_current=max(start_current, config["tuning_search_max_current"], config["max_current"]),
        step_current=config["tuning_current_step"],
        settle_time_s=config["tuning_settle_time_s"],
        stable_samples=config["tuning_stable_current_samples"],
        minimum_current=config["tuning_stable_current_a"],
        emitter=emitter,
        label=label,
        temperature_lower_bound=temperature_lower_bound,
        temperature_upper_bound=temperature_upper_bound,
        display_target_temperature=base_temperature_hint,
        stop_on_high_temperature=True,
    )
    baseline_current = stable_setpoint_current
    print(f"Using {label} baseline current: {baseline_current:.4f} A")

    # A step-response test is a deliberate, one-shot excitation, not the
    # continuously running control loop - it must not be throttled by
    # max_current_step_up (0.01 A), which would otherwise force many
    # multi-minute retries just to reach a step big enough to see clearly.
    # Every sample during the attempt still goes through the same
    # max_current/max_power_w/max_sample_voltage checks as anywhere else.
    response_step = max(
        config["tuning_response_current_step"],
        config["minimum_current_change"],
        config.get("tuning_response_relative_step", 0.5) * baseline_current,
    )
    max_response_voltage = min(config["tuning_search_max_current"], config["max_current"])
    candidate_current = tds_experiment._clamp(
        baseline_current + response_step,
        baseline_current,
        max_response_voltage,
    )
    if candidate_current <= baseline_current + 1e-12:
        raise ValueError(
            f"{label}: could not create a current step above the stable baseline. "
            "Increase tuning_search_max_current carefully."
        )

    seeded_baseline_samples = stable_samples
    last_failure = None
    attempt_number = 0
    while candidate_current <= max_response_voltage + 1e-12:
        attempt_number += 1
        print(
            f"{label} attempt {attempt_number}: baseline={baseline_current:.4f} A, "
            f"response={candidate_current:.4f} A"
        )
        siglent.set_current(power_supply, current=baseline_current)
        _sleep_with_stop(config["tuning_between_attempts_s"], emitter)

        base_temperature = _collect_pid_baseline(
            dmm_v=dmm_v,
            dmm_i=dmm_i,
            power_supply=power_supply,
            temperature_interp=temperature_interp,
            config=config,
            emitter=emitter,
            baseline_current=baseline_current,
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
                f"{label}: target temperature is not above the current temperature, "
                "so controller tuning cannot proceed."
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
            baseline_current=baseline_current,
            response_current=candidate_current,
            base_temperature=base_temperature,
            desired_rise=desired_rise,
            required_rise=required_rise,
            smoothed_required_rise=smoothed_required_rise,
            safe_temperature_limit=safe_temperature_limit,
            temperature_lower_bound=temperature_lower_bound,
            loop_time=loop_time,
        )

        siglent.set_current(power_supply, current=baseline_current)
        _sleep_with_stop(config["tuning_between_attempts_s"], emitter)

        if (
            attempt["peak_rise_c"] >= required_rise
            and attempt["smoothed_rise_c"] >= smoothed_required_rise
            and attempt["response"]
        ):
            tuned = _estimate_pid_from_step(
                response=attempt["response"],
                base_temperature=base_temperature,
                step_current=candidate_current - baseline_current,
                loop_time=loop_time,
                min_temp_rise=required_rise,
                controller_mode=controller_mode,
            )
            tuned["baseline_current"] = baseline_current
            tuned["step_current"] = candidate_current
            tuned["step_delta_current"] = candidate_current - baseline_current
            tuned["process_gain_c_per_a"] = tuned["peak_rise_c"] / max(tuned["step_delta_current"], 1e-9)
            print(
                f"Tuned {label}: Kp={tuned['Kp']:.6f}, Ki={tuned['Ki']:.6f}, "
                f"Kd={tuned['Kd']:.6f}, baseline={baseline_current:.4f} A, "
                f"response={candidate_current:.4f} A, delta={tuned['step_delta_current']:.4f} A, "
                f"peak rise={tuned['peak_rise_c']:.2f} C"
            )
            return tuned

        last_failure = (
            f"Attempt at {candidate_current:.4f} A ended with status {attempt['status']} and produced "
            f"{attempt['smoothed_rise_c']:.2f} C smoothed rise "
            f"({attempt['peak_rise_c']:.2f} C peak)."
        )
        print(f"{label} attempt did not produce enough response. {last_failure}")
        # Recompute relative to the current candidate, not the original
        # baseline, so retries climb geometrically (1.5x, 2.25x, ...) from
        # even a tiny starting current instead of creeping up by a fixed
        # absolute amount that could take dozens of multi-minute attempts
        # to reach a representative operating current.
        response_step = max(
            config["tuning_response_current_step"],
            config["minimum_current_change"],
            config.get("tuning_response_relative_step", 0.5) * candidate_current,
        )
        next_candidate_current = tds_experiment._clamp(
            candidate_current + response_step,
            baseline_current,
            max_response_voltage,
        )
        if next_candidate_current <= candidate_current + 1e-12:
            break
        candidate_current = next_candidate_current

    failure_message = (
        f"{label} could not find a usable step response up to {max_response_voltage:.4f} A. "
        f"{last_failure or ''}"
    ).strip()
    raise ValueError(failure_message)


def tune_pid(experiment_params, config, r_vs_t, base_temperature_hint=None, emitter=None):
    """
    Tune conservative gains from a small guarded current step on the real setup.

    Single-point tuning, kept for callers that want just one gain set.
    tune_pid_schedule is the multi-point version used by the GUI.
    """
    config = tds_experiment.build_control_config(config)
    controller_mode = tds_experiment.get_controller_mode(config)
    loop_time = tds_experiment.resistivity_loop_time(config)
    curve, _, temperature_interp = _prepare_curve_interpolators(r_vs_t, config=config)

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

        tds_experiment.prepare_power_supply_output(power_supply, config)
        siglent.configure_dc_range_from_config(dmm_v, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm_i, "CURR", config)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        _sleep_with_stop(1.0, emitter)

        return _tune_one_point(
            dmm_v=dmm_v,
            dmm_i=dmm_i,
            power_supply=power_supply,
            temperature_interp=temperature_interp,
            config=config,
            controller_mode=controller_mode,
            loop_time=loop_time,
            experiment_params=experiment_params,
            base_temperature_hint=base_temperature_hint,
            start_current=config["tuning_start_current"],
            emitter=emitter,
            label=f"{controller_mode} tuning",
        )
    finally:
        tds_experiment._shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)


def _tuning_schedule_targets(config):
    """Return the (name, start_current) points tune_pid_schedule tests.

    Targets sit at tuning_start_current, 40%, and 80% of max_current, kept
    only when meaningfully separated (>20%) from the previous one so a small
    max_current collapses to fewer points instead of tuning at
    near-duplicate currents.
    """
    max_current = float(config["max_current"])
    low_start = float(config["tuning_start_current"])
    mid_start = tds_experiment._clamp(0.4 * max_current, low_start, max_current)
    high_start = tds_experiment._clamp(0.8 * max_current, low_start, max_current)
    targets = []
    for name, start_current in (("low", low_start), ("mid", mid_start), ("high", high_start)):
        if not targets or start_current > targets[-1][1] * 1.2:
            targets.append((name, start_current))
    return targets


def _cap_next_tuning_target(desired_current, previous_current, previous_gain, config):
    """Bound how far the next schedule point's baseline may jump from the last.

    A wire's process gain can be far larger at low power than a fixed
    fraction of max_current assumes - one observed case measured ~800 C/A at
    a few milliamps, where jumping straight to 40% of a 1 A max_current drove
    the sample past 500 C before the first stability sample was even taken.
    Bounding the jump by the previous point's own measured gain (a
    plausibility check against what has actually been seen, not a promise the
    wire stays that sensitive) keeps each step's own risk in check; the point
    tuned there then supplies a fresh, local gain for the next jump.
    """
    if previous_gain is None or not np.isfinite(previous_gain) or previous_gain <= 0:
        max_jump = float(config.get("tuning_current_step", 0.001)) * 10.0
    else:
        max_safe_rise = float(config.get("tuning_schedule_jump_max_rise_c", 10.0))
        max_jump = max_safe_rise / previous_gain
    capped = min(float(desired_current), float(previous_current) + max_jump)
    return max(capped, float(previous_current) + float(config["minimum_current_change"]))


def tune_pid_schedule(experiment_params, config, r_vs_t, base_temperature_hint=None, emitter=None):
    """
    Tune gains at low/mid/high currents and build a schedule covering the
    whole operating range, instead of one fixed-gain result.

    A wire's process gain (temperature rise per amp) is unlikely to stay
    constant from a near-zero-power tuning point up to the several-watt
    operating point of a real heating run - self-heating shifts which loss
    mechanism (conduction vs. radiation) dominates. Returns one gain set per
    tested current plus suggested max_current_step_up/down and
    low_current_max_step_up/down, derived from the identified process gain
    and time constant at each point (see _suggest_current_step): the low
    pair from the lowest current tested, the normal pair from the highest,
    since that is representative of sustained real operation.
    """
    config = tds_experiment.build_control_config(config)
    controller_mode = tds_experiment.get_controller_mode(config)
    loop_time = tds_experiment.resistivity_loop_time(config)
    curve, _, temperature_interp = _prepare_curve_interpolators(r_vs_t, config=config)

    targets = _tuning_schedule_targets(config)

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

        tds_experiment.prepare_power_supply_output(power_supply, config)
        siglent.configure_dc_range_from_config(dmm_v, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm_i, "CURR", config)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        _sleep_with_stop(1.0, emitter)

        points = []
        previous_current = None
        previous_gain = None
        for name, desired_current in targets:
            _check_stop(emitter)
            if previous_current is None:
                start_current = desired_current
            else:
                start_current = _cap_next_tuning_target(
                    desired_current, previous_current, previous_gain, config
                )
                if start_current < desired_current - 1e-9:
                    print(
                        f"{controller_mode} tuning ({name}): capping the target from "
                        f"{desired_current:.4f} A to {start_current:.4f} A based on the "
                        f"{previous_gain:.3g} C/A gain measured at {previous_current:.4f} A."
                    )

            try:
                tuned = _tune_one_point(
                    dmm_v=dmm_v,
                    dmm_i=dmm_i,
                    power_supply=power_supply,
                    temperature_interp=temperature_interp,
                    config=config,
                    controller_mode=controller_mode,
                    loop_time=loop_time,
                    experiment_params=experiment_params,
                    base_temperature_hint=base_temperature_hint,
                    start_current=start_current,
                    emitter=emitter,
                    label=f"{controller_mode} tuning ({name})",
                )
            except tds_experiment.ExperimentSafetyError as exc:
                print(
                    f"{controller_mode} tuning ({name}) stopped by a safety limit at "
                    f"{start_current:.4f} A: {exc}. Keeping the {len(points)} point(s) already tuned "
                    "and not attempting any higher current."
                )
                break
            except ValueError as exc:
                print(
                    f"{controller_mode} tuning ({name}) did not produce a usable result at "
                    f"{start_current:.4f} A: {exc}. Continuing with the remaining point(s)."
                )
                continue

            tuned["point_name"] = name
            points.append(tuned)
            previous_current = tuned["step_current"]
            previous_gain = tuned["process_gain_c_per_a"]
            print(
                f"{name} point done: current={tuned['step_current']:.4f} A, "
                f"gain={tuned['process_gain_c_per_a']:.3g} C/A, tau={tuned['time_constant_s']:.1f} s"
            )

        if not points:
            raise ValueError(
                f"{controller_mode} tuning could not produce any usable schedule point. "
                "Check the sample connection and Initial Current before retrying."
            )

        points_by_current = sorted(points, key=lambda point: point["step_current"])
        schedule = [
            {
                "current_a": float(point["step_current"]),
                "kp": float(point["Kp"]),
                "ki": float(point["Ki"]),
                "kd": float(point["Kd"]),
            }
            for point in points_by_current
        ]

        low_point = points_by_current[0]
        high_point = points_by_current[-1]
        low_step = _suggest_current_step(
            low_point["process_gain_c_per_a"], low_point["time_constant_s"], loop_time, config
        )
        high_step = _suggest_current_step(
            high_point["process_gain_c_per_a"], high_point["time_constant_s"], loop_time, config
        )

        result = {
            "schedule": schedule,
            "points": points,
            "low_current_max_step_up": low_step,
            "low_current_max_step_down": low_step,
            "max_current_step_up": high_step,
            "max_current_step_down": high_step,
            "Kp": schedule[0]["kp"],
            "Ki": schedule[0]["ki"],
            "Kd": schedule[0]["kd"],
        }
        print(
            f"{controller_mode} gain schedule tuned at {len(points)} point(s): "
            + ", ".join(f"{p['current_a']:.4f} A" for p in schedule)
        )
        print(
            f"Suggested step limits: low_current_max_step={low_step:.4f} A, "
            f"max_current_step={high_step:.4f} A"
        )
        return result
    finally:
        tds_experiment._shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)
