import time

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

from . import siglent
from . import tds_experiment


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


def calibrate_temperature_curve(r_vs_t, room_temp, config=None):
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
        time.sleep(0.04)
        low_voltage = min(config["startup_voltage"], config["tuning_voltage_step"], config["max_voltage"])
        low_voltage = max(low_voltage, 0.005)
        siglent.set_voltage(power_supply, voltage=low_voltage)
        time.sleep(max(2.0, 1.0 / config["experiment_frequency"]))
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])
        time.sleep(1.0)

        measured_currents = []
        measured_voltages = []
        temperatures = []
        attempts = 0
        max_attempts = max(10, config["measurement_fail_limit"] * 5)
        while len(measured_currents) < 5 and attempts < max_attempts:
            attempts += 1
            measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
                calibration=True,
            )
            print(
                f"Room-temperature calibration sample: T={temperature}, V={measured_voltage}, I={measured_current}"
            )
            if tds_experiment._is_valid_measurement(measured_voltage, measured_current, temperature, config):
                if abs(measured_current) <= config["max_current"]:
                    measured_currents.append(measured_current)
                    measured_voltages.append(measured_voltage)
                    temperatures.append(temperature)
            time.sleep(max(0.5, 1.0 / config["experiment_frequency"]))

        if len(measured_currents) < 3:
            raise ValueError("Could not collect enough stable room-temperature calibration samples.")

        measured_current = float(np.mean(np.array(measured_currents)))
        measured_voltage = float(np.mean(np.array(measured_voltages)))
        temperature = float(np.mean(np.array(temperatures)))
        print(f"Final room-temperature sample: T={temperature}, V={measured_voltage}, I={measured_current}")

        measured_resistivity = measured_voltage / measured_current
        if measured_resistivity <= 0:
            print(f"Measured resistivity {measured_resistivity:.4f} Ohm is invalid.")
            return None

        resistivity_room_temp = float(resistivity_interp(room_temp))
        scale = measured_resistivity / resistivity_room_temp
        print(f"Measured resistivity: {measured_resistivity:.4f} Ohm")
        print(f"Reference resistivity: {resistivity_room_temp:.4f} Ohm")
        print(f"Calibration scale: {scale:.4f}")

        calibrated = curve.copy()
        calibrated[0, :] *= scale
        return calibrated

    finally:
        tds_experiment._shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)


def _estimate_pid_from_step(response, base_temperature, step_voltage, loop_time, min_temp_rise):
    if not response:
        raise ValueError("PID tuning did not collect any valid samples.")

    times = np.array([sample["elapsed_s"] for sample in response], dtype=float)
    temperatures = np.array([sample["temperature"] for sample in response], dtype=float)
    temperature_rise = temperatures - base_temperature
    peak_rise = float(np.max(temperature_rise))

    if peak_rise < min_temp_rise:
        raise ValueError(
            "PID tuning did not produce enough temperature change. Increase tuning_voltage_step carefully."
        )

    threshold = max(0.1 * peak_rise, 0.5)
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

    # Resistive temperature measurements are noisy enough that a conservative PI
    # controller is safer than an aggressive derivative term.
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


def tune_pid(experiment_params, config, r_vs_t):
    """
    Tune conservative gains from a small guarded voltage step on the real setup.
    """
    config = tds_experiment.build_control_config(config)
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
        time.sleep(0.04)
        siglent.set_voltage(power_supply, voltage=0.0)
        time.sleep(1.0)
        siglent.set_mode_speed(dmm_i, "CURR", config["DMM_speed"])
        siglent.set_mode_speed(dmm_v, "VOLT", config["DMM_speed"])

        baseline_temperatures = []
        for _ in range(int(config["tuning_baseline_samples"])):
            measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
            )
            if tds_experiment._is_valid_measurement(measured_voltage, measured_current, temperature, config):
                baseline_temperatures.append(temperature)
            time.sleep(loop_time)

        if not baseline_temperatures:
            raise ValueError("Could not get a stable baseline temperature for PID tuning.")

        base_temperature = float(np.mean(np.array(baseline_temperatures)))
        available_rise = max(0.0, experiment_params["target_T"] - base_temperature)
        desired_rise = min(config["tuning_target_rise_c"], available_rise)
        if desired_rise < config["tuning_min_temperature_rise_c"] and available_rise > 0:
            desired_rise = available_rise
        if desired_rise <= 0:
            raise ValueError("Target temperature is not above the current temperature, so PID tuning cannot proceed.")

        safe_temperature_limit = min(
            experiment_params["target_T"],
            base_temperature + desired_rise + config["temperature_tolerance_c"],
        )
        step_voltage = min(config["tuning_voltage_step"], config["max_voltage"])
        step_voltage = max(step_voltage, config["max_voltage_step_up"])

        siglent.set_voltage(power_supply, voltage=step_voltage)
        response = []
        invalid_measurements = 0
        start_time = time.time()

        while time.time() - start_time < config["tuning_max_duration_s"]:
            loop_started = time.time()
            measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
                dmm_v,
                dmm_i,
                siglent,
                temperature_interp,
            )

            if not tds_experiment._is_valid_measurement(measured_voltage, measured_current, temperature, config):
                invalid_measurements += 1
                if invalid_measurements >= config["measurement_fail_limit"]:
                    raise ValueError("Too many invalid measurements during PID tuning.")
                elapsed = time.time() - loop_started
                if elapsed < loop_time:
                    time.sleep(loop_time - elapsed)
                continue

            invalid_measurements = 0
            if abs(measured_current) > config["max_current"]:
                raise tds_experiment.ExperimentSafetyError(
                    f"Measured current {measured_current:.4e} A exceeded max_current during tuning."
                )
            if temperature > safe_temperature_limit:
                break

            elapsed_s = time.time() - start_time
            response.append(
                {
                    "elapsed_s": elapsed_s,
                    "temperature": temperature,
                    "current": measured_current,
                }
            )
            print(
                f"Tuning sample: t={elapsed_s:.1f} s, T={temperature:.2f} C, "
                f"I={measured_current:.4e} A, Vstep={step_voltage:.4f} V"
            )

            if temperature >= base_temperature + desired_rise:
                break

            elapsed = time.time() - loop_started
            if elapsed < loop_time:
                time.sleep(loop_time - elapsed)

        siglent.set_voltage(power_supply, voltage=0.0)
        tuned = _estimate_pid_from_step(
            response=response,
            base_temperature=base_temperature,
            step_voltage=step_voltage,
            loop_time=loop_time,
            min_temp_rise=min(config["tuning_min_temperature_rise_c"], desired_rise),
        )
        print(
            f"Tuned PID parameters: Kp={tuned['Kp']:.6f}, Ki={tuned['Ki']:.6f}, Kd={tuned['Kd']:.6f}, "
            f"peak rise={tuned['peak_rise_c']:.2f} C"
        )
        return tuned

    finally:
        tds_experiment._shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)
