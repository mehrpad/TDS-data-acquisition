import time
from dataclasses import dataclass

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

from . import pid
from . import siglent


CONTROL_DEFAULTS = {
    "pid_kp": 0.008,
    "pid_ki": 0.0004,
    "pid_kd": 0.0,
    "pid_integral_limit": 400.0,
    "pid_derivative_filter": 0.6,
    "startup_voltage": 0.01,
    "min_voltage": 0.0,
    "max_voltage_step_up": 0.02,
    "max_voltage_step_down": 0.08,
    "temperature_tolerance_c": 2.0,
    "hold_entry_tolerance_c": 3.0,
    "safety_temp_margin_c": 15.0,
    "soft_temp_rate_margin_c_min": 1.0,
    "hard_temp_rate_margin_c_min": 4.0,
    "measurement_fail_limit": 3,
    "minimum_current_a": 1e-9,
    "minimum_voltage_change": 1e-4,
    "autosave_flush_interval_s": 5.0,
    "autosave_batch_size": 10,
    "tuning_voltage_step": 0.01,
    "tuning_max_duration_s": 180.0,
    "tuning_baseline_samples": 5,
    "tuning_target_rise_c": 3.0,
    "tuning_min_temperature_rise_c": 1.5,
    "t0_calibration_voltage": 0.1,
    "t0_settle_time_s": 3.0,
    "t0_calibration_samples": 5,
    "t0_warmup_samples": 1,
    "t0_max_temp_error_c": 80.0,
}


class ExperimentSafetyError(RuntimeError):
    """Raised when the experiment should stop to protect the sample or setup."""


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


def build_control_config(config):
    merged = dict(config)
    for key, value in CONTROL_DEFAULTS.items():
        merged.setdefault(key, value)
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
                if target >= self.start_T and measured_temperature >= self.start_T - self.temperature_tolerance_c:
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

    if temperature >= setpoint + config["temperature_tolerance_c"]:
        delta_voltage = min(delta_voltage, 0.0)

    soft_rate_limit = max(
        ramp_speed_c_min + config["soft_temp_rate_margin_c_min"],
        config["soft_temp_rate_margin_c_min"],
    )
    hard_rate_limit = max(
        ramp_speed_c_min + config["hard_temp_rate_margin_c_min"],
        config["hard_temp_rate_margin_c_min"],
    )

    if temp_rate_c_min is not None and np.isfinite(temp_rate_c_min):
        if temp_rate_c_min > soft_rate_limit and delta_voltage > 0.0:
            delta_voltage = 0.0
        if temp_rate_c_min > soft_rate_limit and temperature >= setpoint - config["temperature_tolerance_c"]:
            delta_voltage = min(delta_voltage, -config["max_voltage_step_down"] / 2.0)
        if temp_rate_c_min > hard_rate_limit:
            pid_controller.reset(measurement=temperature)
            delta_voltage = -config["max_voltage_step_down"]

    if abs(measured_current) >= 0.95 * config["max_current"] and delta_voltage > 0.0:
        delta_voltage = 0.0

    if temperature >= target_temperature and setpoint >= target_temperature:
        delta_voltage = min(delta_voltage, 0.0)

    new_voltage = _clamp(current_voltage + delta_voltage, config["min_voltage"], config["max_voltage"])
    return new_voltage


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
            )

            pid_controller = pid.PIDController(
                kp=config["pid_kp"],
                ki=config["pid_ki"],
                kd=config["pid_kd"],
                setpoint=t_zero,
                output_limits=(-config["max_voltage_step_down"], config["max_voltage_step_up"]),
                integral_limits=(-config["pid_integral_limit"], config["pid_integral_limit"]),
                derivative_filter=config["pid_derivative_filter"],
            )

            pid_voltage = _clamp(config["startup_voltage"], config["min_voltage"], config["max_voltage"])
            previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)
            time.sleep(max(2.0, loop_time))

            measured_voltage, measured_current, temperature = measure_resistivity(
                dmm_v, dmm_i, siglent, temperature_interp
            )
            if not _is_valid_measurement(measured_voltage, measured_current, temperature, config):
                raise ExperimentSafetyError("Unable to acquire a valid initial measurement.")

            print(
                f"Initial temperature {temperature:.2f} C, start temperature {program.start_T:.2f} C, "
                f"resistance {measured_voltage / measured_current:.6e} Ohm"
            )

            program.initialize(max(temperature, t_zero))
            pid_controller.reset(measurement=temperature)
            invalid_measurements = 0
            previous_temperature = temperature
            previous_phase = None

            while not emitter.stopped:
                loop_started = time.time()
                measured_voltage, measured_current, temperature = measure_resistivity(
                    dmm_v, dmm_i, siglent, temperature_interp
                )

                if not _is_valid_measurement(measured_voltage, measured_current, temperature, config):
                    invalid_measurements += 1
                    pid_voltage = _clamp(
                        pid_voltage - config["max_voltage_step_down"],
                        config["min_voltage"],
                        config["max_voltage"],
                    )
                    previous_voltage = _set_voltage_if_needed(power_supply, pid_voltage, previous_voltage, config)
                    print(
                        "Invalid measurement received. "
                        f"Reducing voltage to {pid_voltage:.4f} V (attempt {invalid_measurements})."
                    )
                    if invalid_measurements >= config["measurement_fail_limit"]:
                        raise ExperimentSafetyError("Too many invalid measurements in a row.")
                    _persist_measurement(
                        data_saver,
                        program.scheduled_target,
                        np.nan,
                        measured_voltage,
                        measured_current,
                        pid_voltage,
                    )
                    _emit_measurement(emitter, program.scheduled_target, np.nan, measured_voltage, measured_current, pid_voltage)
                    elapsed = time.time() - loop_started
                    if elapsed < loop_time:
                        time.sleep(loop_time - elapsed)
                    continue

                invalid_measurements = 0
                temp_rate_c_min = _temperature_rate_c_min(temperature, previous_temperature, loop_time)
                setpoint, phase, finished = program.update(temperature, loop_time)

                if phase != previous_phase:
                    pid_controller.reset(measurement=temperature)
                    previous_phase = phase

                pid_voltage = _compute_next_voltage(
                    pid_controller=pid_controller,
                    temperature=temperature,
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
                    f"Phase: {phase}, T: {temperature:.2f} C, Setpoint: {setpoint:.2f} C, "
                    f"Voltage: {pid_voltage:.4f} V, Current: {measured_current:.4e} A, "
                    f"Rate: {temp_rate_c_min if temp_rate_c_min is not None else 0.0:.2f} C/min"
                )
                _persist_measurement(
                    data_saver,
                    setpoint,
                    temperature,
                    measured_voltage,
                    measured_current,
                    pid_voltage,
                )
                _emit_measurement(emitter, setpoint, temperature, measured_voltage, measured_current, pid_voltage)
                previous_temperature = temperature

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


def measure_resistivity(dmm_v, dmm_i, siglent_module, temperature_interp, calibration=False):
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

    try:
        temperature = float(temperature_interp(resistance))
    except Exception as exc:
        print(f"An error occurred interpolating temperature: {exc}")
        temperature = np.nan

    if np.isfinite(temperature) and temperature < 0 and not calibration:
        print(f"Calculated temperature is {temperature}; clamping it to 0 C.")
        temperature = 0.0

    return measured_voltage, measured_current, temperature
