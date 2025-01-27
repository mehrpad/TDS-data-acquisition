


import time
import datetime

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

import siglent, tds_experiment




def calibrate_temperature_curve(r_vs_t, room_temp):
    """
    Calibrate the resistivity vs temperature curve so that the measured resistivity at room temperature matches 23 °C.

    Args:
        r_vs_t (numpy.ndarray): 2D array where the first row is resistivity and the second row is temperature.
        room_temp (float): The room temperature in °C.

    Returns:
        interp1d: Calibrated interpolation function for temperature vs resistivity.
    """
    # Create an interpolation function for resistivity vs temperature
    resistivity_interp = interp1d(r_vs_t[1, :], r_vs_t[0, :], kind='linear', fill_value='extrapolate')
    temperature_interp = interp1d(r_vs_t[0, :], r_vs_t[1, :], kind='linear', fill_value='extrapolate')

    # Initialize Resource Manager and Devices
    rm = pyvisa.ResourceManager()
    DMM_v = rm.open_resource('USB0::0xF4EC::0xEE38::SDM35FAC4R0253::INSTR')  # Digital Multimeter
    DMM_i = rm.open_resource('USB0::0xF4EC::0x1201::SDM35HBQ803105::INSTR')  # Digital Multimeter
    PS = rm.open_resource('USB0::0xF4EC::0x1410::SPD13DCC4R0058::INSTR')  # Power Supply
    PS.write_termination = '\n'
    PS.read_termination = '\n'
    siglent.set_output(PS, state='ON')
    time.sleep(0.04)
    siglent.set_voltage(PS, voltage=0.01)
    time.sleep(2)
    measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(DMM_v, DMM_i, siglent,
                                                                          temperature_interp)
    measured_resistivity = measured_voltage / measured_current
    # Calculate the resistivity at room temperature
    resistivity_room_temp = resistivity_interp(room_temp).item()

    # Calculate the shift in resistivity
    delta_resistivity = measured_resistivity - resistivity_room_temp
    print(f"Measured resistivity: {measured_resistivity:.4f} Ohm")
    print(f"Resistivity at room temperature: {resistivity_room_temp:.4f} Ohm")
    print(f"Shift in resistivity: {delta_resistivity:.4f} Ohm")

    # Apply the shift to resistivity values
    r_vs_t_calibrated = r_vs_t.copy()
    r_vs_t_calibrated[0, :] += delta_resistivity  # Adjust resistivity values

    siglent.set_voltage(PS, voltage=0.0)
    time.sleep(0.01)
    DMM_v.close()
    DMM_i.close()
    siglent.set_output(PS, state='OFF')
    time.sleep(1)
    PS.close()
    rm.close()

    return r_vs_t_calibrated


def calibrate_pid(start_T, target_T, ramp_speed, measure_temperature, set_voltage, loop_time=0.1, max_iter=100):
    """
    Calibrate PID parameters for temperature control.

    Parameters:
        start_T (float): Starting temperature.
        target_T (float): Target temperature.
        ramp_speed (float): Ramp speed in °C/min.
        measure_temperature (callable): Function to measure the current temperature.
        set_voltage (callable): Function to set the power supply voltage.
        loop_time (float): Time interval for loop (in seconds).
        max_iter (int): Maximum number of iterations for calibration.

    Returns:
        dict: Optimal PID parameters (Kp, Ki, Kd).
    """
    from scipy.optimize import minimize

    def pid_response(params):
        Kp, Ki, Kd = params
        integral = 0
        prev_error = 0
        temperature = start_T
        time_elapsed = 0
        total_error = 0

        while temperature < target_T:
            desired_temperature = min(start_T + ramp_speed * (time_elapsed / 60), target_T)
            error = desired_temperature - temperature
            integral += error * loop_time
            derivative = (error - prev_error) / loop_time
            prev_error = error
            pid_voltage = Kp * error + Ki * integral + Kd * derivative

            # Limit the voltage
            pid_voltage = max(0.003, min(pid_voltage, 20.0))

            set_voltage(pid_voltage)
            time.sleep(loop_time)
            temperature = measure_temperature()
            time_elapsed += loop_time

            # Evaluate performance (e.g., Integral of Absolute Error)
            total_error += abs(error) * loop_time

        return total_error
