


import time
import datetime

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

import siglent, tds_experiment, pid




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
    time.sleep(3)
    # Set the speed and mode of the DMMs
    siglent.set_mode_speed(DMM_i, 'CURR', 1)
    siglent.set_mode_speed(DMM_v, 'VOLT', 1)
    time.sleep(1)
    # To be accurate, we measure the resistivity at room temperature 3 times and take the average
    measured_current = []
    measured_voltage = []
    temperature = []
    success_num = 0
    while success_num < 3:
        measured_voltage_c, measured_current_c, temperature_c = tds_experiment.measure_resistivity(DMM_v, DMM_i, siglent, temperature_interp)
        if measured_current != 0 and measured_voltage != 0 and temperature != 0:
            success_num += 1
            measured_current.append(measured_current_c)
            measured_voltage.append(measured_voltage_c)
            temperature.append(temperature_c)
        time.sleep(1)

    measured_current = np.mean(np.array(measured_current))
    measured_voltage = np.mean(np.array(measured_voltage))
    temperature = np.mean(np.array(temperature))
    print(f"The Temperature:{temperature}, Voltage:{measured_voltage}, Current:{measured_current}")
    measured_resistivity = measured_voltage / measured_current
    if measured_resistivity > 0:
        # Calculate the resistivity at room temperature
        resistivity_room_temp = resistivity_interp(room_temp).item()

        # Calculate the shift in resistivity
        delta_resistivity = measured_resistivity - resistivity_room_temp
        print(f"Measured resistivity: {measured_resistivity:.4f} Ohm")
        print(f"Resistivity at zero temperature: {resistivity_room_temp:.4f} Ohm")
        print(f"Shift in resistivity: {delta_resistivity:.4f} Ohm")

        # Apply the shift to resistivity values
        r_vs_t_calibrated = r_vs_t.copy()
        # r_vs_t_calibrated[0, :] += delta_resistivity  # Adjust resistivity values
        r_vs_t_calibrated[0, :] *= measured_resistivity / resistivity_room_temp
    else:
        print(f"Measured resistivity: {measured_resistivity:.4f} Ohm")
        print(f"The calibration failed")
        r_vs_t_calibrated = None

    siglent.set_voltage(PS, voltage=0.0)
    time.sleep(0.01)
    DMM_v.close()
    DMM_i.close()
    siglent.set_output(PS, state='OFF')
    time.sleep(1)
    PS.close()
    rm.close()

    return r_vs_t_calibrated


def tune_pid(experiment_params, config, r_vs_t, max_iter=5):
    """
    Calibrate PID parameters for temperature control.
    Returns optimal PID parameters (Kp, Ki, Kd) for the given setup.
    """
    # Initialize instruments
    rm = pyvisa.ResourceManager()
    DMM_v = rm.open_resource(config['DMM_v'])  # Voltage measurement
    DMM_i = rm.open_resource(config['DMM_i'])  # Current measurement
    PS = rm.open_resource(config['PS'])  # Power supply
    PS.write_termination = '\n'
    PS.read_termination = '\n'

    siglent.set_output(PS, state='ON')
    time.sleep(0.04)

    loop_time = 1 / config['experiment_frequency']  # Loop time in seconds
    start_T = experiment_params['start_T']
    target_T = experiment_params['target_T']
    step_T = experiment_params['step_T']
    ramp_speed = (experiment_params['ramp_speed_c_min'] / 60) * config['experiment_frequency']  # Adjust per loop

    temperature_interp = interp1d(r_vs_t[0, :], r_vs_t[1, :], kind='linear', fill_value='extrapolate')

    # Initialize PID tuning parameters
    best_params = {'Kp': 0.02, 'Ki': 0.001, 'Kd': 0.002}  # Initial guess
    best_error = float('inf')

    target_T_tmp = (start_T + target_T) / 2 + step_T * 1
    for Kp in np.linspace(0.01, 0.1, 5):  # Sweep through potential Kp values
        for Ki in np.linspace(0.0005, 0.005, 5):
            for Kd in np.linspace(0.0005, 0.005, 5):
                pid_controller = pid.PIDController(Kp, Ki, Kd, setpoint=target_T_tmp)

                pid_voltage = 0.01  # Start with small voltage
                siglent.set_voltage(PS, voltage=pid_voltage)
                time.sleep(3)

                total_error = 0  # Track cumulative error
                prev_voltage = pid_voltage

                for _ in range(max_iter):
                    measured_voltage, measured_current, temperature = tds_experiment.measure_resistivity(
                        DMM_v, DMM_i, siglent, temperature_interp
                    )

                    error = target_T_tmp - temperature
                    pid_output = pid_controller.compute(temperature)

                    # Update voltage incrementally
                    pid_voltage = prev_voltage + pid_output
                    pid_voltage = max(0, min(20, pid_voltage))  # Clamp voltage
                    siglent.set_voltage(PS, voltage=pid_voltage)

                    total_error += abs(error)  # Sum absolute errors
                    prev_voltage = pid_voltage

                    time.sleep(loop_time)

                # Check if this PID setup is better
                if total_error < best_error:
                    best_error = total_error
                    best_params = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
    try:
        siglent.set_voltage(PS, voltage=0.0)
    except Exception as e:
        print(f"An error occurred in setting voltage: {e}")
    time.sleep(0.)
    DMM_v.close()
    DMM_i.close()
    siglent.set_output(PS, state='OFF')
    time.sleep(0.5)
    PS.close()
    rm.close()
    print(f"Best PID parameters: {best_params}")

    return best_params
