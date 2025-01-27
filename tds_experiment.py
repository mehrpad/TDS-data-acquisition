import time
import datetime

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

import siglent  # Assuming your custom module for SDM3055 functions

def tds(emitter, experiment_params, r_vs_t, freq_acquisition, test_mode=False):
    # Initialize Resource Manager and Devices
    rm = pyvisa.ResourceManager()
    DMM_v = rm.open_resource('USB0::0xF4EC::0xEE38::SDM35FAC4R0253::INSTR')  # Digital Multimeter
    DMM_i = rm.open_resource('USB0::0xF4EC::0x1201::SDM35HBQ803105::INSTR')  # Digital Multimeter
    PS = rm.open_resource('USB0::0xF4EC::0x1410::SPD13DCC4R0058::INSTR')  # Power Supply
    PS.write_termination = '\n'
    PS.read_termination = '\n'
    siglent.set_output(PS, state='ON')
    time.sleep(0.04)
    siglent.set_voltage(PS, voltage=0.0)
    loop_time = 1 / freq_acquisition  # Loop time in seconds
    temperature_interp = interp1d(r_vs_t[0, :], r_vs_t[1, :], kind='linear', fill_value='extrapolate')

    # PID parameters
    Kp = 0.001  # Proportional gain
    Ki = 0.005  # Integral gain
    # Kd = 0.01  # Derivative gain

    for ex_param in experiment_params:
        start_loop_time = time.time()
        print('Experiment parameters: ', ex_param)
        start_T = ex_param['start_T']
        step_T = ex_param['step_T']
        target_T = ex_param['target_T']
        ramp_speed = ex_param['ramp_speed_c_min']
        hold_step = ex_param['hold_step_time_min']
        calculated_voltage = 0.003
        siglent.set_voltage(PS, voltage=calculated_voltage)
        time.sleep(1)
        measured_voltage, measured_current, temperature = measure_resistivity(DMM_v, DMM_i, siglent,
                                                                              temperature_interp)

        # PID initialization
        integral = 0
        # previous_error = 0

        target_T_temp = start_T + step_T
        hold_step_counter = 0
        old_calculated_voltage = 0
        start_preparation_flag = True
        start_time = time.time()
        while not emitter.stopped:
            start_time_loop = time.time()
            time_elapsed = time.time() - start_time
            desired_temperature = min(start_T + (ramp_speed * time_elapsed),
                                      target_T_temp)  # limit to the target temperature
            print(f"Desired temperature: {desired_temperature}, Current temperature: {temperature}")
            error = desired_temperature - temperature
            integral += error * loop_time
            #derivative = (error - previous_error) / loop_time
            #previous_error = error
            pid_voltage = Kp * error + Ki * integral # + Kd * derivative
            print(f"PI Voltage: {pid_voltage}, Proportional: {Kp * error} Integral: {Ki * integral}")
            if 0.003 < pid_voltage < 20:
                calculated_voltage = pid_voltage

            # Voltage limiting to safe values
            # calculated_voltage = max(0.0, min(calculated_voltage, 10.0))  # Example limits: 0 to 10 V

            if start_preparation_flag:
                # First reach the start temperature
                if temperature < start_T:
                    pass
                elif temperature > target_T:
                    print('The temperature is higher than the target temperature. The experiment is finished.')
                    break
                else:
                    start_preparation_flag = False
            elif temperature >= target_T_temp:
                # Wait for the hold step time
                hold_step_counter += 1
                if (hold_step_counter / freq_acquisition) / 60 >= hold_step:
                    target_T_temp += step_T
                    if target_T_temp >= target_T:
                        break
                    hold_step_counter = 0

            # Apply the calculated voltage if it is different from the previous value
            if calculated_voltage != old_calculated_voltage:
                try:
                    old_calculated_voltage = calculated_voltage
                    siglent.set_voltage(PS, voltage=calculated_voltage)
                except Exception as e:
                    print(f"An error occurred in setting voltage: {e}")
            time.sleep(0.01)
            # Measure resistivity and update temperature
            measured_voltage, measured_current, temperature = measure_resistivity(DMM_v, DMM_i, siglent,
                                                                                  temperature_interp)

            elapsed_time = time.time() - start_time_loop
            if elapsed_time < loop_time:
                time.sleep(loop_time - elapsed_time)
            current_time = datetime.datetime.now()
            current_time_with_microseconds = current_time.strftime(
                "%Y-%m-%d %H:%M:%S.%f")  # Format with microseconds
            current_time_unix = datetime.datetime.strptime(current_time_with_microseconds,
                                                           "%Y-%m-%d %H:%M:%S.%f").timestamp()
            emitter.experiment_signal.emit([current_time_unix, target_T, temperature,
                                            0, measured_voltage, measured_current, calculated_voltage])

            loop_elapsed_time = time.time() - start_loop_time
            if loop_elapsed_time < loop_time:
                time.sleep(loop_time - loop_elapsed_time)
            elif loop_elapsed_time > loop_time:
                print(f"Loop time exceeded: {loop_elapsed_time}")

    try:
        siglent.set_voltage(PS, voltage=0.0)
    except Exception as e:
        print(f"An error occurred in setting voltage: {e}")
    time.sleep(0.01)
    DMM_v.close()
    DMM_i.close()
    siglent.set_output(PS, state='OFF')
    time.sleep(1)
    PS.close()
    rm.close()
    print('TDS experiment thread finished.')

def measure_resistivity(DMM_v, DMM_i, siglent, temperature_interp):
    try:
        measured_voltage = float(siglent.measV(DMM_v, 'DC'))
    except Exception as e:
        print(f"An error occurred reading voltage DMM: {e}")
        measured_voltage = np.nan
    try:
        measured_current = float(siglent.measI(DMM_i, 'DC'))
    except Exception as e:
        print(f"An error occurred reading current DMM: {e}")
        measured_current = np.nan
    if measured_current != 0:
        resistance = measured_voltage / measured_current
        temperature = temperature_interp(resistance).item()
    else:
        temperature = np.nan

    return measured_voltage, measured_current, temperature
