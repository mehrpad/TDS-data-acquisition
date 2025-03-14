import time
import datetime

import numpy as np
import pyvisa
from scipy.interpolate import interp1d

import siglent, pid  # Assuming your custom module for SDM3055 functions

def tds(emitter, experiment_params, r_vs_t, config, t_zero):

    def update_max_current(max_current):
        config['max_current'] = max_current

    def update_max_voltage(max_voltage):
        config['max_voltage'] = max_voltage

    # Connect the signal to the slot
    emitter.max_current_signal.connect(update_max_current)
    emitter.max_voltage_signal.connect(update_max_voltage)

    # Initialize Resource Manager and Devices
    rm = pyvisa.ResourceManager()
    DMM_v = rm.open_resource(config['DMM_v'])  # Digital Multimeter for voltage
    DMM_i = rm.open_resource(config['DMM_i'])  # Digital Multimeter for current
    PS = rm.open_resource(config['PS'])  # Power Supply
    PS.write_termination = '\n'
    PS.read_termination = '\n'
    siglent.set_output(PS, state='ON')
    time.sleep(0.04)
    siglent.set_voltage(PS, voltage=0.0)
    time.sleep(1)
    # Set the speed and mode of the DMMs
    siglent.set_mode_speed(DMM_i, 'CURR', config['DMM_speed'])
    siglent.set_mode_speed(DMM_v, 'VOLT', config['DMM_speed'])
    loop_time = 1 / config['experiment_frequency']  # Loop time in seconds
    temperature_interp = interp1d(r_vs_t[0, :], r_vs_t[1, :], kind='linear', fill_value='extrapolate')
    for ex_param in experiment_params:
        print('Experiment parameters: ', ex_param)
        start_T = ex_param['start_T']
        step_T = ex_param['step_T']
        target_T = ex_param['target_T']
        ramp_speed = (ex_param['ramp_speed_c_min'] / 60) # Convert from per minute to per second
        ramp_speed =  ramp_speed  * config['experiment_frequency'] # Convert from per second to per loop
        hold_step_min = ex_param['hold_step_time_min']
        hold_step_counter = 0
        hold_step__time_counter = 0
        old_pid_voltage = 0
        loop_counter = 0

        pid_voltage = 0.01
        siglent.set_voltage(PS, voltage=pid_voltage)
        time.sleep(3)
        measured_voltage, measured_current, temperature = measure_resistivity(DMM_v, DMM_i, siglent,
                                                                              temperature_interp)
        print(f"The initial measured temperature is {temperature} - T zero is {t_zero} - R zero "
              f"is {measured_voltage/measured_current}")
        target_T_temp = start_T + step_T * hold_step_counter
        # Initialize PID controller for incremental PID control
        # New Input = Old Input + PID Output
        pid_controller = pid.PIDController(kp=0.001, ki=0.0, kd=0.0, setpoint=t_zero)
        step_start_temp = (target_T_temp - temperature) / 100
        ini_start_temp = 0
        while temperature < start_T and not emitter.stopped:
            pid_voltage_dz = pid_controller.compute(temperature, setpoint=t_zero + ini_start_temp)
            pid_voltage_dz = max(0.005, min(pid_voltage_dz, 0.1))
            pid_voltage += pid_voltage_dz
            pid_voltage = max(0.0, min(pid_voltage, config['max_voltage']))
            if ini_start_temp < target_T_temp:
                if t_zero + ini_start_temp < temperature:
                    ini_start_temp = temperature - t_zero
                else:
                    ini_start_temp += step_start_temp
            pid_voltage = max(0.005, min(pid_voltage, config['max_voltage']))
            siglent.set_voltage(PS, voltage=pid_voltage)
            time.sleep(0.9)
            measured_voltage, measured_current, temperature = measure_resistivity(DMM_v, DMM_i, siglent,
                                                                                  temperature_interp)

            current_time = datetime.datetime.now()
            current_time_with_microseconds = current_time.strftime(
                "%Y-%m-%d %H:%M:%S.%f")  # Format with microseconds
            current_time_unix = datetime.datetime.strptime(current_time_with_microseconds,
                                                           "%Y-%m-%d %H:%M:%S.%f").timestamp()
            emitter.experiment_signal.emit([current_time_unix, target_T_temp, temperature,
                                            0, measured_voltage, measured_current, pid_voltage])
            # print(f"Start phase: Temperature: {temperature}, Voltage: {pid_voltage}")
        print(f"The start temperature is reached: {temperature}")
        while not emitter.stopped:
            start_time_loop = time.time()
            # print(f"Temperature: {temperature}, Voltage: {pid_voltage}, Target temperature: {target_T_temp}")
            if target_T_temp > target_T:
                print('The temperature is higher than the target temperature. The experiment is finished.')
                break

            if target_T_temp >= start_T + step_T * hold_step_counter:
                hold_step__time_counter += 1
                # print(f"Hold step time counter: {hold_step__time_counter}")
                pid_voltage += pid_controller.compute(temperature, setpoint=target_T_temp)
                pid_voltage = max(0.005, min(pid_voltage, config['max_voltage']))

                if hold_step__time_counter * config['experiment_frequency'] >= hold_step_min * 60:
                    hold_step_counter += 1
                    hold_step__time_counter = 0
                    loop_counter = 0
            else:
                # print(f"Ramp speed: {ramp_speed}, Loop counter: {loop_counter}")
                target_T_temp = start_T + step_T * (hold_step_counter - 1) + ramp_speed * loop_counter
                pid_voltage += pid_controller.compute(temperature, setpoint=target_T_temp)
                pid_voltage = max(0.005, min(pid_voltage, config['max_voltage']))

            # Apply the calculated voltage if it is different from the previous value
            if pid_voltage != old_pid_voltage:
                try:
                    old_pid_voltage = pid_voltage
                    siglent.set_voltage(PS, voltage=pid_voltage)
                except Exception as e:
                    print(f"An error occurred in setting voltage: {e}")
            time.sleep(0.1)
            # Measure resistivity and update temperature
            measured_voltage, measured_current, temperature = measure_resistivity(DMM_v, DMM_i, siglent,
                                                                                  temperature_interp)
            current_time = datetime.datetime.now()
            current_time_with_microseconds = current_time.strftime(
                "%Y-%m-%d %H:%M:%S.%f")  # Format with microseconds
            current_time_unix = datetime.datetime.strptime(current_time_with_microseconds,
                                                           "%Y-%m-%d %H:%M:%S.%f").timestamp()
            emitter.experiment_signal.emit([current_time_unix, target_T_temp, temperature,
                                            0, measured_voltage, measured_current, pid_voltage])

            loop_elapsed_time = time.time() - start_time_loop
            if loop_elapsed_time < loop_time:
                time.sleep(loop_time - loop_elapsed_time)
            elif loop_elapsed_time > loop_time:
                print(f"Loop time exceeded: {loop_elapsed_time}")

            loop_counter += 1

    try:
        siglent.set_voltage(PS, voltage=0.0)
    except Exception as e:
        print(f"An error occurred in setting voltage: {e}")
    time.sleep(0.1)
    DMM_v.close()
    DMM_i.close()
    siglent.set_output(PS, state='OFF')
    time.sleep(0.5)
    PS.close()
    rm.close()
    print('TDS experiment thread finished.')

def measure_resistivity(DMM_v, DMM_i, siglent, temperature_interp):
    try:
        measured_voltage = float(siglent.read_DMM(DMM_v))
    except Exception as e:
        print(f"An error occurred reading voltage DMM: {e}")
        measured_voltage = np.nan
    try:
        measured_current = float(siglent.read_DMM(DMM_i))
    except Exception as e:
        print(f"An error occurred reading current DMM: {e}")
        measured_current = np.nan
    if measured_current != 0:
        resistance = measured_voltage / measured_current
        temperature = temperature_interp(resistance).item()
    else:
        temperature = np.nan
    if temperature < 0:
        print(f"Calculated temperature is : {temperature} - put temperature to 0")
        temperature = 0
    return measured_voltage, measured_current, temperature
