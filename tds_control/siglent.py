import time
import pyvisa
from scipy.interpolate import interp1d


def measV(DMM, acdc):
    cmd1 = f'MEAS:VOLT:{acdc}? AUTO'
    return float(DMM.query(cmd1))  # No need to parse manually

def measI(DMM, acdc):
    cmd1 = f'MEAS:CURR:{acdc}? AUTO'
    return float(DMM.query(cmd1))  # No need to parse manually

# Function to set voltage on the power supply
def set_voltage(ps, voltage):
    # SCPI command to set voltage
    ps.write(f"VOLT {voltage}")


def set_output(ps, state):
    # SCPI command to turn on output
    ps.write(f"OUTP CH1,{state}")

def read_current(ps):
    # SCPI command to read current
    current = ps.query(f"MEASure:CURRent?", delay=0.01)
    return current

def set_mode_speed(DMM, mode, speed):
    # SCPI command to set speed
    DMM.write(f"{mode}:DC:NPLC {speed}")


def configure_dc_range(DMM, mode, range_value):
    mode = str(mode).strip().upper()
    if mode not in {"VOLT", "CURR"}:
        raise ValueError(f"Unsupported DMM mode for range configuration: {mode}")

    if range_value is None:
        return

    if isinstance(range_value, str) and range_value.strip().upper() == "AUTO":
        DMM.write(f"CONF:{mode}:DC AUTO")
        return

    try:
        numeric_range = float(range_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid DMM range value for {mode}: {range_value!r}") from exc

    DMM.write(f"CONF:{mode}:DC {numeric_range}")


def read_DMM(DMM):
    # SCPI command to read
    return DMM.query("READ?")

if __name__ == "__main__":
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    DMM_v = rm.open_resource('USB0::0xF4EC::0xEE38::SDM35FAC4R0253::INSTR')  # Digital Multimeter
    DMM_i = rm.open_resource('USB0::0xF4EC::0x1201::SDM35HBQ803105::INSTR')  # Digital Multimeter
    PS = rm.open_resource('USB0::0xF4EC::0x1410::SPD13DCC4R0058::INSTR')  # Power Supply
    PS.write_termination = '\n'
    PS.read_termination = '\n'
    # Set the voltage
    time.sleep(0.04)
    set_output(PS, state='ON')
    set_voltage(PS, voltage=0.5)
    time.sleep(1)
    # On the front panel,0.3|1|10 corresponds to the Speed menu under Fast|Middle|Slow respectively
    set_mode_speed(DMM_i, 'CURR', 1)
    set_mode_speed(DMM_v, 'VOLT',1)
    start_time = time.time()
    for i in range(10):
        # set_voltage(PS, voltage=0.5+0.1*i)
        set_voltage(PS, voltage=0.01)
        time.sleep(0.3)
        measured_voltage = float(read_DMM(DMM_v))
        measured_current = float(read_DMM(DMM_i))
        print(measured_voltage/measured_current)
        print(f"Voltage: {measured_voltage} V, Current: {measured_current} A, Applied Voltage: {0.5+0.1*i} V")
        time.sleep(2)
    print(f"Time taken: {time.time() - start_time}")

    # print(float(read_current(PS)))
    set_voltage(PS, voltage=0.0)
    set_output(PS, state='OFF')
    time.sleep(1)
    PS.close()
    print('DONE')



