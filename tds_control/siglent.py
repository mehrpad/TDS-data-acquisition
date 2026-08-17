import math
import time
import pyvisa
from scipy.interpolate import interp1d


_SDM3055_DC_RANGES = {
    "VOLT": (0.2, 2.0, 20.0, 200.0, 1000.0),
    "CURR": (0.2, 2.0, 10.0),
}


def _pick_sdm3055_dc_range(expected_max, allowed_ranges):
    try:
        value = float(expected_max)
    except (TypeError, ValueError):
        return None
    if value <= 0 or not math.isfinite(value):
        return None
    for candidate in allowed_ranges:
        if value <= candidate:
            return candidate
    return None

# Function to set voltage on the power supply
def set_voltage(ps, voltage):
    """Set the PSU voltage and make sure CH1 output is enabled when needed."""
    numeric_voltage = float(voltage)
    ps.write(f"VOLT {numeric_voltage}")

    # An operator can turn the output off on the front panel between runs.
    # Reassert the output state before every non-zero setpoint so all experiment
    # paths recover automatically without energising the supply for a 0 V command.
    if abs(numeric_voltage) > 0:
        set_output(ps, state="ON")


def set_output(ps, state):
    """Set the Siglent SPD CH1 output state using the documented SCPI command."""
    normalized_state = str(state).strip().upper()
    if normalized_state not in {"ON", "OFF"}:
        raise ValueError(f"Unsupported power-supply output state: {state!r}")

    command = f"OUTP CH1,{normalized_state}"
    try:
        ps.write(command)
    except Exception as exc:
        raise RuntimeError(f"Could not set power-supply CH1 output {normalized_state}.") from exc

    # Siglent recommends allowing a short delay after single write commands.
    time.sleep(0.05)
    print(f"Power supply CH1 output command sent: {normalized_state}")

def read_current(ps):
    # SCPI command to read current
    current = ps.query(f"MEASure:CURRent?", delay=0.01)
    return current

def set_mode_speed(DMM, mode, speed):
    # SCPI command to set speed
    DMM.write(f"{mode}:DC:NPLC {speed}")


def configure_dc_range(DMM, mode, range_value):
    mode = str(mode).strip().upper()
    if mode not in _SDM3055_DC_RANGES:
        raise ValueError(f"Unsupported DMM mode for range configuration: {mode}")

    if isinstance(range_value, str) and range_value.strip().upper() == "AUTO":
        raise ValueError(
            "DMM auto-ranging is disabled for resistivity measurements. "
            "Use a supported fixed DC range."
        )

    try:
        numeric_range = float(range_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid DMM range value for {mode}: {range_value!r}") from exc

    allowed_ranges = _SDM3055_DC_RANGES[mode]
    if not math.isfinite(numeric_range) or numeric_range not in allowed_ranges:
        allowed_text = ", ".join(str(value) for value in allowed_ranges)
        raise ValueError(
            f"Unsupported fixed {mode} DC range {range_value!r}. "
            f"Supported ranges are: {allowed_text}."
        )

    DMM.write(f"CONF:{mode}:DC {numeric_range}")
    return numeric_range


def configure_dc_range_from_limits(DMM, mode, expected_max):
    mode = str(mode).strip().upper()
    if mode not in _SDM3055_DC_RANGES:
        raise ValueError(f"Unsupported DMM mode for range configuration: {mode}")

    allowed_ranges = _SDM3055_DC_RANGES[mode]
    picked_range = _pick_sdm3055_dc_range(expected_max, allowed_ranges)
    if picked_range is None:
        raise ValueError(
            f"{mode} limit {expected_max!r} is not a positive finite value within the "
            f"largest supported fixed range ({allowed_ranges[-1]}). "
            "DMM auto-ranging is disabled."
        )
    return configure_dc_range(DMM, mode, picked_range)


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



