"""Pinpoint why SPD1000X CH1 cannot be switched ON remotely.

Safe by design: sets CH1 to 0.001 V / 0.010 A before it ever enables the output,
and leaves the output OFF when it finishes.

    python tools/psu_output_diag.py
"""
import sys
import time
import tomllib
from pathlib import Path

import pyvisa

STATUS_BITS = {
    0: ("CV mode", "CC mode"),
    4: ("Output OFF", "Output ON"),
    5: ("2W mode", "4W mode"),
    6: ("Timer OFF", "Timer ON"),
    8: ("digital display", "waveform display"),
}


def decode(status):
    return ", ".join(
        f"bit{bit}={names[(status >> bit) & 1]}" for bit, names in sorted(STATUS_BITS.items())
    )


def status_of(ps):
    raw = str(ps.query("SYSTem:STATus?", delay=0.05)).strip()
    return int(raw, 16), raw


def errors(ps):
    try:
        return str(ps.query("SYSTem:ERRor?", delay=0.05)).strip()
    except Exception as exc:
        return f"<query failed: {exc}>"


def step(ps, label, command, expect_on):
    print(f"\n--- {label}: sending {command!r}")
    ps.write(command)
    time.sleep(0.2)
    value, raw = status_of(ps)
    on = bool((value >> 4) & 1)
    print(f"    status={raw} -> {decode(value)}")
    print(f"    error queue: {errors(ps)}")
    verdict = "OK" if on is expect_on else "IGNORED BY INSTRUMENT"
    print(f"    output is {'ON' if on else 'OFF'} (expected {'ON' if expect_on else 'OFF'}) -> {verdict}")
    return on is expect_on


def main():
    config_path = Path(__file__).resolve().parents[1] / "files" / "config.toml"
    address = tomllib.loads(config_path.read_text(encoding="utf-8"))["PS"]

    rm = pyvisa.ResourceManager()
    print("VISA resources:", rm.list_resources())
    ps = rm.open_resource(address)
    ps.write_termination = "\n"
    ps.read_termination = "\n"
    ps.timeout = 5000

    try:
        print("\n*IDN? ->", ps.query("*IDN?", delay=0.05).strip())
        print("SYSTem:VERSion? ->", ps.query("SYSTem:VERSion?", delay=0.05).strip())
        value, raw = status_of(ps)
        print(f"initial status={raw} -> {decode(value)}")
        print("initial error queue:", errors(ps))

        print("\nSetting a harmless operating point (0.001 V, 0.010 A).")
        ps.write("CH1:VOLTage 0.001")
        time.sleep(0.1)
        ps.write("CH1:CURRent 0.010")
        time.sleep(0.1)

        step(ps, "1. switch OFF", "OUTP CH1,OFF", expect_on=False)
        on_without_unlock = step(ps, "2. switch ON (no *UNLOCK)", "OUTP CH1,ON", expect_on=True)

        if not on_without_unlock:
            print("\n--- 3. sending *UNLOCK, then retrying ON")
            ps.write("*UNLOCK")
            time.sleep(0.2)
            on_after_unlock = step(ps, "3. switch ON (after *UNLOCK)", "OUTP CH1,ON", expect_on=True)
            print("\n==> DIAGNOSIS:", "KEY LOCK was blocking remote output control."
                  if on_after_unlock else
                  "Not the key lock. The instrument accepts OUTP but does not act on it.")
        else:
            print("\n==> DIAGNOSIS: remote output enable works with this command sequence.")

        print("\nLeaving the output OFF.")
        ps.write("OUTP CH1,OFF")
        time.sleep(0.2)
        value, raw = status_of(ps)
        print(f"final status={raw} -> {decode(value)}")
    finally:
        ps.close()
        rm.close()


if __name__ == "__main__":
    sys.exit(main())
