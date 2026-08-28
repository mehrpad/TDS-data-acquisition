# Measurement Setup: Kelvin Wiring and Fixed DMM Ranges

This software calculates sample resistance as `R = V / I`. That result is only the sample resistance when the voltage reading is taken with a true four-wire (Kelvin) connection.

## Required wiring

```text
                                  CURRENT PATH
PS+ -- Ammeter -- force cable -- o======== SAMPLE ========o -- force cable -- PS-
                                  |                        |
                              V+ sense                  V- sense
                                  \                        /
                                   \---- voltage DMM -----/
                                      high-impedance sense wires
```

1. Connect `PS+` to the ammeter and connect the ammeter output to the sample's positive **force** contact.
2. Connect the sample's negative **force** contact to `PS-`.
3. Connect the voltage DMM's `V+` and `V-` directly to separate sample **sense** contacts. Put them inside the current contacts when the fixture allows it.
4. Keep the sense wires separate from the force path. The voltage DMM draws negligible current, so their resistance has negligible effect on the reading.

The current DMM must stay in series. Never connect an ammeter in parallel with the sample.

## Do not use cable-end/two-wire sensing

This is not a Kelvin measurement:

```text
PS+ -- Ammeter -- cable -- SAMPLE -- cable -- PS-
                   |                 |
                   +--- voltage DMM--+
```

It measures cable and contact drops as well as the sample:

```text
R_measured = R_sample + R_contacts + R_cables
```

With correct Kelvin contacts, the voltage DMM instead measures approximately the sample drop:

```text
V_DMM ≈ I × R_sample
R_sample = V_DMM / I
```

Use an `R vs. T` calibration obtained with the same Kelvin setup. Leave `fixed_series_resistance_ohm = 0` unless there is a known, independently verified external series resistor that should be removed from the result.

## DMM range and low-voltage startup policy

Auto Range is not permitted for either DMM. It can insert range-change delays and transient readings that corrupt resistance calculations and the temperature-control loop.

At the start of T0 calibration, controller tuning, or an experiment, the software:

1. sends the small `psu_keepalive_voltage` setpoint and enables CH1 once,
2. selects the explicit fixed `dmm_voltage_range_v` and `dmm_current_range_a`,
3. configures the DMM integration speed, and
4. requires a stable startup resistance median before enabling control.

The defaults are `0.2 V DC` and `0.2 A DC`, the smallest supported SDM3055 DC ranges in this application. They are appropriate for the millivolt/milliamp signals in the current setup and give much better resolution than the former `200 V`/`10 A` ranges selected from `max_voltage = 30` and `max_current = 3`. If an expected sample signal can exceed a default range, increase that explicit DMM range before running. `max_voltage` and `max_current` remain separate software safety limits.

`DMM_speed = 10` uses the slow 10-NPLC integration setting. The two meters are read sequentially, so the control-loop period must be long enough for both readings and instrument communication. The default `experiment_frequency = 1` Hz normally provides that margin. A lower frequency can provide more settling time after a voltage change, but it also slows controller response and does not make Auto Range safe.

The GUI `Initial Voltage` is an enforced controlled-experiment floor as well as the starting value for T0, tuning, and curve sweep. Startup requires five consistent resistance readings by default. If necessary, it searches upward in `0.001 V` steps, and the stable voltage it finds becomes the active floor. A stable inferred temperature more than `startup_temperature_margin_c` above `max(T0, start_T)` stops startup rather than beginning control from an implausibly hot reading.

At PSU voltages up to `low_voltage_step_threshold` (default `0.05 V`), normal control, invalid-reading recovery, T0 search, and tuning search are restricted to `0.001 V` changes. This avoids alternating directly between `0.01 V` and `0.02 V` on a sensitive wire.

T0 and tuning baseline searches require resistance stability as well as current stability. T0 also checks the final calibration samples before accepting the scale, so noisy readings cannot silently become a misleading low-TCR calibration.

Ordinary voltage updates do not resend the PSU `ON` command. At the end, the software returns to `psu_keepalive_voltage` and intentionally leaves CH1 enabled.

## Pre-run checklist

- Four distinct sample contacts are used: two force and two sense.
- `V+` and `V-` land on the sample itself, not cable ends or supply terminals.
- Both DMMs show fixed DC ranges, not Auto Range.
- The configured `dmm_voltage_range_v` and `dmm_current_range_a` cover the expected sample signals.
- `Initial Voltage` is low enough that the equilibrated sample begins near T0/start temperature.
- `max_voltage` and `max_current` are positive, conservative, and within the DMM's supported fixed ranges.
- The power supply has its own independent current limit/OCP configured. The application's `max_current` is a software shutdown threshold, not a hardware current clamp.
