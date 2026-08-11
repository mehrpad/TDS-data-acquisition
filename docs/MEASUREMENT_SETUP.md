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

## DMM range policy

Auto Range is not permitted for either DMM. It can insert range-change delays and transient readings that corrupt resistance calculations and the temperature-control loop.

Before the PSU output is enabled, the software:

1. switches the PSU output off and sends its setpoint to `0 V`,
2. selects the smallest supported fixed DC voltage range that covers `max_voltage`,
3. selects the smallest supported fixed DC current range that covers `max_current`, and
4. configures the DMM integration speed.

For the repository defaults of `max_voltage = 30` and `max_current = 3`, the selected fixed ranges are `200 V DC` and `10 A DC`. Confirm the limits and meter ranges before every experiment; never re-enable Auto Range on the DMM front panel during a run.

## Pre-run checklist

- Four distinct sample contacts are used: two force and two sense.
- `V+` and `V-` land on the sample itself, not cable ends or supply terminals.
- Both DMMs show fixed DC ranges, not Auto Range.
- `max_voltage` and `max_current` are positive, conservative, and within the DMM's supported fixed ranges.
- The power supply has its own independent current limit/OCP configured. The application's `max_current` is a software shutdown threshold, not a hardware current clamp.