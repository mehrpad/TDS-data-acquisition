# Measurement Setup: Kelvin Wiring and Fixed DMM Ranges

This software calculates sample resistance as `R = V / I`. That result is only the sample resistance when the voltage reading is taken with a true four-wire (Kelvin) connection.

## Constant-current operation

The supply is driven in constant current. At the start of every operation the software
sets `compliance_voltage` (default 30 V) as the CV ceiling, then commands current only.
The controlled variable everywhere - the linear ramp, the slew limits, the T0 and tuning
searches, the recovery probes - is current in amps.

Resistivity measurement wants a known, held current rather than a known voltage, which is
why the control variable is the current and the sample voltage is a reading.

Two consequences worth knowing:

- **A failing contact becomes a voltage runaway.** In constant voltage a bad joint simply
  passes less current. In constant current the supply raises its terminal voltage to hold
  the setpoint, up to the compliance limit. `max_sample_voltage` (default 15 V) aborts the
  run when the measured sample voltage passes it, which is the signature of an open or
  degrading contact well before `max_power_w` would react.
- **The SPD1000X programs current in 1 mA steps.** `minimum_current_change` is 0.001 A for
  that reason, and nothing finer reaches the instrument. At the 0.01 A starting point that
  is 10 % resolution, so the first few steps of a ramp are coarse.

`max_current` is both the highest current the controller may command and the software abort
threshold; the GUI `Max Current (A)` field sets it. With `max_power_w = 2.5 W` on a ~20 ohm
sample the power limit binds first, at about 0.35 A.

Configuration files from the constant-voltage version are migrated on load: `startup_voltage`
becomes `startup_current`, `max_voltage` becomes `compliance_voltage`, and so on. The numbers
are carried over unchanged, because volts do not convert to amps - check them before a run.

Controller gains are now amps per degree. **Re-run Tune PI/PID before any real experiment.**

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

## Resistivity measurement mode

The GUI `Resistivity Mode` field selects how resistance is obtained. All three modes
feed the same R-vs-T inversion, so T0 calibration, PI/PID tuning, and experiment runs
always use the mode currently selected.

| Mode | What it does | Wiring |
| --- | --- | --- |
| `V_OVER_I` | Continuous `R = V / I` while the heating current flows. | The Kelvin wiring above. |
| `OFFSET_CORRECTED` | Each cycle: heat, read `V_on`/`I_on`, switch CH1 off, read the sense voltage again. With no current flowing that second reading is the contact thermal EMF, so `R = (V_on - V_off) / I_on`. | Unchanged from the Kelvin wiring above. |
| `FOUR_WIRE` | Each cycle: heat, read `V_on`/`I_on`, switch CH1 off, and let the voltage DMM source its own test current for a direct `CONF:FRES` reading. | The voltage DMM additionally needs its **Input HI/LO** source leads on the sample, not just **Sense HI/LO**. |

`V_OVER_I` measures the sample while it is being heated, so any contact thermal EMF
adds directly to a sense voltage that is only a few millivolts. On a 2-5 ohm sample
run to several hundred degrees, that offset is the largest error in the measurement.
The other two modes remove it by measuring while no heating current flows.

### Duty cycle

The two duty-cycled modes set the control-loop period themselves from
`resistivity_heat_time_s + resistivity_measure_time_s` (default 4 s + 1 s = 5 s);
`experiment_frequency` does not apply to them. Within each cycle the supply heats for
`resistivity_heat_time_s`, then CH1 switches off, `resistivity_output_settle_s` elapses,
the quiet reading is taken, and CH1 switches back on.

Because the sample only heats for part of each cycle, controller gains tuned under
`V_OVER_I` will be too weak. Re-run **Tune PI/PID** after changing the mode.

The per-loop voltage slew limits (`max_voltage_step_up`, `low_voltage_max_step_up`, and
the catch-up steps) are scaled by the ratio of the active cycle to the
`1 / experiment_frequency` period they were chosen for, so the achievable volts-per-minute
is the same whatever cycle you configure. Lengthening the cycle therefore costs control
bandwidth - corrections arrive less often, and the setpoint moves further between them -
but not ramp rate. At 30 C/min a 5 s cycle lets the setpoint advance 2.5 C per correction;
a 12 s cycle lets it advance 6.4 C, which will not hold a 2 C tolerance.

In `OFFSET_CORRECTED` and `FOUR_WIRE` the reported resistance is the one measured
with the heating current off. `V` and `I` remain on their own displays for monitoring,
but `V / I` is no longer the resistance and is not used as one.

A resistance that falls outside the loaded R-vs-T table is still a valid measurement:
only its conversion to temperature is unavailable. T0 calibration depends on this,
because its whole job is to anchor a curve whose absolute scale does not yet match the
sample. If the four-wire resistance sits outside the table by more than T0 can absorb,
re-measure the R-vs-T curve in the same mode.

`FOUR_WIRE` uses the fixed `dmm_resistance_range_ohm` range. The SDM3055's lowest
four-wire range is 200 ohm, so a low-resistance sample sits near the bottom of it;
repeatability rather than absolute accuracy is what carries the temperature inversion,
and the R-vs-T calibration must be taken in the same mode.

### Remote output control

The duty-cycled modes require the SPD1000X to accept `OUTPut CH1,ON` over USB. If the
instrument's key lock is engaged the supply accepts that command and ignores it, so the
software verifies every switch against `SYSTem:STATus?` bit 4 and sends `*UNLOCK` before
retrying. Clear the lock from the front panel by holding `Ver/Lock` until the lock icon
disappears. `python tools/psu_output_diag.py` reports what the instrument is doing.

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
