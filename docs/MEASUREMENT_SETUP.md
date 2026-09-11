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

1. sends the small `psu_keepalive_current` setpoint and enables CH1 once,
2. selects the explicit fixed `dmm_voltage_range_v` and `dmm_current_range_a`,
3. configures the DMM integration speed, and
4. requires a stable startup resistance median before enabling control.

The defaults are `20 V DC` and `0.02 A DC`. These are sized to the two ends of a typical run rather than to the smallest instrument range: at the `~10 W`/`15 V` operating point (`max_power_w`, `max_sample_voltage`), the sample settles near `0.667 A` at `~22.5 ohm`, which is `75%` of the `20 V` range and `33%` of a `2 A` range - both stay put, with no unwanted escalation to the much coarser `200 V`/`10 A` ranges. But `T0`/`startup` begin at `0.005 A`, which would be `0.25%` of a `2 A` range - deep enough in the range's noise floor that the `% of range` error term in the DMM's own accuracy spec dominates the reading, corrupting the resistance `T0` anchors on. Starting the current range at `0.02 A` instead puts that same `5 mA` reading at `25%` of scale. Staged ranging (`dmm_staged_ranging_enabled`) still applies from there: it only ever steps up, never autoranges, and never steps back down mid-run, so the climb from `0.02 A` to the `0.667 A` operating point costs only two range changes (`0.02 -> 0.2 -> 2 A`) over the whole run.

The upward threshold is `dmm_range_switch_fraction` (default `0.8`, i.e. escalate at `80%` of the active range) - except on the `0.02 A` range itself, where the per-loop current step (`max_current_step_up`, default `0.01 A`) is half the entire range. Waiting for `80%` there (`0.016 A`) would leave less than one step of headroom: a single normal step could jump straight past `0.02 A` before the range has caught up. The threshold on that range is pulled down to `50%` (`0.01 A`) instead, specifically so the escalation lands a full step ahead of the jump rather than reacting to it. Ranges much larger than the step - `0.2 A`, `2 A`, and the voltage ranges - are unaffected, since the flat `80%` fraction already leaves ample headroom there. `max_current` and `max_sample_voltage` remain separate software safety limits, independent of the DMM range.

`DMM_speed = 10` uses the slow 10-NPLC integration setting. The two meters are read sequentially, so the control-loop period must be long enough for both readings and instrument communication. The default `experiment_frequency = 1` Hz normally provides that margin. A lower frequency can provide more settling time after a current change, but it also slows controller response and does not make Auto Range safe.

The GUI `Initial Current (A)` is an enforced controlled-experiment floor as well as the starting value for T0, tuning, and curve sweep (default `0.005 A`; if that does not produce a stable reading, T0's own search steps upward from there). Startup requires five consistent resistance readings by default. If necessary, it searches upward in `0.001 A` steps, and the stable current it finds becomes the active floor. A stable inferred temperature more than `startup_temperature_margin_c` above `max(T0, start_T)` stops startup rather than beginning control from an implausibly hot reading.

At PSU voltages up to `low_voltage_step_threshold` (default `0.05 V`), normal control, invalid-reading recovery, T0 search, and tuning search are restricted to `0.001 V` changes. This avoids alternating directly between `0.01 V` and `0.02 V` on a sensitive wire.

T0 and tuning baseline searches require resistance stability as well as current stability. T0 also checks the final calibration samples before accepting the scale, so noisy readings cannot silently become a misleading low-TCR calibration.

Ordinary current updates do not resend the PSU `ON` command. At the end of T0 calibration, PI/PID tuning, and an experiment, the software zeroes the current and switches CH1 off.

## PI/PID gain schedule, step-limit suggestions, and material profiles

A wire's process gain (temperature rise per amp) is not constant from a near-zero-power tuning point up to the several-watt point of a real heating run - self-heating shifts which loss mechanism (conduction vs. radiation) dominates. **Tune PI/PID** in the GUI runs the step-response test at three currents - `tuning_start_current`, 40%, and 80% of `max_current` (fewer if `max_current` is too small to separate them) - and builds a schedule of `{current_a, Kp, Ki, Kd}` points instead of one fixed gain set. During a run, the controller interpolates by the present current, clamping at the ends of the schedule rather than extrapolating past it.

Each tuning point also derives a suggested current step limit from its identified process gain and time constant: the lowest point's suggestion becomes `low_current_max_step_up/down`, the highest point's becomes `max_current_step_up/down` (representative of sustained real operation). These are suggestions, not guarantees - they bound how much of a step's effect can appear within one control loop period, not sustained ramp rate, which the existing rate-limiting logic still governs separately. Review them before a run; both are editable fields in the GUI next to the PID gains.

Editing Kp, Ki, or Kd by hand clears the gain schedule (the flat value would otherwise be silently overridden by the schedule during a run); tuning again rebuilds one.

**Material Profiles** save a named snapshot of the gain schedule, step limits, and the other settings that go with a specific sample - resistivity mode, DMM ranges, current/power/voltage limits - to `files/material_profiles/<name>.json`, separate from `config.toml`. Tune once per material, save a profile, and load it in a future session instead of re-tuning. Loading a profile overwrites the corresponding live settings immediately; save your current state as its own profile first if you want to keep it.

## Pre-run checklist

- Four distinct sample contacts are used: two force and two sense.
- `V+` and `V-` land on the sample itself, not cable ends or supply terminals.
- Both DMMs show fixed DC ranges, not Auto Range.
- The configured `dmm_voltage_range_v` and `dmm_current_range_a` cover the expected sample signals.
- `Initial Current (A)` is low enough that the equilibrated sample begins near T0/start temperature.
- `compliance_voltage`, `max_current`, and `max_sample_voltage` are positive, conservative, and within the DMM's supported fixed ranges.
- The power supply has its own independent current limit/OCP configured. The application's `max_current` is a software shutdown threshold, not a hardware current clamp.
