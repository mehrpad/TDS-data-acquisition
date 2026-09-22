# Temperature control

Temperature mode commands absolute current:

`I_requested = I_feedforward(T_set) + Kp*(T_set - T_measured) + integral_correction_A - prediction_correction_A`

The integral error is always T_set minus T_measured, independent of prediction.
The integral increases by `Ki*error*actual_elapsed_seconds` once per accepted
measurement. The previous current is used for actuator slew limits, not added
to the PI output. Current and power safety aborts remain active.

## Feed-forward and gain schedules

The GUI has an editable **Integral time Ti = Kp/Ki (s)** field beneath the
gain/current-step row. Larger Ti slows integration. Editing Ti calculates
`Ki = Kp/Ti`; entering zero disables integral action. Editing Kp or Ki directly
recalculates the displayed Ti. Manual changes to Ti or gains clear an existing
gain schedule, so that schedule cannot silently override the new settings.

`pid_integral_time_s` is saved in both `files/config.toml` and Material Profile
JSON files. For example, these equivalent settings produce Ki = 0.000002:

```json
{
  "pid_kp": 0.0002,
  "pid_ki": 0.000002,
  "pid_integral_time_s": 100
}
```

On file/profile load, explicit Ti takes precedence over flat Ki. You can edit
the JSON's Ti to change integration speed, or omit Ti to use a legacy Ki-only
file. Older profiles derive Ti from their existing gains without changing Ki.
For I-only control (`Kp = 0`, `Ki > 0`), Ti is undefined: the GUI shows N/A and
you edit Ki directly. Saved Ti is zero in that case while Ki is retained.

With a gain schedule enabled, the flat Ti is the fallback value; each scheduled
point's Ti is displayed alongside its gains and may differ.

Add equilibrium measurements for the actual wire to `files/config.toml` using
`current_feedforward_table`. Each point contains `temperature_c` and `current_a`.
For example, the **format only**, with illustrative values that must be replaced:

```toml
current_feedforward_table = [
  { temperature_c = 100.0, current_a = 0.05 },
  { temperature_c = 200.0, current_a = 0.08 },
]
```

Current is interpolated between measured points and held at the endpoints
outside the table. There is no feed-forward extrapolation. With an empty table,
the measurement-current floor is the bias and the integral supplies the rest of the heating
current. P-only control needs an appropriate bias/map to reach elevated targets.
Do not treat current measured during a ramp as equilibrium current.

`Tune PI/PID` saves measured `pid_gain_schedule` points with `current_a`, `kp`,
`ki`, and `kd`. The schedule is evaluated at feed-forward current when a map is
present, or at filtered accepted current otherwise. `gain_schedule_filter_time_s`
defaults to 10 s. Changes to proportional gain preserve the integral-supported
holding correction. Ramp/hold transitions preserve that correction as well.

Save separate Material Profiles for Ni and NiCr, including their feed-forward
maps and gain schedules. The configuration writer supports TOML inline tables,
so these maps and schedules survive GUI saves.

## Commands, filtering, and anti-windup

- The default `max_current_step_up` and `max_current_step_down` are 0.001 A per
  reference period (`1/experiment_frequency`). Live PI slew limits scale with
  actual elapsed time. The physical output is rounded to the 1 mA programming
  grid, so instantaneous changes can differ from continuous requests by rounding.
- The next cycle starts from the setting actually transmitted, including when
  `minimum_current_change` suppresses a write. This setting is not a verified
  PSU readback; external manual changes or a rejected command require hardware
  diagnostics. The DMM current remains available for safety monitoring.
- `pid_integral_current_limit_a` bounds the correction in amps, also bounded by
  `max_current`. The old `pid_integral_limit` error-seconds setting is obsolete.
- `pid_tracking_time_s` defaults to 30 s. Back-calculation uses the accepted
  actuator command after slew limits, quantization, and recovery overrides.
  Differences below half a programming step do not unwind the integral, allowing
  small corrections to accumulate into a real command.
- Invalid temperatures do not integrate error. The default hold policy freezes
  current and the temperature program, clears rate history, and stops after
  measurement_fail_limit failures. Legacy recovery is opt-in.
- `temperature_rate_window_s` defaults to 8 s. Regression uses monotonic
  timestamps and at least three accepted measurements. Normal temperature
  median filtering is still controlled by `measurement_filter_samples`.
- `temperature_prediction_time_s` defaults to 2 s. When measured heating rate
  exceeds the programmed ramp rate, a short temperature prediction smoothly
  reduces the requested heat before target crossing. Set it to zero to disable.
  The programmed rate is zero during holds. The old normal catch-up and rate
  threshold settings no longer force current steps or reset the PI.

## Retuning and validation

Existing gains from the accumulating-current implementation are not transferable
to this absolute-current PI. Software tuning now permits small gains below the
old 0.001 Kp / 1e-5 Ki floors and uses an integral time at least as long as its
conservative response time (minimum 30 s). The September 2026 trial profiles below use explicitly provisional ramp-derived biases.

Validate resistance-derived temperature against an independent thermometer
before retuning. The reviewed NiCr runs reported T0 scatter equivalent to
33.5/58.7 C and a later resistance decrease while electrical power increased.
Feedback changes cannot establish actual wire temperature from that signal.
The reviewed exported curves also have straight extensions from approximately
284 C (Ni) and 293 C (NiCr) to 1000 C; these are not measured high-temperature
calibration points.

Begin with independently calibrated low/mid-temperature holds and small,
resolvable open-loop current rise/fall steps. Measure equilibrium feed-forward
points separately. Tune conservative PI for each wire geometry, save a Material
Profile, and evaluate settled hold error separately from ramp tracking error.
Keep derivative action disabled initially. Hardware tests are required to
establish the resulting temperature tolerance.

## September 2026 Ni / NiCr trial profiles

The application profiles Ni_100_152 and NiCr_100_163 were rebuilt from
runs 139 and 136 respectively. tools/build_ramp_profiles.py reproduces them.
They contain **provisional 10 C/min ramp-derived current biases**, not measured
equilibrium maps and not validated PI gains.

| Setting | Ni | NiCr |
| --- | ---: | ---: |
| Kp (A/C) | 0.001 | 0.0001 |
| Ki (A/(C s)) | 0.00001 | 0.000001 |
| Ti (s) | 100 | 100 |
| Current slew step per 2 s | 0.001 A | 0.001 A |
| Prediction horizon | 0 s | 0 s |
| Trial target ceiling | 600 C | 600 C |
| Current ceiling | 0.1 A | 0.1 A |
| Sample power cutoff | 0.05 W | 0.25 W |

The reduced electrical limits bound this comparison; they are not wire ratings.
Programs above the trial target ceiling or with ramp rates other than 10 C/min
are rejected before instruments open. Both trial ceilings are now 600 C at the user's request.

Table construction uses RMS commanded current in target-temperature bins,
paired with mean logged temperature. This includes both phases of Ni's
oscillation and approximately preserves average I-squared heating. Ni uses
target bins 30-120 C (measured-temperature points about 32-111 C). NiCr uses
100-260 C bins (points about 107-245 C), excluding noisy startup, the
range-transition spike and the late collapse. Source run names, hashes, bins,
counts and limitations are saved in current_feedforward_provenance.
Both maps include a 23 C / 10 mA startup-command anchor, which is not an
equilibrium measurement. Interpolation below the first derived point is
particularly uncertain for NiCr. Endpoint values are held, never extrapolated.

Use the same wire geometry/mounting as the source runs. Restart the application,
load the matching material profile and R(T) reference, then recalibrate T. Zero.
Loading a profile invalidates the previous material's T0 calibration. If NiCr
still reports large calibration scatter, the temperature uncertainty remains;
the software cannot correct an unreliable sensor by tuning.

Suggested next comparison programs, in the GUI's actual input syntax:

Ni (continuous ramp followed by a five-minute final hold):
    {start_T=23; step_T=0; target_T=600; ramp_speed_min=10; hold_step_time_min=5}

NiCr (continuous ramp followed by a five-minute final hold):
    {start_T=23; step_T=0; target_T=600; ramp_speed_min=10; hold_step_time_min=5}

The final hold now honors hold_step_time_min for both simple and stepped ramps.
The ramp-derived bias can initially overheat a hold; the integral corrects its
residual. Settled hold data should replace these provisional currents.

Both profiles use a 2 V voltage range, a 20 mA initial current range with staged
increases, 10 PLC integration, 0.5 s post-command/range settling, and three
discarded pairs after a range change. Complete V/I pairs are retried together.
Resistance retries remain enabled even though the legacy dynamic jump/probe
guard is off. Retry checks include equivalent temperature spread, so low-TCR
errors cannot hide behind a large ohmic threshold. Only the latest two agreeing
fresh retries can establish a new resistance state.

The sustained resistance/power guard compares medians at each end of a 30 s
window. It stops heating when indicated T falls at least 15 C while resistance
falls, current rises at least 3 mA, power rises at least 20%, and the target
does not fall. It operates above 20 mA and resets on invalid measurements. This
is a diagnostic abort for inconsistent feedback, not a temperature estimator.

## New diagnostics and calibration exports

control_diagnostics.jsonl records raw/filtered T, first-candidate readings,
paired-acquisition timestamps, meter ranges and range-change count, setpoint,
requested/applied/accepted current, integral/P/prediction/feed-forward terms,
output limiting and invalid-hold/abort status. Invalid numeric values are JSON
null. Existing CSV, XLSX and HDF5 measurement columns remain compatible.

Run metadata includes the full temperature program and SHA-256 hashes of the
controller source files. New exports also include r_vs_t_source.csv and
curve_metadata.json. Reloading r_vs_t.csv verifies both hashes and restores
the unextended source curve, rather than treating extrapolation endpoints as
measured calibration. Old exports lacking this sidecar cannot recover their
original measured bounds automatically.

Flat R(T) sections now use a deterministic midpoint temperature (except the
source endpoint used to anchor extrapolation), expose their ambiguity intervals,
and print the maximum ambiguity. This removes arbitrary duplicate selection;
it does not make a flat calibration physically invertible.

### Temporary extension to 600 C

Both profiles now allow 600 C programs at 10 C/min. Each table adds a 600 C
placeholder with the last data-derived current (Ni 0.062327 A, NiCr 0.059810 A).
This holds the bias constant above the measured range; PI adjusts the current.
These are not measurements or validated high-temperature heating currents.
The original derived ranges and source evidence remain in the JSON provenance.

R(T) extrapolation is enabled through 600 C using the existing endpoint-fit
method when the loaded source reference ends earlier. Temperature outside the
source reference is an estimate, not a validated calibration. Independently
validate high-temperature readings before treating them as measurement results.
Existing current/power cutoffs and feedback guards remain enabled; they may
stop the run or prevent reaching 600 C. No higher electrical limits were inferred.

Click Load again for the installed wire's profile, reload its R(T) reference,
and recalibrate T. Zero. An already loaded GUI profile does not update from disk.
Replace the placeholder bias with reviewed data after the next experiment.
