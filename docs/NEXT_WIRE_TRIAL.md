# Next Ni and NiCr trial

Restart the updated application. Select the matching Material Profile and click
Load: Ni_100_152 for Ni, NiCr_100_163 for NiCr. Load the matching R(T) reference
and run Calibrate T. Zero with the wire cooled. Use the same geometry/mounting
as the source measurements. Disk changes do not update an already loaded GUI.

Profiles are installed in files/material_profiles and T:/Monajem/tds_sofia.
Existing network profiles are backed up before replacement.

## Settings

| Parameter | Ni | NiCr |
| --- | ---: | ---: |
| Kp | 0.001 | 0.0001 |
| Ki | 0.00001 | 0.000001 |
| Ti | 100 s | 100 s |
| Current steps up/down | 1 mA | 1 mA |
| Prediction | Off | Off |
| Maximum current | 0.31 A | 0.100 A |
| Maximum sample power | 0.83 W | 0.250 W |
| Maximum Temperature (GUI cutoff) | 600 C | 600 C |
| Trial target ceiling | 600 C | 600 C |

Ni's current and power ceilings are the measured maxima from run 107, rounded upward
to two decimal places as explicitly requested by the user. They are observed values, not verified wire ratings.
NiCr's electrical limits are unchanged and may prevent reaching the target.
Both use 0.5 Hz control, 10 PLC integration and the existing measurement guards.

Maximum Temperature is an independent software cutoff. A raw indicated
reading above it stops heating; a temperature program above it cannot start.
The cutoff applies in both temperature and current modes, depends on valid
R(T) conversion and cannot guarantee absence of physical overshoot. If target
and cutoff are both 600 C, any indicated overshoot stops the run, including a
final hold. The field is locked during a run and saved with the material profile.

## Table sources (updated September 23)

Ni now uses run 108; NiCr uses run 109. Both were 10 C/min temperature-controlled
ramps. The tables exclude invalid readings and range-change transients and use
RMS applied current paired with indicated temperature. PI gains remain unchanged.
See [the latest findings and plot](WIRE_RUNS_108_109.md).

Ni's observed table reaches about 598 C; its 600 C estimated bias is 0.202568 A.
NiCr was tested only to 500 C; its 500-600 C extension is explicitly estimated,
ending at 0.083911 A. Neither table guarantees smooth tracking on the next run.
NiCr's T0 calibration warning (20.24 C-equivalent scatter) remains unresolved.
R(T) extrapolation above the source reference remains enabled; high-temperature
readings need independent validation.

## Example temperature program for either installed wire

    {start_T=23; step_T=0; target_T=600; ramp_speed_min=10; hold_step_time_min=5}

Choose target and Maximum Temperature intentionally before starting; see the
cutoff behavior above. Settled holds can replace the provisional current biases.

The controller now reports CURRENT LIMIT when a current ceiling blocks heating,
and records that state/threshold in control_diagnostics.jsonl. Invalid feedback
still pauses the temperature ramp and stops after the configured failure count.
Save the complete run folder, especially data.csv, run_metadata.json,
control_diagnostics.jsonl and both R(T) source/export files.

The software has been tested without instruments. Hardware stability still
needs the next comparison run.
