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

## Table sources

Ni retains the previous table/interpolation through 100 C. Above 100 C it uses
19 reviewed finite readings from current ramp 107, excluding a spike and missing
temperatures. The 100-150 C transition is blended. Gaps from about 136-415 C and
484-575 C are interpolated. The 600 C point holds the last binned current.

The 0.1 A/min current sweep heated much faster than a 10 C/min temperature
program. Its bias may overestimate the required current; it is provisional,
not an equilibrium map. PI gains are unchanged because run 100 was stable
before reaching its old current ceiling. See [analysis and plot](NI_RUNS_100_107.md).

NiCr retains the earlier provisional run 136 table. Its 600 C endpoint holds
its last data-derived current, 0.059810 A. Neither new run supplies NiCr data.
R(T) extrapolation through 600 C remains enabled; temperatures beyond the
source reference are estimates that need independent validation.

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
