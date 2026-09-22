# Next Ni and NiCr trial

Restart the updated application. Select the matching Material Profile:
- Ni: Ni_100_152
- NiCr: NiCr_100_163

The matching JSON files are installed in the application and copied to
T:/Monajem/tds_sofia. The previous network profiles are retained as .bak files.
Load the appropriate R(T) reference and run Calibrate T. Zero again after
loading the profile. Use the same wire geometry and mounting as the source run.

## Settings

| Parameter | Ni | NiCr |
| --- | ---: | ---: |
| Kp | 0.001 | 0.0001 |
| Ki | 0.00001 | 0.000001 |
| Ti | 100 s | 100 s |
| Current steps up/down | 1 mA | 1 mA |
| Prediction | Off | Off |
| Maximum current | 0.100 A | 0.100 A |
| Maximum sample power | 0.050 W | 0.250 W |
| Trial target ceiling | 600 C | 600 C |

Both use 0.5 Hz control, a 2 V starting voltage range, staged current ranges,
10 PLC meter integration, and 0.5 s settling. These are provisional trial
settings. The electrical limits bound this test, not a certified wire rating.

The feed-forward tables come from the two 10 C/min ramps. They are not
equilibrium calibration. Ni has derived points around 32-111 C; NiCr around
107-245 C. NiCr startup, the five-minute spike and the late collapse were excluded.
The 23 C / 10 mA startup anchor is a recorded command, not an equilibrium point.
NiCr's low-temperature interpolation therefore remains particularly uncertain.

## Paste one program for the installed wire

Ni:

    {start_T=23; step_T=0; target_T=600; ramp_speed_min=10; hold_step_time_min=5}

NiCr:

    {start_T=23; step_T=0; target_T=600; ramp_speed_min=10; hold_step_time_min=5}

The final five-minute hold now executes instead of stopping on arrival.
The profiles reject a faster/different ramp or a target above the trial ceiling.
After these runs, settled hold data can replace the provisional ramp bias.
The requested 600 C trials extend beyond both measured feed-forward ranges.

If feedback is invalid, current holds and the ramp pauses. Ten consecutive
invalid cycles stop the run. A sustained fall in resistance-derived temperature
while current and power rise also stops it. Do not interpret a stopped run or
persisting NiCr calibration-scatter warning as solved temperature accuracy.

Save the complete run folder, especially data.csv, run_metadata.json,
control_diagnostics.jsonl, r_vs_t.csv, r_vs_t_source.csv, curve_metadata.json,
and calibration_info.txt when present. The new diagnostics distinguish raw
spikes, rejected readings, range changes, current limits and PI behavior.

The software changes are tested without instruments. Actual stability and
temperature accuracy still need this hardware comparison.

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
