# Thin NiCr run 123: honor the 1 mA startup setting

Run 123 saved `startup_current: 0.001` but retained
`measurement_current_floor: 0.002`. The controller clamped startup and every
subsequent command to that higher floor. This explains the GUI mismatch and
why the wire could not cool toward its low initial target.

Initial Current now lowers the effective active-control floor. The GUI also
saves the lowered floor, so a saved configuration reflects the command used.
The hard `min_current` and `max_current` bounds remain enforced. An initial
current below `min_current` is rejected before starting. T0 remains independent
of this GUI field. The supply grid is 0.001 A = 1 mA; 0.001 mA would be 1 uA
and is not an available nonzero command on this supply.

The renamed thin NiCr files are:

- `NiCr_50.json`
- `NiCr_50_current_table.csv`
- `NiCr_50_R_vs_T_estimated.csv`

The wire length remains unknown, so the name does not invent a length.
The JSON, CSV and reference all retain their estimated/provisional labels.
The startup, tuning, active-control floor, T0 search start and T0 command
ceiling are all 1 mA. The 23 C current-table anchor is now 1 mA, a startup
choice rather than a measured equilibrium point. Higher table rows retain
their diameter-based estimates. Maximum current, power and temperature stay
at 0.10 A, 0.50 W and 600 C.

## Why the run stopped and cannot calibrate the higher table rows

The supplied traceback reports a raw inferred temperature of 1194.15 C,
which triggered the 600 C cutoff. The preceding saved temperatures were much
lower. A measurement spike is plausible, but these resistance-only readings
cannot establish the physical temperature. Every raw reading continues to
pass the safety checks; averaging cannot hide an over-temperature reading.

The saved controller and calibration SHA-256 hashes exactly match commit
`ca010af` with Windows line endings. They do not match `1ebb454`, which added
paired averaging and corrected premature T0 meter range changes. Metadata
also contains the older PI gains, 0.5 s settling and no batch-averaging
settings. Run 123 therefore did not test those latest noise fixes.

In 147.74 s, 74 cycles were saved: 32 accepted and 42 rejected/backoff cycles
(56.8%). The target only advanced to 33.38 C. Filtered accepted temperatures
ranged from 1.71 to 81.61 C. Commands were 2 mA for 67 cycles, 3 mA for six
and 4 mA for one. T0 reported 44.15 C of temperature-equivalent scatter.
These data do not provide a defensible current-versus-temperature fit,
particularly above 100 C. Run 123 is recorded in the profile's provenance
with the reason its measurements were excluded from table calibration.

## Next run

Update the application code as well as loading the new JSON. The JSON alone
cannot enable the acquisition fixes. Restart, select `NiCr_50`, click Load,
load `NiCr_50_R_vs_T_estimated.csv`, and check Initial Current is 0.001 A.
Let the wire cool fully, then recalibrate T0. The unchanged low-current noise
settings use nine paired readings per T0 measurement, five per experiment
measurement, lower PI gains and one-second current settling.

The console should report startup and the active floor as 0.0010 A. New run
metadata should include `measurement_pair_samples: 5`, `t0_pair_samples: 9`
and `t0_dmm_range_switch_fraction: 0.95`. Actual current can differ from a
1 mA command; lowering the command does not certify supply regulation.
If T0 scatter remains large, investigate the contacts and measurement signal
before trusting inferred temperatures. A short ramp with settled holds can
provide better table data; smooth, physically accurate operation to 600 C
has not yet been demonstrated. Save the complete next run including
`run_outcome.json` and the batch diagnostics.
