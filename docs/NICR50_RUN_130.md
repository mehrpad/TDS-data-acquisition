# Thin NiCr run 130: adopt the working PI and correct the ramp bias

Source: `T:\Monajem\tds_sofia\130_test`. Original logs, CSV, diagnostics and
R(T) files are unchanged. Only the thin NiCr profile is updated.

## Findings

Run 130 used flat **Kp=0.001 A/°C, Ki=0.000001666666667 A/(°C·s), Ti=600 s**,
with an empty gain schedule. The run stopped by user request at an indicated
200.04 °C against a 208.59 °C target; it does not contain a new measurement
to 600 °C. There were 550 saved cycles, of which 533 were accepted and 17 backed
off invalid readings. Peak wire current was 13.44 mA and peak wire power 16.12 mW.

Median indicated temperature minus target was approximately +6.87 °C at
100–150 °C target, −0.98 °C at 150–200 °C, and −7.37 °C over the final
200–209 °C target segment. This is much closer than run 129's sustained high
lag, although early readings still overshot by up to 62.21 °C. T0 reported
**82.37 °C scatter**, so these indicated temperatures are not independently
validated physical temperatures.

At the end, old bias was 59.14 mA plus 12.94 mA integral correction and
8.55 mA proportional correction, requesting 80.63 mA total. The old bias table
therefore required substantial compensation even with the improved PI.

## Changes

- Preserve the exact run-130 flat PI values and empty gain schedule. The GUI
  shows Kp=0.001 and Ti=600 s; no old schedule overrides them.
- Derive five provisional ramp-bias points around 131–198 °C from RMS applied
  commands paired with raw indicated temperature. Select accepted positive
  finite readings after 300 s, with tracking error ≤12 °C and raw/filter
  difference ≤8 °C; omit range-change cycles and the following two cycles.
  Require at least 20 samples per bin. The updater checks alignment and rejects
  current reversals rather than silently manufacturing a fit.
- Retain low-temperature run-127 points below 120 °C rather than fitting noisy
  startup. Bias near 200 °C rises to approximately 79 mA total. Higher-temperature
  bias uses the old shape plus an approximately 20 mA command offset, with a
  rising I² join to the 249 °C point. All shifted higher-temperature currents
  and wire-current estimates are explicitly marked unmeasured. The estimated
  600 °C command is approximately 175.38 mA total, not a new 600 °C measurement.
- Retain 1 mA startup/T0, paired averaging, 1 mA slew, 5 s rate prediction and
  the 600 °C indicated-temperature cutoff. Increase final T0 calibration
  samples from five to nine; this cannot guarantee a reliable cold anchor.
- Apply the user's **5 V PSU compliance and sample-voltage cutoff, 3 A total
  command and branch-current cutoffs, and 15 W measured wire-power cutoff**.
  The old 30 mA independent wire cutoff is replaced explicitly. Integral-state
  limit follows the selected total command ceiling, with output anti-windup.
- Add a GUI readout for PSU and sample voltage limits. Profile loading already
  refreshes current and power fields; a GUI regression verifies these now show
  3 A, 15 W and 5 V and persist into the active configuration.

Reproduce with `python -m tools.update_nicr50_run130 --run <run-130-folder>`.
The JSON retains the previous table, observed bins, source hashes and estimation
method. Repeated application uses the original baseline rather than adding the
offset again. R(T) is unchanged: refitting it from temperatures it already
produced would be circular. Its extension above 293.4 °C remains unvalidated.

## Electrical implications

The five 100 Ω / 0.25 W resistors form a 20 Ω / 1.25 W nominal bank. At **5 V**
it dissipates **1.25 W total, 0.25 W per resistor**: the full nominal rating, with
no tolerance, temperature or cooling margin. For sustained operation near 5 V,
use an adequately mounted higher-rated 20 Ω bank, for example at least 2.5 W
nominal. These selected settings are not verified ratings for the thin wire.

A 3 A command ceiling does not mean 3 A can flow through the current load at
5 V: the bank draws at most approximately 0.25 A and an approximately 90 Ω wire
draws at most approximately 56 mA, ignoring branch burden. The supply may enter
voltage compliance before the requested current is reached; then raising a
current command cannot further heat this load. This follows the usual CC/CV
operation described by [Siglent](https://siglentna.com/operating-tip/spd-constant-current/).
The GUI power value is wire V × branch I, not bank or total PSU power.

Cool and recalibrate before judging the new tuning. Run 130 establishes improved
indicated tracking only up to about 200 °C. A further run is needed to validate
higher-temperature tracking, and reliable independent thermometry is needed to
validate physical temperature.
