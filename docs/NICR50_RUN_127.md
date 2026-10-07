# Thin NiCr run 127: table update and shutdown review

Source: `T:\Monajem\tds_sofia\127_test`, its per-run log and
`T:\Monajem\tds_sofia\tds log.txt`. Original measurements are unchanged.

## Findings

- Hardware was five 100 Ω / 0.25 W resistors in parallel (20 Ω), but run metadata
  still recorded 10 Ω. The old 600 °C table requested approximately 252 mA total;
  the run needed approximately 156 mA.
- Filtered temperature exceeded target by as much as **66.75 °C**, near a
  311.45 °C target. The final integral correction was **−93.41 mA**, compensating
  for oversized feed-forward. Of 1,698 saved cycles, only one was rejected;
  systematic overshoot was the dominant problem.
- Shutdown was **raw indicated temperature 600.023 °C exceeding the 600 °C
  maximum**. The terminal measurement was **2.43919 V, 26.7112 mA wire current
  and 91.3172 Ω**, about **65.15 mW wire power**. The wire was still conducting;
  this reading did not show an open circuit. Its physical integrity after
  shutdown cannot be determined from the log alone. With PSU output off, check
  continuity and compare cold resistance with the approximately **85.6 Ω** T0 value.
- T0 scatter improved to **7.32 °C** but remained above the 5 °C warning limit.
  The unscaled reference produced roughly 131–137 °C before cold-resistance
  calibration; this alone does not establish physical heating to those values.
  NiCr's small resistance change amplifies meter errors.

## Derivation

`tools/update_nicr50_run_profile.py` pairs CSV readings with diagnostics within
0.1 s. It selects accepted positive finite measurements after the first 180 s,
omits DMM range-transition cycles and the following two cycles, and forms 20 °C
bins containing at least 12 readings. The temperature axis is raw indicated T;
current is RMS **applied PSU command**, not the next command or target T.
Measured wire RMS current is recorded separately.

A weighted least-squares monotonic fit in I² smooths small reversals. The 29-point
table retains the unmeasured 23 °C / 1 mA anchor, includes observed bins from
approximately 74–590 °C, and estimates the final interval to 600 °C using the last
four bins. The endpoint is **155.235 mA total / 26.557 mA wire**. Measured/estimated
labels, selection details and source SHA-256 hashes are retained. No equilibrium
hold was measured.

R(T) is unchanged: its normalized shape matches run 127's source curve, whose
measured range ends at **293.4 °C**. Refitting R(T) from T already inferred through
that curve would be circular. Higher indicated temperatures remain estimates
requiring independent validation.

The sharp indicated acceleration near 300 °C also coincides with crossing this
curve boundary and a current-DMM range change. The last source interval has
dR/dT about 0.405 Ω/°C, whereas the extrapolated segment uses about 0.0093 Ω/°C.
Consequently, the same resistance change maps to roughly 44 times more degrees
just above the boundary. This conversion artifact can magnify the apparent rate;
it is not independent evidence of a sudden physical temperature jump. A reliable
high-temperature R(T) calibration is needed to resolve it.

## Changes and next trial

The profile records the 20 Ω / 1.25 W bank and the observed ramp table. Kp and Ki
are scaled down for the smaller divider ratio, with Ti retained at 150 s. Integral
correction is limited to ±20 mA and 5 s temperature prediction provides rate
damping. T0 uses 15 paired samples and a 2 mA DMM range to avoid the immediate
switch observed on the former 0.2 mA range. T0 and startup remain 1 mA total.
Total command ceiling is reduced to 180 mA; the 30 mA wire-current, 0.10 W
wire-power, 3 V and 600 °C guards remain.

Load the profile, cool and recalibrate, then compare a **590 °C target at
10 °C/min with the 600 °C cutoff**. These changes address the observed bias but
do not guarantee smooth tracking or physical accuracy to 600 °C before another
measurement.

These were the run-127 settings. [The run-129 revision](NICR50_RUN_129.md)
subsequently removed the separate 20 mA integral cap, smoothed the flat bias
segment and added stronger PI gains above startup. Electrical guards remain.
