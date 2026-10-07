# Thin NiCr run 129: stronger correction without removing electrical guards

Source: `T:\Monajem\tds_sofia\129_NiCr_50_hydrogen_curr_1`, its per-run log,
metadata, CSV and controller diagnostics. No original measurements are changed.

## Why the command plateaued

The run was stopped by the user. Its final indicated temperature was 293.10 °C
against a 511.14 °C target. Wire power peaked below 0.065 W, against 0.10 W;
total command peaked at 163 mA, against 180 mA; measured wire current peaked
at 26.83 mA, against 30 mA. The electrical limits were not active.

The separate integral correction reached +20 mA and remained there for about
22.75 minutes. At the end the requested command was 125.57 mA table bias +
17.99 mA proportional correction + 20 mA integral = 163.55 mA. The bias section
near 330–430 °C was nearly flat. The longest fixed-command period was about
70 seconds at 125 mA while target increased from 369.2 to 380.9 °C. The requested
command only changed from 124.76 to 125.70 mA, with 1 mA quantization and switching
hysteresis holding the applied value. This was a controller command plateau,
not evidence of the PSU ignoring a rising command.

## Revised profile

- Replace the near-flat interior bias points between 331.52 and 429.12 °C with
  linear interpolation in I² between the unchanged run-127 anchors. This modest
  increase is an engineering bias adjustment, not a new thermal measurement.
  Original points remain in JSON provenance; corrected current/wire estimates
  are marked `measured_on_this_wire=false` in JSON and CSV.
- Keep startup Kp=0.0000825 and Ki=0.00000055 below 40 mA table bias. Blend to
  Kp=0.000165 and Ki=0.00000165 at 70 mA bias: twice proportional gain and three
  times integral gain, with Ti=100 s. This retains gentler startup correction.
- Tie integral-state limits to the existing total command ceiling (±180 mA)
  rather than the separate ±20 mA limit. Output anti-windup, 1 mA/cycle slew,
  rate prediction and raw electrical/temperature guards remain active.
- Reduce switching hysteresis from 0.2 to 0.1 mA. The PSU still has 1 mA command
  resolution, so brief constant-command periods are expected and cannot all
  be eliminated.

The reproducible updater is `python -m tools.retune_nicr50_after_run129`. If
rebuilding from source run 127, run its baseline updater first and this updater
second. Repeated application is idempotent. Other material profiles are unchanged.

## Why full PSU output is unsuitable

Five 100 Ω / 0.25 W resistors in parallel give 20 Ω and 1.25 W nominal total
rating. At 3 V, nominal dissipation is 0.45 W total (0.09 W each). At 30 V,
it would be 45 W total (9 W each): 36 times the summed rated power. The present
bank therefore cannot support full 30 V PSU compliance. A bank failure would
also change current division and send more current through the thin wire.

The profile retains 3 V compliance, 180 mA total command, 30 mA wire-current,
0.10 W wire-power and 600 °C indicated-temperature cutoffs. These limits may
restrict a future run; that must be evaluated against the actual wire, bank,
mounting and chamber conditions rather than bypassed. The stronger controller
can now use the existing headroom when there is persistent lag.

## Calibration limitation

Run 129 reported **92.42 °C T0 scatter**, above the 5 °C warning threshold.
Cold resistance changed from approximately 85.60 Ω in run 127 to 87.44 Ω in
run 129. The cause is not established: actual wire/contact changes, insufficient
cooling and measurement offsets must be distinguished. The R(T) reference also
has an unusually steep endpoint near 293.4 °C and an unvalidated extension above
it. Run 129 therefore cannot establish the true physical plateau temperature.

Repeat calibration with a fully cooled wire and check measurement stability
before judging the new tuning. No new measured R(T) curve was fabricated from
this run. The profile remains provisional and does not guarantee 600 °C tracking.

Validation includes a controller regression with sustained lag proving that
correction continues beyond 20 mA, remains within slew/output limits, retains
lower startup gains, and still trips raw electrical and temperature guards.

These settings are historical. [Run 130's revision](NICR50_RUN_130.md) adopts
the user's improved flat PI, partially updates the ramp bias, and applies the
explicitly requested 5 V / 3 A / 15 W ceilings. It replaces the old 30 mA cutoff.
