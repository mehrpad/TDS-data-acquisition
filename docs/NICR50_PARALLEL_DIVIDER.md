# 50 µm NiCr with a 20 Ω parallel resistor bank

Load the updated `NiCr_50.json` explicitly in the GUI. The profile now uses
run 130's flat PI and a partial bias update, retaining information from run 127.
The bank is five 100 Ω / 0.25 W resistors in parallel: **20 Ω / 1.25 W nominal**.
Other material profiles are unchanged.

## Wiring

Keep the bank outside the chamber across PSU output terminals, the ammeter only
in the wire branch, and separate Kelvin voltage leads directly across the active
wire. Exclude the ammeter, current leads and bank from voltage sensing. Keep PSU
remote sense at the PSU terminals.

```text
              +-- ammeter -- NiCr wire --+
PSU + --------+                         +-------- PSU -
              +--- 20 Ω / 1.25 W bank --+
```

## Current table and limits

Table `current_a`, GUI Initial/Max PSU Current, PI outputs and slew commands are
**total PSU amperes**. The CSV separately records `wire_current_a`, measured for
observed rows and estimated for adjusted higher-temperature rows.
Applied commands are paired with the readings they produced; they are not
independently measured total PSU currents.

At the approximately 85.6 Ω cold resistance, the nominal ratio is
`I_PSU / I_wire = 1 + 85.6/20 = 5.28`. Ammeter burden, leads and resistor tolerance
affect this ratio. The new table uses observed commands rather than a fixed
conversion. The cold anchor is **0.001 A total at 23 °C**, approximately **0.19 mA
wire current**, and is not a measured equilibrium point.

Run 130 supplies provisional ramp bins at approximately 131–198 °C indicated.
Below 120 °C, the prior run-127 points are retained. Higher-temperature bias
keeps the earlier shape plus an approximately 20 mA command offset and a rising
I² join to 249 °C. All adjusted higher-temperature currents are labelled estimates;
previous observations remain in JSON provenance. The 600 °C estimate is
**0.17538 A total / 0.03000 A wire**. No final hold was measured.
Temperature uses the existing R(T) curve, extrapolated above 293.4 °C; the table
does not independently validate physical temperature.

| Setting | Value |
| --- | --- |
| Total PSU command ceiling | 3 A |
| Raw measured wire-current cutoff | 3 A |
| PSU compliance / sample-voltage cutoff | 5 V / 5 V |
| Wire power cutoff (wire V × branch I) | 15 W |
| Indicated temperature cutoff / target ceiling | 600 °C / 600 °C |
| Flat Kp / Ki / Ti from run 130 | 0.001 A/°C / 0.000001666666667 A/(°C·s) / 600 s |
| Gain schedule | None |
| Integral correction limit | ±3 A total; tied to PSU command ceiling |
| Temperature prediction | 5 s |
| Current slew steps | 0.001 A total |
| Current switching hysteresis | 0.0001 A total |
| T0 / startup total current | 0.001 A |
| T0 paired samples per measurement | 15 |
| Final T0 calibration samples | 9 |
| Startup/T0 current-DMM range | 2 mA |
| Startup/T0 voltage-DMM range | 0.2 V |

At 5 V, bank dissipation is **1.25 W nominal**, or **0.25 W per resistor**, the
full nominal rating with no margin. Use an adequately cooled higher-rated bank
for sustained operation near 5 V. A 3 A ceiling does not make 3 A attainable
through this load at 5 V; voltage compliance may prevent further current increase.
These are user-selected ceilings, not verified wire ratings. A missing bank is
not detected before output enable. Raw electrical and temperature guards remain.

## Next measurement

Cool the wire and recalibrate T0 with this wiring. Run 130's T0 spread was
82.37 °C, above the 5 °C warning threshold; startup accuracy remains uncertain.
If scatter persists, check the measurement chain with a stable precision resistor
near 85–90 Ω. Record the active wire length and measured bank resistance.

Use **590 °C target, 10 °C/min and 600 °C maximum** for the next comparison.
A 600 °C target leaves no overshoot allowance below the raw safety cutoff.
Do not raise the cutoff to conceal shutdown. Updated tuning needs a hardware
trial; smooth tracking is not yet verified.

See [run 127 findings and derivation](NICR50_RUN_127.md). Per-run logs and
diagnostics record the settings and branch measurements.
See [run 130's current profile](NICR50_RUN_130.md) for the
latest table adjustment and settings. A stable cold calibration is needed before
assessing the physical temperature or rebuilding the thermal table.
