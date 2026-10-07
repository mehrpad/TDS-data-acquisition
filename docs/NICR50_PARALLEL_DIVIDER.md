# 50 µm NiCr with a 20 Ω parallel resistor bank

Load the updated `NiCr_50.json` explicitly in the GUI. Its current table uses
run 127 with five 100 Ω / 0.25 W resistors in parallel: **20 Ω / 1.25 W nominal**.
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
**total PSU amperes**. The CSV separately records measured `wire_current_a`.
Applied commands are paired with the readings they produced; they are not
independently measured total PSU currents.

At the approximately 85.6 Ω cold resistance, the nominal ratio is
`I_PSU / I_wire = 1 + 85.6/20 = 5.28`. Ammeter burden, leads and resistor tolerance
affect this ratio. The new table uses observed commands rather than a fixed
conversion. The cold anchor is **0.001 A total at 23 °C**, approximately **0.19 mA
wire current**, and is not a measured equilibrium point.

The fitted ramp bins cover approximately 74–590 °C indicated. The small extension
to 600 °C estimates **0.15524 A total / 0.02656 A wire**. No final hold was measured.
Temperature uses the existing R(T) curve, extrapolated above 293.4 °C; the table
does not independently validate physical temperature.

| Setting | Value |
| --- | --- |
| Total PSU command ceiling | 0.18 A |
| Raw measured wire-current cutoff | 0.03 A |
| PSU compliance / sample-voltage cutoff | 3 V / 3 V |
| Wire power cutoff (wire V × branch I) | 0.10 W |
| Indicated temperature cutoff / target ceiling | 600 °C / 600 °C |
| Kp / Ki / Ti | 0.0000825 A/°C / 0.00000055 A/(°C·s) / 150 s |
| Integral correction limit | ±0.02 A total |
| Temperature prediction | 5 s |
| Current slew steps | 0.001 A total |
| T0 / startup total current | 0.001 A |
| T0 paired samples per measurement | 15 |
| Startup/T0 current-DMM range | 2 mA |
| Startup/T0 voltage-DMM range | 0.2 V |

At 3 V, bank dissipation is at most **0.45 W nominal**, or **0.09 W per resistor**.
The 1.25 W summed rating assumes suitable cooling and mounting for each resistor.
Keep 3 V compliance; 30 V would permit 45 W in this bank. A missing bank is not
detected before output enable. Raw wire-current, power and temperature guards
remain active on every measurement pair.

## Next measurement

Cool the wire and recalibrate T0 with this wiring. Run 127's T0 spread was
7.32 °C, above the 5 °C warning threshold; startup accuracy remains uncertain.
If scatter persists, check the measurement chain with a stable precision resistor
near 85–90 Ω. Record the active wire length and measured bank resistance.

Use **590 °C target, 10 °C/min and 600 °C maximum** for the next comparison.
A 600 °C target leaves no overshoot allowance below the raw safety cutoff.
Do not raise the cutoff to conceal shutdown. Updated tuning needs a hardware
trial; smooth tracking is not yet verified.

See [run 127 findings and derivation](NICR50_RUN_127.md). Per-run logs and
diagnostics record the settings and branch measurements.
