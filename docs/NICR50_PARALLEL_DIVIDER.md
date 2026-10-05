# 50 µm NiCr with a 10 Ω parallel resistor

The current `NiCr_50.json` replaces the direct-connected trial settings for this
profile only. Pull the updated code and explicitly load `NiCr_50` in the GUI.
The other three profiles retain their existing settings.

## Wiring

Use a 10 Ω resistor rated **at least 2 W**, outside the chamber, across the PSU
output terminals. Keep the ammeter only in the wire branch. Connect separate
Kelvin voltage leads directly across the active wire, excluding the ammeter,
current leads and parallel resistor. Keep PSU remote sense at the PSU terminals.

```text
              +-- ammeter -- NiCr wire --+
PSU + --------+                         +-------- PSU -
              +------ 10 Ω / 2 W -------+
```

## Current table and limits

For the provisional conversion, the nominal wire resistance is 90 Ω:

`I_PSU = I_wire × (1 + 90/10) = 10 × I_wire`.

All table `current_a` values, GUI Initial/Max PSU Current values, PI outputs,
step sizes and tuning commands are **total PSU amperes**. The companion CSV
also preserves `estimated_wire_current_a`. These are estimates, not new measured
hold currents. Wire resistance varies with temperature; ammeter/lead burden adds
branch resistance. The temperature feedback corrects departures from the nominal ratio.

The cold table anchor, T0 search/ceiling and experiment startup remain **0.001 A
total**, approximately **0.0001 A in the wire**. Later table commands are ten times
the previous estimates. The 600 °C endpoint is **0.25173432 A total**, targeting
approximately 0.025173432 A in the wire at the nominal ratio.

- Total PSU command ceiling: **0.30 A**.
- Independent raw measured wire-current cutoff: **0.03 A**.
- PSU voltage compliance and sample-voltage limit: **3 V**.
- Wire power cutoff: **0.10 W**, measured from wire voltage × branch current.
- Indicated temperature cutoff and target ceiling: **600 °C**.
- Kp: **0.00015 A/°C**; Ki: **0.000001 A/(°C·s)**; Ti: **150 s**.
- Slew steps remain **0.001 A total** for finer wire-current control.
- Startup/T0 current-DMM range: **0.2 mA**; voltage-DMM range: **0.2 V**.
- Minimum valid measured branch current: **1 µA**; T0 current qualification: **10 µA**.

At 3 V, a 10 Ω parallel resistor dissipates at most **0.9 W nominal**; the 2 W
rating provides headroom. Never restore the former 30 V compliance with this
resistor: that would permit 90 W in it. A missing resistor is not detected before
output enable; verify the wiring and loaded profile first. The raw wire-current,
power and temperature checks remain active on every measurement pair.

The DMM range headroom calculation now reserves only the wire's estimated share
of a total PSU step. The raw V/I resistance calculation stays unchanged. The new
wire cutoff is separate from the total PSU command ceiling. Switching back to
another material profile resets the optional divider settings to direct connection.

## Next measurement

First verify the measurement chain with a stable precision resistor near 90 Ω.
Then let the wire cool, recalibrate T0 with this wiring and inspect the reported
resistance scatter. Lower current reduces heating but also reduces the wire
voltage signal; it does not establish temperature accuracy. Run 124's noisy
temperature readings cannot be used as a validated thermal calibration.

Record the next wire run with this profile. Save the resistor value/rating and
actual active wire length with the results. The per-run `tds_log.txt` and control
diagnostics will record the settings and wire readings. Use accepted data to
replace the nominal divider conversion and retune if necessary. The existing
estimated R(T) reference remains unchanged; its extension above the donor's
measured range is still unvalidated.
