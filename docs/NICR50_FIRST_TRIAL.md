# Provisional 50 um NiCr profile

There are no heating measurements for this wire. Every point in the new
`NiCr_50_provisional` current table and reference curve is an **estimate**.
The profile permits targets through 600 C at 10 C/min so a comparison run can
replace the estimates. It does not establish accurate physical temperature or
stable control over that range.

## Basis

The existing `NiCr_100_163` profile, based on run 109, is the donor. The estimated
50 um current bias is 0.30 times the donor bias, except for a lower 2 mA startup
anchor. At 600 C this gives about 25.17 mA. The donor itself has an estimated
500-600 C extension, so this part has two layers of uncertainty.

At the same length and alloy, halving diameter gives four times the resistance.
In a simplified model with equal surface heat loss per unit area, heating power
scales with diameter, so `I_new/I_old = (d_new/d_old)^1.5 = 0.354`.
If end conduction or heat capacity dominates, a factor near 0.25 can result.
The selected factor 0.30 is an engineering starting estimate between these
models. Convection, emissivity, thermal contacts and the atmosphere can change
it substantially. The diameter/heat-loss model is our inference; it is not an
apparatus calibration. [Kanthal's design guidance](https://prodshop.kanthal.com/en/knowledge-hub/heating-material-knowledge/design-calculations-and-standard-tolerances/design-calculations/)
relates surface load to power and surface area and describes the dependence on
heat dissipation conditions. Its industrial surface-load ratings are not used
as ratings for this 50 um wire.

The companion R(T) file is the donor's reference multiplied by four. It initially
assumes the donor's 163 mm length; the actual new length has not been supplied.
Calibrate T. Zero with the wire cooled to rescale to its measured cold resistance.
This corrects a uniform geometry factor, not differences in alloy/TCR, contact
resistance, or resistance changes during heating. Use an independently measured
reference instead when available. Do not load the Ni reference for NiCr.

The original NiCr reference ends at 293.4 C. The new reference preserves those
temperature rows and labels them `measured_on_this_wire = False`; software
extrapolation supplies the conversion above that range. Earlier NiCr runs showed
calibration scatter and apparent resistance drift. Creating this profile does
not resolve those thermometry concerns.

## Initial settings

| Parameter | Setting |
| --- | ---: |
| Startup / measurement floor / T0 search start | 2 mA |
| Kp / Ki / Ti | 0.00003 / 0.0000003 / 100 s |
| Integral correction bound | 10 mA |
| Current step up/down | 1 mA per nominal 2 s cycle |
| Median filter / prediction horizon | 3 samples / 2 s |
| Maximum current / sample power | 0.10 A / 0.50 W |
| Maximum sample voltage / supply compliance | 20 V / 30 V |
| Maximum indicated temperature / trial target | 600 C / 600 C |
| Invalid feedback | Reduce current, pause ramp/integral; stop after 10 failures |

The current and power limits provide headroom for the roughly 25 mA estimated
bias at 600 C. They are software ceilings, not measured wire ratings. A thinner
wire does not justify arbitrarily increasing maximum current. Loading this
profile does not command the maximum values. The supply's 1 mA resolution is a
significant fraction of the low-temperature current and can limit smoothness.

## Files and first measurement

The JSON, current CSV, estimated R(T) CSV and these instructions are also installed
in `T:/Monajem/tds_sofia`:

- `NiCr_50_provisional.json`
- `NiCr_50_provisional_current_table.csv`
- `NiCr_50_provisional_R_vs_T_estimated.csv`
- `NICR50_FIRST_TRIAL.md`

Restart the updated application. Select **NiCr_50_provisional**, click **Load**,
select **TEMPERATURE** mode, and load the matching estimated R(T) CSV. Let the
wire cool and stabilize, enter its actual room temperature, then Calibrate T.
Zero. Review any calibration-scatter warning before treating the indicated
temperature as reliable.

Start with a 100-150 C comparison before using the unmeasured higher range:

    {start_T=23; step_T=0; target_T=150; ramp_speed_min=10; hold_step_time_min=5}

The requested full-range program is enabled:

    {start_T=23; step_T=50; target_T=600; ramp_speed_min=10; hold_step_time_min=5}

Settled holds help separate equilibrium current from ramp heating. A raw
indicated temperature above the 600 C cutoff shuts down even during a final
600 C hold. Save the whole experiment folder, especially diagnostics, metadata,
data CSV and R(T) snapshots. These can update the current table; independent
temperature measurements are needed to validate the R(T) conversion itself.

Rebuild from the existing donor without instrument access:

    python -m tools.build_nicr50_profile --source-curve T:/Monajem/tds_sofia/109_NiCr_uncharged_test/r_vs_t_source.csv
