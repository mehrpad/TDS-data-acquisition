# Separate low-current T0 calibration

The calibration search previously forced at least 5 mA, even when a thin-wire
profile requested less. The GUI's Initial Current field also overwrote the
profile's T0 current. Both paths are corrected: T0 now uses independent settings
saved and loaded with each material profile.

| Profile | T0 start / hard command ceiling | T0 voltage / current DMM ranges | Experiment Initial Current |
| --- | ---: | ---: | ---: |
| Ni_100_152 | 0.005 A / 0.005 A | 0.2 V / 0.02 A | 0.010 A |
| NiCr_100_163 | 0.005 A / 0.005 A | 0.2 V / 0.02 A | 0.010 A |
| Ni_50_200 | 0.001 A / 0.001 A | 0.2 V / 0.002 A | 0.005 A |
| NiCr_50_provisional | 0.001 A / 0.001 A | 0.2 V / 0.002 A | 0.002 A |

The supply's programmable current increment is 1 mA. This is the lowest nonzero
command available to the present controller. At equal resistance, moving from
5 mA to 1 mA reduces Joule heating to 1/25. Actual heating and measurement
precision still need to be checked on the mounted wire.

The JSON keys are `t0_current_search_start` and `t0_calibration_current` (the
ceiling), both in amperes; `t0_current_step` is 0.001 A. Start equals ceiling in
these four profiles, so calibration cannot search upward. The preparation
current is also bounded by the T0 ceiling. The limit applies to commanded
current; it does not certify supply regulation accuracy at 1 mA. Unstable
readings produce a calibration failure and supply shutdown instead of a
larger current command. Do not raise the limit just to clear a failure.

`t0_dmm_voltage_range_v` and `t0_dmm_current_range_a` select more sensitive
initial fixed DMM ranges for T0 only. Normal range recovery remains enabled if
a reading overloads; changing a meter range does not raise supply current.
Experiment DMM ranges are preserved. Settling, sample count and stability
settings are now also explicitly included in all four JSON files.

T0 range selection now reserves no heating-step margin and uses
`t0_dmm_range_switch_fraction` (default 0.95). A 1.5 mA reading therefore stays
on the sensitive 2 mA range; actual overloads still trigger range recovery.
The thin NiCr profile uses `t0_pair_samples: 9` to average paired voltage and
current readings before calculating each calibration resistance. The other
profiles retain single-pair acquisition. Averaging cannot establish that a
wire is cool or remove a systematic measurement offset.

Restart the updated application, select the matching Material Profile, and
click Load. Let the wire cool fully and stabilize, enter its actual room
temperature, load the matching R(T) curve, then click Calibrate T. Zero. The
profile-load message and calibration button tooltip display the T0 current and
ceiling. Initial Current controls experiment/tuning startup and does not change
T0. Recalibrate after switching profiles.

An indicated 70 C during T0 does not by itself prove the wire physically heated
to 70 C: the uncalibrated curve can infer the wrong temperature for a different
wire geometry or contact resistance. T0 anchors the stable measured resistance
to the entered room temperature. That assumption fails if the wire is still
hot; a stable resistance alone cannot prove it is at room temperature. Check
cooling and any calibration-scatter warning before running. The estimated
NiCr 50 um R(T) curve still needs independent validation.

All four updated JSON files and these instructions are installed in
`T:/Monajem/tds_sofia`. The heating tables and maximum experiment temperature,
current and power limits are unchanged by this T0 correction.
