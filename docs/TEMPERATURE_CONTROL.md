# Temperature control

Temperature mode commands absolute current:

`I_requested = I_feedforward(T_set) + Kp*(T_set - T_predicted) + integral_correction_A`

The integral increases by `Ki*error*actual_elapsed_seconds` once per accepted
measurement. The previous current is used for actuator slew limits, not added
to the PI output. Current and power safety aborts remain active.

## Feed-forward and gain schedules

Add equilibrium measurements for the actual wire to `files/config.toml` using
`current_feedforward_table`. Each point contains `temperature_c` and `current_a`.
For example, the **format only**, with illustrative values that must be replaced:

```toml
current_feedforward_table = [
  { temperature_c = 100.0, current_a = 0.05 },
  { temperature_c = 200.0, current_a = 0.08 },
]
```

Current is interpolated between measured points and held at the endpoints
outside the table. There is no feed-forward extrapolation. With an empty table,
Initial Current is the bias and the integral supplies the rest of the heating
current. P-only control needs an appropriate bias/map to reach elevated targets.
Do not treat current measured during a ramp as equilibrium current.

`Tune PI/PID` saves measured `pid_gain_schedule` points with `current_a`, `kp`,
`ki`, and `kd`. The schedule is evaluated at feed-forward current when a map is
present, or at filtered accepted current otherwise. `gain_schedule_filter_time_s`
defaults to 10 s. Changes to proportional gain preserve the integral-supported
holding correction. Ramp/hold transitions preserve that correction as well.

Save separate Material Profiles for Ni and NiCr, including their feed-forward
maps and gain schedules. The configuration writer supports TOML inline tables,
so these maps and schedules survive GUI saves.

## Commands, filtering, and anti-windup

- The default `max_current_step_up` and `max_current_step_down` are 0.001 A per
  reference period (`1/experiment_frequency`). Live PI slew limits scale with
  actual elapsed time. The physical output is rounded to the 1 mA programming
  grid, so instantaneous changes can differ from continuous requests by rounding.
- The next cycle starts from the setting actually transmitted, including when
  `minimum_current_change` suppresses a write. This setting is not a verified
  PSU readback; external manual changes or a rejected command require hardware
  diagnostics. The DMM current remains available for safety monitoring.
- `pid_integral_current_limit_a` bounds the correction in amps, also bounded by
  `max_current`. The old `pid_integral_limit` error-seconds setting is obsolete.
- `pid_tracking_time_s` defaults to 30 s. Back-calculation uses the accepted
  actuator command after slew limits, quantization, and recovery overrides.
  Differences below half a programming step do not unwind the integral, allowing
  small corrections to accumulate into a real command.
- Invalid/reused temperatures do not integrate error. Recovery overrides still
  update actuator tracking and clear heating-rate history.
- `temperature_rate_window_s` defaults to 8 s. Regression uses monotonic
  timestamps and at least three accepted measurements. Normal temperature
  median filtering is still controlled by `measurement_filter_samples`.
- `temperature_prediction_time_s` defaults to 2 s. When measured heating rate
  exceeds the programmed ramp rate, a short temperature prediction smoothly
  reduces the requested heat before target crossing. Set it to zero to disable.
  The programmed rate is zero during holds. The old normal catch-up and rate
  threshold settings no longer force current steps or reset the PI.

## Retuning and validation

Existing gains from the accumulating-current implementation are not transferable
to this absolute-current PI. Software tuning now permits small gains below the
old 0.001 Kp / 1e-5 Ki floors and uses an integral time at least as long as its
conservative response time (minimum 30 s). No new Ni/NiCr gains or feed-forward
currents are automatically fabricated from the old ramp data.

Validate resistance-derived temperature against an independent thermometer
before retuning. The reviewed NiCr runs reported T0 scatter equivalent to
33.5/58.7 C and a later resistance decrease while electrical power increased.
Feedback changes cannot establish actual wire temperature from that signal.
The reviewed exported curves also have straight extensions from approximately
284 C (Ni) and 293 C (NiCr) to 1000 C; these are not measured high-temperature
calibration points.

Begin with independently calibrated low/mid-temperature holds and small,
resolvable open-loop current rise/fall steps. Measure equilibrium feed-forward
points separately. Tune conservative PI for each wire geometry, save a Material
Profile, and evaluate settled hold error separately from ramp tracking error.
Keep derivative action disabled initially. Hardware tests are required to
establish the resulting temperature tolerance.
