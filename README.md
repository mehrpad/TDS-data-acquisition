# TDS Control Software

Desktop software for running a temperature-programmed resistivity experiment with:

- one Siglent power supply,
- one DMM for voltage,
- one DMM for current,
- a loaded resistivity-vs-temperature calibration curve.

The application is written with PyQt6 and PyVISA. It controls the power supply, estimates sample temperature from resistance, follows a temperature program, and saves data continuously while the experiment is running.

![TDS GUI](files/gui.png)

![Measurement map](files/map.jpg)

## Wiring: required four-wire (Kelvin) connection

Use a true Kelvin connection. The current path and voltage-sense path must be separate all the way to the sample:

```text
                                  CURRENT PATH
PS+ -- Ammeter -- force cable -- o======== SAMPLE ========o -- force cable -- PS-
                                  |                        |
                              V+ sense                  V- sense
                                  \                        /
                                   \---- voltage DMM -----/
                                      high-impedance sense wires
```

- `DMM_i` is the ammeter in series, so all source current flows through it and the two force leads.
- `DMM_v` is the voltage DMM. Connect its `V+` and `V-` leads directly to the sample, ideally to separate sense contacts inside the current contacts.
- Do **not** connect the voltage DMM at the power-supply ends of the cables, at the ammeter terminals, or at other points outside the sample contacts.
- The voltage DMM draws negligible current, so the resistance of its own sense wires is practically irrelevant.

With cable-end/two-wire voltage sensing, the reading includes unwanted lead and contact resistance:

```text
R_measured = R_sample + R_contacts + R_cables
```

With the required Kelvin connection, the voltage DMM measures the sample drop:

```text
V_DMM ≈ I × R_sample
R_sample = V_DMM / I
```

Do not connect the current meter in parallel across the sample, because that can short the source and damage the setup. Keep `fixed_series_resistance_ohm = 0` for a Kelvin measurement unless a real, separately verified series component must be removed from the calculation. The loaded `R vs. T` calibration must represent the same Kelvin sample resistance.

See [the measurement setup guide](docs/MEASUREMENT_SETUP.md) for the wiring checklist and the prohibited two-wire layout.

## Features

- Load an `R vs. T` file from `.xlsx` or `.csv`
- Auto-load the selected file immediately after `Find R vs. T`
- Reload the same file later with `Reload R vs. T`
- Calibrate the room-temperature `T0` reference against the loaded curve
- Tune conservative PI/PID gains from a low-rise step test before the experiment
- Run stepped ramps or a simple continuous ramp
- Live plots for temperature and flux placeholder
- Continuous background autosave to:
  - `data.csv`
  - `data.h5`
  - `r_vs_t.csv`
- Close confirmation when exiting the GUI

## Requirements

Install Python packages from [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

The project currently depends on:

- `pyqt6`
- `pyvisa`
- `pyvisa-py`
- `pandas`
- `openpyxl`
- `h5py`
- `scipy`
- `pyqtgraph`

You also need working VISA access for the connected instruments.

## Configuration

The instrument addresses and control defaults now live in [files/config.toml](files/config.toml).

This file uses TOML instead of JSON so comments can be kept directly in the file.
Each setting has a short explanation above it to make hand-editing easier for normal users.
If you still have an older `files/config.json`, the app will import it automatically the first time and write a new `config.toml`.

Important fields:

- `controller_mode`: choose `"PI"` or `"PID"`
- `experiment_mode`: choose `"TEMPERATURE"` or `"VOLTAGE"`
- `DMM_v`: VISA address for the voltage DMM
- `DMM_i`: VISA address for the current DMM
- `PS`: VISA address for the power supply
- `experiment_frequency`: control loop frequency in Hz; default `0.5 Hz` (one update every 2 seconds)
- `max_power_w`: GUI sample-power cutoff calculated as `|Vsample x I|`; default `2.5 W`
- `max_voltage`: internal absolute PSU-voltage ceiling retained as a last-resort guard
- `max_current`: absolute software current limit
- `dmm_voltage_range_v`: initial fixed voltage-DMM range; default `0.2 V`
- `dmm_current_range_a`: initial fixed current-DMM range; default `0.002 A` (2 mA)
- `dmm_staged_ranging_enabled`: enables deterministic upward fixed-range changes without instrument autorange
- `dmm_range_switch_fraction`: upward range-change threshold; default `0.8` (80% of full scale)
- `dmm_range_settle_time_s` / `dmm_range_discard_readings`: settling controls after a range change
- `dmm_range_recovery_attempts`: maximum upward range changes within one measurement; default `5`
- `dmm_synchronized_reading`: start both DMM conversions before fetching either value
- `startup_settle_time_s`: short delay after applying Initial Voltage before control begins
- `low_voltage_max_step_up` / `low_voltage_max_step_down`: fine controller steps used near zero voltage
- `t0_voltage_search_start`: starting voltage for the `T0` search
- `t0_calibration_voltage`: highest voltage the `T0` search is allowed to use
- `t0_settle_time_s`: how long the wire is allowed to settle before `T0` samples are accepted
- `tuning_start_voltage`: starting voltage for the PI/PID tuning search
- `tuning_search_max_voltage`: highest voltage the PI/PID tuning search is allowed to use
- `tuning_response_voltage_step`: how much each PI/PID tuning attempt increases above the safe baseline voltage

### Staged fixed DMM range policy

Auto-ranging must remain **off** on both DMMs. Before power-supply output is enabled, the software locks the
meters to the initial `dmm_voltage_range_v` and `dmm_current_range_a` settings. The defaults are the SDM3055
`0.2 V` voltage range and `0.002 A` (2 mA) current range, giving good resolution for low-voltage startup.
When `dmm_staged_ranging_enabled = true`, a reading at 80% of full scale moves that meter directly to the
smallest larger supported fixed range that covers the signal. Voltage progresses through 0.2, 2, 20, 200, and
1000 V; current progresses through 0.2 mA, 2 mA, 20 mA, 200 mA, 2 A, and 10 A. A range never moves downward
during one operation, avoiding range chatter.

With `dmm_synchronized_reading = true`, the software sends `INIT` to both DMMs before fetching either result.
This greatly reduces the time offset that occurs when two blocking `READ?` commands are executed sequentially.
If synchronized acquisition is unavailable, that individual sample falls back to the older `READ?` method and
prints a warning.

After a range change, the software waits `dmm_range_settle_time_s`, discards
`dmm_range_discard_readings` synchronized pairs, and then uses a fresh pair. This prevents a transition sample
from entering the resistance or temperature calculation. The SDM3055 remote overload response
`+9.90000000E+37` is detected explicitly. An overloaded meter is moved upward by one fixed range and measured
again, up to `dmm_range_recovery_attempts` times inside the same sample. These transitions do not count as
experiment measurement failures. If overload remains on the largest range, that sample is rejected safely.
`max_power_w`, `max_current`, and the internal
`max_voltage` ceiling remain independent software safety limits; the power supply must also have an appropriate
hardware current limit/OCP.
Do not enable Auto Range manually during calibration, tuning, or an experiment.

### Low-voltage startup policy

The GUI `Initial Voltage` is shared by T0 calibration, PI/PID tuning, Temperature mode, and Voltage mode.
It is an enforced runtime floor: an experiment starts directly at this voltage and cannot request a
lower value. There is no separate experiment-start measurement search. After `startup_settle_time_s`, the normal
controller loop reads the meters and increases, holds, or decreases voltage from live measurements, while the
Initial Voltage floor remains enforced. At or below `low_voltage_step_threshold`, controller, recovery, T0-search,
and tuning-search actions use the finer low-voltage step sizes.

Logged `PSU command` values are requested power-supply setpoints. They are not the same as `Vsample`, which is the
Kelvin voltage measured directly across the wire, and either value may differ slightly from the PSU front-panel
readback because of output accuracy, display resolution, settling, and wiring voltage drop.

In Temperature mode, the programmed temperature target starts from calibrated T0 and advances according to
`ramp_speed_min`, interpreted as degrees C per minute.
Reaching `start_T` changes the ramp phase but does not wait for the measured temperature, so noisy or lagging
measurements cannot freeze the target. Deliberately configured step holds still pause the target as requested.

At low PSU voltage, one large inferred-temperature jump cannot replace calibrated T0 or the last trusted
temperature. The controller initially holds its present voltage while checking the candidate and requires
`low_signal_jump_confirm_samples` matching temperature-and-resistance readings (default `3`). If the jump is not
confirmed, subsequent invalid readings continue from the prior trusted temperature instead of reusing the spike.
This confirmation does not pause the programmed target ramp.

If low-signal readings remain invalid, the controller no longer stays indefinitely at Initial Voltage. After five
consecutive invalid readings it increases the commanded PSU voltage by `0.01 V`, observes five more measurements,
and repeats when necessary. It performs up to five upward probes in one recovery episode, while still enforcing
`max_power_w`, `max_voltage`, `max_current`, and the Initial Voltage floor. One valid measurement resets this recovery sequence;
the programmed temperature target continues to ramp throughout it.

Controller mode notes:

- `controller_mode = "PI"` is the default and recommended starting point
- set `controller_mode = "PID"` if you want derivative action enabled
- the `Tune PI/PID` button uses the selected mode from `config.toml`

Experiment mode notes:

- `experiment_mode = "TEMPERATURE"` is the default
- `TEMPERATURE` uses the loaded `R vs. T` curve, T0 calibration, and PI/PID control to follow the programmed temperature path
- `VOLTAGE` disables PI/PID tuning and ramps the PSU command from Initial Voltage according to elapsed time
- in `VOLTAGE`, `ramp_speed_min` is volts per minute; for example, `0.001` adds 0.001 V after one minute
- Voltage mode does not use inferred temperature as feedback, but still logs temperature while resistance remains inside the calibrated range
- Voltage mode continues until the user stops it or a power, current, or internal voltage safety limit is reached
- both modes require T0 calibration so the displayed and saved resistance-to-temperature conversion is anchored to the current wire

The software also stores tuned controller gains and autosave defaults in this file after you run the GUI.

## R vs. T File Format

Accepted input:

- Excel `.xlsx`
- CSV `.csv`

Required columns:

- `resistivity`
- `temperature [C]`

Also accepted for reloads created by this app:

- `temperature`

## Running the GUI

Start the application with:

```bash
python -m tds_control
```

or keep using the compatibility launcher:

```bash
python TDS.py
```

## Typical Workflow

1. Click `Find R vs. T` and select the calibration file.
   The file is loaded automatically.
2. Check the `Zero Temperature (°C)` value.
   This should be the actual room or base temperature.
3. Click `Calibrate T. Zero`.
   This rescales the loaded resistivity curve so the measured low-voltage room-temperature resistance matches the entered `Zero Temperature`.
4. Optional: click `Tune PI/PID`.
   The software performs a guarded low-voltage step test with a small temperature rise and stores the tuned gains in `files/config.toml`.
   By default this tunes a PI controller because `controller_mode = "PI"` is the default.
5. Enter the experiment program in the text box.
6. Confirm the four-wire Kelvin contacts and set conservative `Max Power` and `Max Current` limits. Verify both DMMs are on fixed DC ranges, not Auto Range.
7. Set an independent current limit/OCP on the power supply.
8. Click `Start`.

For `VOLTAGE` mode:

1. Click `Find R vs. T` and select the calibration file.
2. Choose `VOLTAGE` in the `Mode` selector.
3. Enter the actual equilibrated wire temperature in `Zero Temperature` and click `Calibrate T. Zero`.
4. Enter a voltage ramp such as `{ramp_speed_min=0.001}`.
5. Set conservative `Max Power` and `Max Current` limits and verify both DMMs use fixed ranges.
6. Set an independent current limit/OCP on the power supply.
7. Click `Start` and supervise the run.

## Experiment Program Format

In Temperature mode, the text box accepts one line per experiment step.

Example:

```text
{start_T=23;step_T=200;target_T=200;ramp_speed_min=10;hold_step_time_min=1}
```

Meaning:

- `start_T`: temperature where the programmed sequence begins
- `step_T`: step size in degrees C
- `target_T`: final temperature target
- `ramp_speed_min`: target ramp speed in degrees C per minute
- `hold_step_time_min`: hold time at each step in minutes

Behavior:

- If `step_T >= target_T - start_T`, the program becomes a simple ramp.
- Otherwise the loop ramps to each intermediate step, holds for the requested time, and stops as soon as the final plateau is reached.

In Voltage mode, enter one parameter:

```text
{ramp_speed_min=0.001}
```

Here `ramp_speed_min` is volts per minute. The requested elapsed-time ramp is also constrained by the normal
per-loop voltage slew limits. There is no temperature target in Voltage mode.

## Safety Behavior

The control loop now includes:

- voltage step-up and step-down limits
- PID anti-windup
- rate limiting when temperature rises too quickly
- software current cutoff using `max_current`
- measured sample-power cutoff using `max_power_w`
- invalid-measurement detection
- direct controlled startup at the enforced Initial Voltage floor
- time-driven target advancement at the configured ramp speed
- repeated confirmation before a large low-signal reading replaces T0 or the last trusted temperature
- five staged `+0.01 V` recovery probes when invalid low-signal measurements would otherwise stall control
- an enforced Initial Voltage floor and 0.001 V low-voltage micro-steps
- controlled micro-voltage probing before a large resistance/temperature jump is accepted
- one PSU output-enable command at operation start, followed by a 0.001 V keep-alive setpoint at the end

`max_current` and `max_power_w` are software stop thresholds after synchronized DMM readings; they are not
hardware limiters. The 2.5 W default is a general ceiling, not a validated safe value for a thin wire. Your
100-micrometre NiCr sample glowed below 1 W in the reviewed run, so use a substantially lower independently
validated limit for that sample. Set an independent current limit/OCP on the power supply before each run.

When an inferred-temperature jump is at least `measurement_jump_probe_threshold_c`, the controller does not immediately trust it. An upward jump causes a small PSU decrease; a downward jump causes a small increase only while the measured current is safely below `max_current`. Follow-up temperatures must remain inside the fixed window centered on the first probe candidate (default +/-50 C), and resistance must also be consistent before a new reference is established. The reference window does not move when a sample is inconsistent. By default, an unstable probe may try for 20 cycles before stopping rather than continuing with stale data.

## Data Output

Each run creates a folder like:

```text
data/<experiment_counter>_<experiment_name>/
```

Saved files:

- `data.csv`: continuously appended human-readable data file
- `data.xlsx`: Excel workbook containing the completed experiment data and the corrected curve used by the run
- `data.h5`: continuously appended HDF5 data file
- `r_vs_t.csv`: the exact calibration curve used for that run
- `corrected_r_vs_t_curve.pdf`: plotted temperature-versus-corrected-resistance curve used for conversion
- `calibration_info.txt`: written when T0 calibration detected a large raw curve-to-T0 mismatch

Autosave runs in a background thread so disk writing does not block experiment control.
The `P` / `sample_power` column records `|Vsample x I|` in watts.
The `R_ohm` / `calculated_resistance` value records the corrected Kelvin resistance
`Vsample / I - fixed_series_resistance_ohm` in both Temperature and Voltage modes.
For compatibility with the GUI's historical terminology, HDF5 also exposes the same data through
the `calculated_resistivity` dataset name. These values are resistance in ohms, not bulk resistivity
in ohm-metre, because the software does not have the wire length and cross-sectional area.
The Excel workbook's `Corrected R vs T` sheet and the PDF are produced from the same post-T0 curve
that the controller used during that experiment.

## Notes on Calibration

### `Calibrate T. Zero`

This function does not shift the temperature setpoint directly.
It measures the sample resistance near room temperature and rescales the loaded resistivity curve so the measured room-temperature point matches the entered `Zero Temperature`.
For the current wire, it constructs `R_cal(T) = R0 * rho_ref(T) / rho_ref(T0)` and then uses that calibrated `R(T)` curve for the live resistance-to-temperature conversion.

During this step the software now:

- starts at `t0_voltage_search_start` and increases in bounded micro-steps until both current and resistance are stable,
- never goes above `t0_calibration_voltage` during that search,
- waits `t0_settle_time_s` before collecting data,
- treats the entered `Zero Temperature` as the calibration anchor instead of rejecting samples based on the uncalibrated curve,
- shows and records a warning when the raw inferred temperature differs from T0 by more than `t0_max_temp_error_c`,
- uses nine synchronized voltage/current pairs by default,
- discards warmup readings and rejects invalid resistance/current readings or robustly detected resistance outliers before calculating the final scale,
- reports the temperature-equivalent spread of the accepted T0 resistances and warns when it exceeds `t0_temperature_spread_warning_c`.

The T0 value printed for accepted calibration samples is the entered anchor temperature, not an independent
temperature measurement. For a low-TCR wire, use the reported equivalent spread to judge whether the electrical
signal is precise enough for temperature control.

### `Tune PI/PID`

PI/PID tuning uses a guarded low-rise step response:

- it starts at `tuning_start_voltage` and increases in bounded micro-steps until both current and resistance are stable,
- it uses that lowest stable current-and-resistance voltage as a safe baseline voltage,
- it then applies a real step above that baseline and only increases the response voltage in bounded attempts up to `tuning_search_max_voltage`,
- it measures the baseline temperature before each response attempt,
- stops once the requested small temperature rise is reached,
- estimates conservative gains for the selected controller mode.

Mode details:

- `PI` mode is the default and leaves `Kd = 0`
- `PID` mode also estimates a conservative derivative term and uses it during the real experiment

This is intended to reduce aggressive heating before the real experiment starts.

### Bounded curve extrapolation

Curve extrapolation is disabled by default. In this safe configuration, the software accepts only experiment
temperatures covered by the loaded calibration curve and rejects an out-of-range program before opening the
instruments or enabling the PSU. A completed T0 calibration adds its measured resistance and entered temperature
as an explicit curve endpoint, so a small gap between room temperature and the source file's first furnace point
does not require extrapolation.

Setting `curve_extrapolation_enabled = true` explicitly extends a calibrated R-vs-T curve to the configured
`curve_extrapolation_min_temperature_c` and `curve_extrapolation_max_temperature_c` limits. Outside the
measured range, the software fits a straight line to at least `curve_extrapolation_fit_points` points at the
corresponding end. Each endpoint fit also covers at least `curve_extrapolation_min_fit_span_c`, so dense
tables do not create a false flat or reversed slope from only a tiny temperature interval.

Whether extrapolation is enabled or disabled, the curve is made single-valued and monotonic before resistance
is converted to temperature. Clean curves keep their original temperature resolution and receive only a
least-squares monotonic correction within `curve_monotonic_correction_ratio`. When
`curve_smoothing_enabled = true`, a dense curve that exceeds that limit can be recovered without modifying its
source file: the software groups rows into
`curve_smoothing_temperature_bin_c` bins, uses the smallest centered-median window that works up to
`curve_smoothing_max_window_c`, and then applies a least-squares monotonic fit. Smoothing is allowed only for
files with at least `curve_smoothing_min_points` rows, and the 99th-percentile raw deviation must stay below
`curve_smoothing_max_residual_ratio` of the curve's total resistance span. The console reports the selected
window, correction, and residual whenever this recovery path is used.

The configured extrapolation limits remain 0 to 600 C, but they have no effect while extrapolation is disabled.
Extrapolated temperatures are estimates rather than measured calibration data and must not be used for
closed-loop heating unless they have been independently validated for the particular material and geometry.
The application rejects experiment targets outside the active conversion range. Use reference or measured data
covering the full experiment range whenever possible.

## Development

The main implementation now lives in the package directory:

```text
tds_control/
```

Key modules:

- `tds_control/app.py`: GUI and application entry point
- `tds_control/tds_experiment.py`: experiment loop and safety logic
- `tds_control/calibration.py`: `T0` calibration and PI/PID tuning
- `tds_control/data_saver.py`: background CSV/HDF5 autosave
- `tds_control/pid.py`: PID controller
- `tds_control/siglent.py`: instrument SCPI helpers

If you edit the Qt Designer file and regenerate Python code:

```bash
pyuic6 -x files/TDS.ui -o tds_control/app.py
```

If you regenerate `tds_control/app.py` from the `.ui` file, remember that manual logic changes in the Python file will be overwritten unless they are merged back in.
