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
- `experiment_mode`: choose `"CONTROLLED"` or `"CURVE_SWEEP"`
- `DMM_v`: VISA address for the voltage DMM
- `DMM_i`: VISA address for the current DMM
- `PS`: VISA address for the power supply
- `experiment_frequency`: control loop frequency in Hz
- `max_voltage`: absolute software voltage limit
- `max_current`: absolute software current limit
- `t0_voltage_search_start`: starting voltage for the `T0` search
- `t0_calibration_voltage`: highest voltage the `T0` search is allowed to use
- `t0_settle_time_s`: how long the wire is allowed to settle before `T0` samples are accepted
- `tuning_start_voltage`: starting voltage for the PI/PID tuning search
- `tuning_search_max_voltage`: highest voltage the PI/PID tuning search is allowed to use
- `tuning_response_voltage_step`: how much each PI/PID tuning attempt increases above the safe baseline voltage
- `curve_sweep_voltage_step`: basis used to derive the number of open-loop sweep steps from the GUI `Max Voltage`

### Fixed DMM range policy

Auto-ranging must remain **off** on both DMMs. Before power-supply output is enabled, the software locks the voltage DMM and current DMM to the smallest supported fixed DC range that covers `max_voltage` and `max_current`, respectively. It does not request `AUTO` range during a run.

Set `max_voltage` and `max_current` to positive, realistic limits before starting. For example, the current configuration (`30 V`, `3 A`) selects fixed `200 V` and `10 A` ranges. Verify the selected fixed ranges are appropriate for the installed DMMs and the expected measurement before every run; do not enable Auto Range manually on either meter.

Controller mode notes:

- `controller_mode = "PI"` is the default and recommended starting point
- set `controller_mode = "PID"` if you want derivative action enabled
- the `Tune PI/PID` button uses the selected mode from `config.toml`

Experiment mode notes:

- `experiment_mode = "CONTROLLED"` is the default mode
- `CONTROLLED` uses the loaded `R vs. T` curve, `T0` calibration, and PI/PID control to follow the programmed temperature path
- `CURVE_SWEEP` keeps `Calibrate T. Zero` available and requires it before `Start`, disables `Tune PI/PID`, ignores the experiment-program text box, and performs an open-loop voltage sweep
- in `CURVE_SWEEP`, the sweep starts from `curve_sweep_start_voltage` and ends at the smaller of the GUI `Max Voltage` and the software limit from `config.toml`
- the number of sweep points is calculated automatically as `ceil(Max Voltage / curve_sweep_voltage_step)`
- the loaded `R(T)` curve is used only to shape the voltage progression, not to claim a true physical `V(T)` law
- the shaping idea is simple: the software samples the loaded curve across temperature, converts that sampled curve into normalized fractions, and then maps those fractions onto the voltage range so the early and late parts of the sweep follow the same general order as the calibration curve
- `T0` calibration rescales that curve to the actual four-wire resistance of the current wire, so the live `R -> T` conversion uses the current wire rather than the unscaled reference values

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
6. Confirm the four-wire Kelvin contacts and set positive, conservative `Max Voltage` and `Max Current` limits. Verify both DMMs are on fixed DC ranges, not Auto Range.
7. Set an independent current limit/OCP on the power supply.
8. Click `Start`.

For `CURVE_SWEEP` mode:

1. Click `Find R vs. T` and select the calibration file.
2. Choose `CURVE_SWEEP` in the `Experiment Mode` selector.
3. Enter the actual equilibrated wire temperature in `Zero Temperature` and click `Calibrate T. Zero`.
4. Confirm the four-wire Kelvin contacts, set the GUI `Max Voltage` and `Max Current`, and verify both DMMs use fixed ranges.
5. Set an independent current limit/OCP on the power supply.
6. Click `Start`.

In this mode the program text box is disabled because the sweep shape comes from the loaded curve and the sweep resolution comes from `curve_sweep_voltage_step`.

## Experiment Program Format

The text box accepts one line per experiment step.

Example:

```text
{start_T=23;step_T=600;target_T=600;ramp_speed_c_min=10;hold_step_time_min=1}
```

Meaning:

- `start_T`: temperature where the programmed sequence begins
- `step_T`: step size in degrees C
- `target_T`: final temperature target
- `ramp_speed_c_min`: target ramp speed in degrees C per minute
- `hold_step_time_min`: hold time at each step in minutes

Behavior:

- If `step_T >= target_T - start_T`, the program becomes a simple ramp.
- Otherwise the loop ramps to each intermediate step, holds for the requested time, and stops as soon as the final plateau is reached.

## Safety Behavior

The control loop now includes:

- voltage step-up and step-down limits
- PID anti-windup
- rate limiting when temperature rises too quickly
- software current cutoff using `max_current`
- invalid-measurement detection
- automatic voltage shutdown at the end of the run

`max_current` is a software stop threshold after the current DMM is read; it is not a hardware current limiter. Set an independent current limit/OCP on the power supply before each run. Even with these protections, first runs on a new sample should be done with conservative limits and supervision.

## Data Output

Each run creates a folder like:

```text
data/<experiment_counter>_<experiment_name>/
```

Saved files:

- `data.csv`: continuously appended human-readable data file
- `data.h5`: continuously appended HDF5 data file
- `r_vs_t.csv`: the exact calibration curve used for that run
- `calibration_info.txt`: written when T0 calibration detected a large raw curve-to-T0 mismatch

Autosave runs in a background thread so disk writing does not block experiment control.

## Notes on Calibration

### `Calibrate T. Zero`

This function does not shift the temperature setpoint directly.
It measures the sample resistance near room temperature and rescales the loaded resistivity curve so the measured room-temperature point matches the entered `Zero Temperature`.
For the current wire, it constructs `R_cal(T) = R0 * rho_ref(T) / rho_ref(T0)` and then uses that calibrated `R(T)` curve for the live resistance-to-temperature conversion.

During this step the software now:

- starts at `t0_voltage_search_start` and increases only until a stable positive current is found,
- never goes above `t0_calibration_voltage` during that search,
- waits `t0_settle_time_s` before collecting data,
- treats the entered `Zero Temperature` as the calibration anchor instead of rejecting samples based on the uncalibrated curve,
- shows and records a warning when the raw inferred temperature differs from T0 by more than `t0_max_temp_error_c`,
- discards warmup readings and rejects invalid resistance/current readings or obvious resistance outliers before calculating the final scale.

### `Tune PI/PID`

PI/PID tuning uses a guarded low-rise step response:

- it starts at `tuning_start_voltage` and increases only until a stable positive current is found,
- it uses that lowest stable-current voltage as a safe baseline voltage,
- it then applies a real step above that baseline and only increases the response voltage in bounded attempts up to `tuning_search_max_voltage`,
- it measures the baseline temperature before each response attempt,
- stops once the requested small temperature rise is reached,
- estimates conservative gains for the selected controller mode.

Mode details:

- `PI` mode is the default and leaves `Kd = 0`
- `PID` mode also estimates a conservative derivative term and uses it during the real experiment

This is intended to reduce aggressive heating before the real experiment starts.

### Bounded curve extrapolation

`curve_extrapolation_enabled = true` extends a monotonic calibrated R-vs-T curve to the configured
`curve_extrapolation_min_temperature_c` and `curve_extrapolation_max_temperature_c` limits. The
measured portion remains piecewise-linear between adjacent file rows. Outside that portion, the
software fits a straight line to at least `curve_extrapolation_fit_points` points at the corresponding end.
Each endpoint fit also covers at least `curve_extrapolation_min_fit_span_c`, so dense or noisy tables do not
produce a false flat/reversed endpoint from only a tiny temperature interval. Small resistance reversals
caused by table rounding or measurement noise may be corrected only within the configured correction ratio.

The default configured conversion range is 0 to 600 C. Extrapolated temperatures are estimates,
not measured calibration data. The application rejects non-monotonic curves and experiment targets
outside the configured limits; use reference or measured data covering the full range whenever possible.

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
