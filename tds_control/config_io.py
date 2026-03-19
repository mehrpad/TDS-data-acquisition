import json

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

from .paths import CONFIG_PATH, LEGACY_CONFIG_PATH, ensure_runtime_dirs


CONFIG_GROUPS = [
    (
        "Controller mode and hardware",
        [
            ("controller_mode", 'Choose "PI" or "PID". PI is the default and recommended starting mode.'),
            ("experiment_frequency", "Control loop frequency in Hz."),
            ("max_voltage", "Absolute software voltage limit for the power supply."),
            ("max_current", "Absolute software current limit in amps."),
            ("DMM_speed", "Measurement speed setting sent to both DMMs."),
            ("DMM_v", "VISA address of the voltage DMM."),
            ("DMM_i", "VISA address of the current DMM."),
            ("PS", "VISA address of the power supply."),
        ],
    ),
    (
        "Controller gains and output shaping",
        [
            ("pid_kp", "Proportional gain used by the live controller."),
            ("pid_ki", "Integral gain used by the live controller."),
            ("pid_kd", 'Derivative gain used only when controller_mode = "PID".'),
            ("pid_integral_limit", "Clamp for the internal integral state."),
            ("pid_derivative_filter", "Low-pass filter factor for derivative smoothing."),
            ("startup_voltage", "Initial PSU voltage at the start of an experiment."),
            ("min_voltage", "Lowest voltage the controller may request."),
            ("max_voltage_step_up", "Largest normal upward voltage step per loop."),
            ("max_voltage_step_down", "Largest downward voltage step per loop."),
            ("max_voltage_step_up_far", "Larger upward step allowed when the sample is far below the setpoint."),
            ("aggressive_step_band_c", "Temperature gap below the setpoint that enables aggressive catch-up."),
            ("rate_limit_activation_band_c", "Band around the setpoint where heating-rate limiting becomes active."),
            ("under_target_no_decrease_band_c", "Below this band the controller avoids decreasing voltage."),
        ],
    ),
    (
        "Safety, filtering, and invalid-reading handling",
        [
            ("temperature_tolerance_c", "Main temperature tolerance around the setpoint."),
            ("hold_entry_tolerance_c", "Tolerance allowed when entering a hold plateau."),
            ("safety_temp_margin_c", "Extra safety margin above the final target temperature."),
            ("soft_temp_rate_margin_c_min", "Extra ramp-rate margin before soft limiting starts."),
            ("hard_temp_rate_margin_c_min", "Extra ramp-rate margin before hard backoff starts."),
            ("measurement_fail_limit", "How many consecutive invalid measurements are tolerated before stopping."),
            ("minimum_current_a", "Smallest current treated as a valid reading."),
            ("minimum_voltage_change", "Smallest PSU change worth sending to the instrument."),
            ("measurement_voltage_floor", "Minimum allowed PSU voltage during active control."),
            ("measurement_filter_samples", "Median filter window for accepted temperature readings."),
            ("resistance_range_margin_ratio", "Extra fractional margin allowed outside the loaded R-vs-T range."),
            ("resistance_range_margin_ohm", "Extra absolute resistance margin allowed outside the loaded R-vs-T range."),
            ("warmup_stable_samples", "How many stable warmup samples are needed before the main ramp starts."),
            ("resistance_glitch_jump_ohm", "Base resistance-jump threshold before a retry is triggered."),
            ("resistance_glitch_jump_ratio", "Additional relative resistance-jump threshold used at higher resistance."),
            ("measurement_retry_attempts", "How many extra measurements are taken after a resistance jump."),
            ("measurement_retry_delay_s", "Delay between retry measurements after a resistance jump."),
            ("measurement_retry_consensus_ohm", "Allowed spread for stable retry consensus."),
            ("stable_current_invalid_advance_count", "How many invalid search samples at one voltage are allowed before stepping higher."),
            ("measurement_temp_jump_c", "Legacy symmetric temperature-jump threshold kept for compatibility."),
            ("measurement_temp_jump_up_c", "Upward temperature jump threshold before a reading is questioned."),
            ("measurement_temp_jump_down_c", "Downward temperature jump threshold before a reading is questioned."),
            ("measurement_temp_jump_accept_up_c", "Largest upward jump that may still be accepted as physically plausible."),
            ("measurement_temp_jump_accept_setpoint_margin_c", "How far above the live setpoint an accepted upward jump may land."),
            ("measurement_cooldown_confirm_samples", "How many downward jump confirmations are required before accepting cooldown."),
            ("ignore_invalid_below_voltage", "Below this PSU voltage, invalid readings are treated more leniently."),
            ("invalid_voltage_step_down", "Backoff step used when an invalid reading appears too hot."),
        ],
    ),
    (
        "Autosave",
        [
            ("autosave_flush_interval_s", "How often buffered data is flushed to disk."),
            ("autosave_batch_size", "How many samples are batched before a forced autosave write."),
        ],
    ),
    (
        "Controller tuning",
        [
            ("tuning_voltage_step", "Voltage increment while searching for the safe tuning baseline."),
            ("tuning_start_voltage", "Starting PSU voltage for controller tuning."),
            ("tuning_search_max_voltage", "Maximum PSU voltage allowed during controller tuning."),
            ("tuning_settle_time_s", "Settling delay before a tuning search sample is judged."),
            ("tuning_response_voltage_step", "Step added above the baseline during each tuning attempt."),
            ("tuning_between_attempts_s", "Pause between tuning attempts."),
            ("tuning_max_duration_s", "Maximum duration of one tuning response measurement."),
            ("tuning_baseline_samples", "How many baseline samples are required before a tuning step."),
            ("tuning_stable_current_samples", "How many stable-current samples define the tuning baseline."),
            ("tuning_stable_current_a", "Minimum stable current for the tuning baseline."),
            ("tuning_temperature_window_c", "Allowed temperature window around the hinted base temperature during tuning."),
            ("tuning_target_rise_c", "Preferred temperature rise for the tuning step response."),
            ("tuning_min_temperature_rise_c", "Minimum temperature rise needed for a usable tuning result."),
            ("tuning_no_response_timeout_s", "How long tuning waits before concluding that nothing is happening."),
            ("tuning_min_observable_rise_c", "Minimum smoothed rise considered observable during tuning."),
            ("tuning_plateau_timeout_s", "Maximum time allowed once the tuning response has plateaued."),
            ("tuning_plateau_idle_timeout_s", "Idle timeout once the smoothed rise stops improving."),
            ("tuning_plateau_growth_c", "Rise improvement required to count as continued growth during tuning."),
        ],
    ),
    (
        "T0 calibration",
        [
            ("t0_calibration_voltage", "Highest voltage allowed during T0 calibration."),
            ("t0_voltage_search_start", "Starting PSU voltage for the T0 stable-current search."),
            ("t0_voltage_step", "Voltage increment used during the T0 stable-current search."),
            ("t0_settle_time_s", "How long the sample is allowed to settle before T0 sampling."),
            ("t0_calibration_samples", "How many T0 samples are used for the final calibration point."),
            ("t0_warmup_samples", "How many initial T0 samples are discarded as warmup."),
            ("t0_stable_current_samples", "How many stable-current samples define the T0 baseline."),
            ("t0_stable_current_a", "Minimum stable current for the T0 baseline."),
            ("t0_max_temp_error_c", "Largest allowed mismatch from the entered base temperature during T0 calibration."),
        ],
    ),
]


def _format_toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported config value type: {type(value).__name__}")


def load_config():
    ensure_runtime_dirs()
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as file:
            return tomllib.load(file)

    if LEGACY_CONFIG_PATH.exists():
        with LEGACY_CONFIG_PATH.open(encoding="utf-8") as file:
            data = json.load(file)
        save_config(data)
        return data

    raise FileNotFoundError(
        f"Configuration file not found. Expected {CONFIG_PATH} or legacy {LEGACY_CONFIG_PATH}."
    )


def save_config(config):
    ensure_runtime_dirs()
    lines = [
        "# TDS control configuration",
        "#",
        "# This file uses TOML so you can keep comments next to the settings.",
        "# The GUI will rewrite this file with comments preserved whenever it saves settings.",
        "#",
        "",
    ]
    written_keys = set()

    for group_name, entries in CONFIG_GROUPS:
        lines.append(f"# {group_name}")
        for key, comment in entries:
            if key not in config:
                continue
            lines.append(f"# {comment}")
            lines.append(f"{key} = {_format_toml_value(config[key])}")
            lines.append("")
            written_keys.add(key)

    extra_keys = [key for key in config if key not in written_keys]
    if extra_keys:
        lines.append("# Additional settings")
        for key in sorted(extra_keys):
            lines.append(f"# Extra setting preserved by the application.")
            lines.append(f"{key} = {_format_toml_value(config[key])}")
            lines.append("")

    CONFIG_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
