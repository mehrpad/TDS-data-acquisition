"""Save and load named per-material controller/measurement profiles.

A profile captures the settings that are specific to one sample/material -
above all the PID gain schedule from multi-point tuning and the current
step limits derived from it, plus the other measurement settings that go
with a given wire - so a user can tune once per material and reuse the
result in a future session without re-tuning.

Profiles are separate JSON files under files/material_profiles/, distinct
from the day-to-day config.toml, so switching samples does not require
re-editing (or losing) a previous material's tuned values.
"""
import json

from .paths import FILES_DIR

PROFILES_DIR = FILES_DIR / "material_profiles"

# Fields captured in a profile: the tuning result plus the other settings
# that realistically change together with a sample/material.
PROFILE_FIELDS = (
    "pid_gain_schedule",
    "pid_kp",
    "pid_ki",
    "pid_kd",
    "controller_mode",
    "max_current_step_up",
    "max_current_step_down",
    "measurement_temperature_jump_guard_enabled",
    "resistivity_mode",
    "dmm_voltage_range_v",
    "dmm_current_range_a",
    "dmm_resistance_range_ohm",
    "max_current",
    "max_power_w",
    "max_sample_voltage",
    "compliance_voltage",
    "resistivity_heat_time_s",
    "resistivity_measure_time_s",
    "resistivity_output_settle_s",
)


def _sanitize_profile_name(name):
    cleaned = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in str(name).strip())
    cleaned = cleaned.strip(" _")
    if not cleaned:
        raise ValueError("Profile name must contain at least one letter, digit, space, - or _.")
    return cleaned


def profile_path(name):
    return PROFILES_DIR / f"{_sanitize_profile_name(name)}.json"


def list_profiles():
    """Return saved profile names, sorted, without touching the filesystem if empty."""
    if not PROFILES_DIR.exists():
        return []
    return sorted(path.stem for path in PROFILES_DIR.glob("*.json"))


def save_profile(name, config):
    """Snapshot the material-specific fields of config under a saved name."""
    sanitized_name = _sanitize_profile_name(name)
    data = {field: config[field] for field in PROFILE_FIELDS if field in config}
    data["profile_name"] = sanitized_name
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = profile_path(sanitized_name)
    with path.open("w", encoding="utf-8") as profile_file:
        json.dump(data, profile_file, indent=2, sort_keys=True)
    return path


def load_profile(name):
    """Return the saved fields for a profile as a plain dict."""
    path = profile_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No saved profile named {name!r} at {path}.")
    with path.open("r", encoding="utf-8") as profile_file:
        return json.load(profile_file)


def delete_profile(name):
    path = profile_path(name)
    if path.exists():
        path.unlink()
        return True
    return False
