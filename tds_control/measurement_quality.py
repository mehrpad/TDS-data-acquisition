"""Independent diagnostic guard for loss of credible resistance thermometry."""
import numpy as np


class ResistancePowerGuard:
    """Compare robust endpoints over a sustained interval, not individual noisy slopes."""

    def __init__(self, config):
        self.enabled = bool(config.get("resistance_power_guard_enabled", False))
        self.window = float(config.get("resistance_power_guard_window_s", 30.0))
        self.drop = float(config.get("resistance_power_guard_drop_c", 15.0))
        self.power_ratio = float(config.get("resistance_power_guard_power_ratio", 1.2))
        self.minimum_current = float(config.get("resistance_power_guard_min_current_a", .02))
        if (not all(np.isfinite(v) for v in (self.window, self.drop, self.power_ratio, self.minimum_current))
                or self.window <= 0 or self.drop <= 0 or self.power_ratio <= 1 or self.minimum_current < 0):
            raise ValueError("Invalid resistance/power guard configuration.")
        self.samples = []

    def reset(self):
        self.samples.clear()

    def update(self, timestamp, temperature, resistance, current, power, setpoint):
        if not self.enabled:
            return None
        row = (timestamp, temperature, resistance, abs(current), power, setpoint)
        if not all(np.isfinite(v) for v in row) or abs(current) < self.minimum_current:
            self.reset()
            return None
        if self.samples and timestamp <= self.samples[-1][0]:
            self.reset()
        self.samples.append(row)
        self.samples = [r for r in self.samples if timestamp - r[0] <= self.window]
        if len(self.samples) < 6 or timestamp - self.samples[0][0] < .8 * self.window:
            return None
        first = np.median(self.samples[:3], axis=0)
        last = np.median(self.samples[-3:], axis=0)
        if (last[5] >= first[5] and first[1] - last[1] >= self.drop
                and last[2] < first[2] and last[3] >= first[3] + .003
                and first[4] > 0 and last[4] >= first[4] * self.power_ratio):
            return ("Resistance-derived temperature fell while current and power increased. "
                    "Stopped: verify temperature sensing, sample resistance and thermal conditions.")
        return None
