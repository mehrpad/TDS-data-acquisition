





def _clamp(value, limits):
    lower, upper = limits
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    return value


class PIDController:
    def __init__(
        self,
        kp,
        ki,
        kd,
        setpoint,
        output_limits=(None, None),
        integral_limits=(None, None),
        derivative_filter=0.0,
    ):
        """
        Initialize a PID controller that supports output limiting and anti-windup.

        Output and integral state are in actuator units (amps for wire heating).
        The output is absolute: callers must not add it to the previous command.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_filter = derivative_filter

        self.previous_error = 0.0
        self.previous_measurement = None
        self.integral = 0.0
        self.derivative = 0.0
        self.output = 0.0
        self.requested_output = 0.0
        self.previous_setpoint = None
        self.operating_current = None

    def update_setpoint(self, setpoint):
        """
        Update the controller setpoint.
        """
        self.setpoint = setpoint

    def reset(self, measurement=None, preserve_integral=False):
        """
        Reset the dynamic controller state between experiment phases.
        """
        self.previous_error = 0.0
        self.previous_measurement = measurement
        if not preserve_integral:
            self.integral = 0.0
        self.derivative = 0.0
        self.output = 0.0

    def set_gains(self, kp, ki, kd):
        # Keep the last correction continuous when scheduled Kp changes.
        if self.ki > 0 or ki > 0:
            self.integral = _clamp(
                self.integral + (self.kp - kp) * self.previous_error,
                self.integral_limits,
            )
        self.kp, self.ki, self.kd = kp, ki, kd

    def track_output(self, applied_output, dt, tracking_time_s=30.0, deadband=0.0):
        """Back-calculate from the accepted actuator setting, including overrides.

        Ignore sub-resolution differences so small integral corrections can
        accumulate across the hardware's quantization grid.
        """
        difference = applied_output - self.requested_output
        if self.ki > 0 and abs(difference) > deadband + 1e-12:
            weight = dt / (max(tracking_time_s, 0.0) + dt)
            self.integral = _clamp(self.integral + weight * difference, self.integral_limits)
        self.output = applied_output

    def compute(self, current_temperature, dt=1.0, setpoint=None, bias=0.0, integrate=True):
        """
        Compute the control output for the current measurement.

        :param current_temperature: Current measured temperature.
        :param dt: Loop time in seconds.
        :param setpoint: Optional setpoint update.
        :return: Requested control output.
        """
        if setpoint is not None:
            self.setpoint = setpoint

        if dt <= 0:
            dt = 1.0

        error = self.setpoint - current_temperature
        proportional = self.kp * error

        candidate_integral = self.integral + (self.ki * error * dt if integrate else 0.0)
        candidate_integral = _clamp(candidate_integral, self.integral_limits)
        integral_term = candidate_integral

        derivative_term = 0.0
        if self.previous_measurement is not None:
            measurement_slope = (current_temperature - self.previous_measurement) / dt
            raw_derivative = -self.kd * measurement_slope
            if 0.0 < self.derivative_filter < 1.0:
                derivative_term = (
                    self.derivative_filter * self.derivative
                    + (1.0 - self.derivative_filter) * raw_derivative
                )
            else:
                derivative_term = raw_derivative

        unclamped_output = bias + proportional + integral_term + derivative_term
        output = _clamp(unclamped_output, self.output_limits)

        at_upper_limit = self.output_limits[1] is not None and output >= self.output_limits[1]
        at_lower_limit = self.output_limits[0] is not None and output <= self.output_limits[0]
        should_integrate = (
            unclamped_output == output
            or (at_upper_limit and error < 0)
            or (at_lower_limit and error > 0)
        )
        if integrate and should_integrate:
            self.integral = candidate_integral

        self.previous_error = error
        self.previous_measurement = current_temperature
        self.derivative = derivative_term
        self.output = output
        self.requested_output = unclamped_output
        return output
