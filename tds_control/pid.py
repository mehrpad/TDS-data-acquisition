





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

        The controller returns a delta voltage request, so output_limits usually
        represent the maximum allowed voltage change per control loop.
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

    def update_setpoint(self, setpoint):
        """
        Update the controller setpoint.
        """
        self.setpoint = setpoint

    def reset(self, measurement=None):
        """
        Reset the dynamic controller state between experiment phases.
        """
        self.previous_error = 0.0
        self.previous_measurement = measurement
        self.integral = 0.0
        self.derivative = 0.0
        self.output = 0.0

    def compute(self, current_temperature, dt=1.0, setpoint=None):
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

        candidate_integral = self.integral + error * dt
        candidate_integral = _clamp(candidate_integral, self.integral_limits)
        integral_term = self.ki * candidate_integral

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

        unclamped_output = proportional + integral_term + derivative_term
        output = _clamp(unclamped_output, self.output_limits)

        at_upper_limit = self.output_limits[1] is not None and output >= self.output_limits[1]
        at_lower_limit = self.output_limits[0] is not None and output <= self.output_limits[0]
        should_integrate = (
            unclamped_output == output
            or (at_upper_limit and error < 0)
            or (at_lower_limit and error > 0)
        )
        if should_integrate:
            self.integral = candidate_integral

        self.previous_error = error
        self.previous_measurement = current_temperature
        self.derivative = derivative_term
        self.output = output
        return output
