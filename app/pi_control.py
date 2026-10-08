"""A PI update in actuator units, independent of the physical object."""


def pi_update(error, gain, integral_time, dt, bias, lower, upper, integral):
    candidate = integral + gain * error * dt / integral_time
    raw = bias + gain * error + candidate
    actual = min(upper, max(lower, raw))
    if (raw - actual) * (candidate - integral) <= 0:
        integral = candidate
    else:
        actual = min(upper, max(lower, bias + gain * error + integral))
    return actual, integral
