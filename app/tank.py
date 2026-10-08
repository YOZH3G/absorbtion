"""Constant-area tank: independent physics, signals in SI, terminal boundaries."""

import copy
import math
from bisect import bisect_left, bisect_right
import numpy as np

from .background_tasks import check_cancelled
from .pi_control import pi_update

MODEL_ID = "level_tank"
MODEL_VERSION = 1
DEFAULTS = dict(area=1., coefficient=.01, height=3., pump_limit=.03,
                initial_level=1., pump=.01, withdrawal=0., step_time=20.,
                pump_step=.001, withdrawal_step=0., horizon=700., sample_time=.2)
PARAMETERS = ("area", "coefficient", "height", "pump_limit")
SIGNAL_UNITS = dict(time="с", level="м", pump="м³/с", withdrawal="м³/с", commanded="м³/с")


def validate_inputs(values):
    if not isinstance(values, dict) or set(values) != set(DEFAULTS):
        raise ValueError("Бак: нужны все параметры опыта.")
    if any(isinstance(v, bool) for v in values.values()):
        raise ValueError("Бак: параметры должны быть числами.")
    try:
        result = {k: float(v) for k, v in values.items()}
    except (ValueError, TypeError) as error:
        raise ValueError("Бак: параметры должны быть числами.") from error
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Бак: параметры должны быть конечными.")
    if any(result[k] <= 0 for k in (*PARAMETERS, "horizon", "sample_time")):
        raise ValueError("A, c, Hmax, Qmax, горизонт и период должны быть положительными.")
    if not 0 < result["initial_level"] < result["height"]:
        raise ValueError("Исходный уровень должен быть между нулём и Hmax.")
    if not 0 <= result["step_time"] < result["horizon"]:
        raise ValueError("Ступень должна находиться внутри горизонта опыта.")
    if min(result["pump"], result["withdrawal"], result["pump"] + result["pump_step"],
           result["withdrawal"] + result["withdrawal_step"]) < 0:
        raise ValueError("Подача и отбор до и после ступени не могут быть отрицательными.")
    if result["horizon"] / result["sample_time"] > 50000:
        raise ValueError("Бак: допускается не больше 50000 интервалов.")
    return result


def equilibrium(values, level=None):
    p = validate_inputs(values)
    if level is not None:
        if not np.isfinite(level) or not 0 < level < p["height"]:
            raise ValueError("Рабочий уровень должен находиться внутри бака.")
        pump = p["coefficient"] * math.sqrt(level) + p["withdrawal"]
        if pump > p["pump_limit"]:
            raise ValueError("Рабочий уровень недостижим при пределе насоса.")
        return pump
    net = min(p["pump"], p["pump_limit"]) - p["withdrawal"]
    level = (net / p["coefficient"])**2 if net > 0 else 0.
    if not 0 < level < p["height"]:
        raise ValueError("Для этих расходов нет рабочего равновесия внутри бака.")
    return level


def advance(level, pump, withdrawal, duration, p):
    """Exact separation of variables for a held input; return the terminal event."""
    a, c = p["area"], p["coefficient"]
    b = pump - withdrawal
    z0, zmax = math.sqrt(level), math.sqrt(p["height"])
    speed = b - c * z0
    if abs(speed) <= 1e-14 * max(c*z0, abs(b), 1e-15):
        return level, duration, None
    if b == 0:
        empty = 2 * a * z0 / c
        elapsed = min(duration, empty)
        return (z0 - c*elapsed/(2*a))**2, elapsed, "empty" if duration >= empty else None

    def elapsed(z):
        x = -c*(z-z0)/speed
        if x <= -1:
            return math.inf
        if abs(x) < .001:
            remainder = sum((-1)**(n+1)*x**n/n for n in range(2,10))
        else:
            remainder = math.log1p(x)-x
        return 2*a/(c*c)*(-c*z0*x-b*remainder)

    target = min(zmax, b/c) if speed > 0 else max(0., b/c)
    event = "full" if speed > 0 and b/c > zmax else "empty" if b < 0 else None
    boundary_time = elapsed(target) if event is not None else math.inf
    if duration >= boundary_time:
        return p["height"] if event == "full" else 0., boundary_time, event
    low, high = min(z0, target), max(z0, target)
    for _ in range(55):
        z = (low+high)/2
        if (elapsed(z) < duration) == (speed > 0):
            low = z
        else:
            high = z
    z = (low+high)/2
    return z*z, duration, None


def simulate(values, *, controller=None, local_model=None, cancel=None, progress=None):
    p = validate_inputs(values)
    if controller is not None:
        kp, ti, setpoint = (float(controller[k]) for k in ("gain", "integral_time", "setpoint"))
        if not np.isfinite([kp, ti, setpoint]).all() or kp < 0 or ti <= 0 or not 0 < setpoint < p["height"]:
            raise ValueError("PI: нужны Kp ≥ 0, Ti > 0 и задание внутри бака.")
        bias = equilibrium(p, p["initial_level"])
        p["pump"] = bias
    else:
        bias = min(p["pump"], p["pump_limit"])
    if local_model is not None:
        k, constant, delay = (float(local_model[n]) for n in ("gain", "time_constant", "delay"))
        if not np.isfinite([k, constant, delay]).all() or k <= 0 or constant <= 0 or delay < 0:
            raise ValueError("Локальная модель бака: нужны K > 0, T > 0, L ≥ 0.")
    grid = np.unique(np.r_[np.arange(0, p["horizon"], p["sample_time"]), p["step_time"], p["horizon"]])
    time, level, pump, commanded, withdrawal = [0.], [p["initial_level"]], [], [], []
    forecast = [p["initial_level"]]
    integral, event, saturation = 0., None, 0.
    for index, end in enumerate(grid[1:]):
        check_cancelled(cancel)
        start = time[-1]
        dt = float(end-start)
        changed = start >= p["step_time"]
        d = p["withdrawal"] + (p["withdrawal_step"] if changed else 0.)
        if controller is None:
            command = p["pump"] + (p["pump_step"] if changed else 0.)
            q = min(p["pump_limit"], command)
        else:
            q, integral = pi_update(setpoint-level[-1], kp, ti, dt, bias, 0., p["pump_limit"], integral)
            command = q
        next_level, elapsed, boundary = advance(level[-1], q, d, dt, p)
        pump.append(float(q)); commanded.append(float(command)); withdrawal.append(float(d))
        time.append(start+elapsed); level.append(float(next_level))
        if q == 0 or q == p["pump_limit"]:
            saturation += elapsed
        if local_model is not None:
            # Predict the response to the actual pump signal; PI closes around physical h.
            cuts = [start]
            first = bisect_right(time, start-delay)
            last = bisect_left(time, start+elapsed-delay)
            cuts.extend(t+delay for t in time[first:last])
            cuts.append(start+elapsed)
            predicted = forecast[-1]
            for left, right in zip(cuts, cuts[1:]):
                delayed = left-delay
                delayed_q = bias if delayed < 0 else pump[min(len(pump)-1,max(0,bisect_right(time,delayed)-1))]
                target = p["initial_level"] + k*(delayed_q-bias)
                predicted = target+(predicted-target)*math.exp(-(right-left)/constant)
            forecast.append(float(predicted))
        if boundary:
            event = dict(kind=boundary, time=time[-1], level=level[-1])
            break
        if progress and index % 100 == 0:
            progress((index+1)/(len(grid)-1))
    pump.append(pump[-1]); commanded.append(commanded[-1]); withdrawal.append(withdrawal[-1])
    result = dict(model_id=MODEL_ID, model_version=MODEL_VERSION, parameters=p, units=SIGNAL_UNITS.copy(),
                  time=time, level=level, pump=pump, commanded=commanded, withdrawal=withdrawal,
                  event=event, controller=copy.deepcopy(controller), sample_time=p["sample_time"],
                  baseline_pump=bias, method="held-input exact tank balance; sampled conditional PI")
    if local_model is not None:
        result["prediction"] = forecast
        result["local_model"] = {n: float(local_model[n]) for n in ("gain", "time_constant", "delay")}
    if controller is not None:
        error = setpoint-np.asarray(level)
        span = abs(setpoint-p["initial_level"])
        outside = np.flatnonzero(abs(error) > max(.05*span, 1e-8))
        at = int(outside[-1]+1) if outside.size else 0
        local_t = 2*p["area"]*math.sqrt(p["initial_level"])/p["coefficient"]
        settled = time[at] if at < len(time) and time[-1]-time[at] >= local_t else None
        reachable = all(p["coefficient"]*math.sqrt(setpoint)+d <= p["pump_limit"] for d in withdrawal)
        if event or not reachable:
            settled = None
        result["error"] = error.tolist()
        result["metrics"] = dict(final_error=float(error[-1]), iae=float(np.trapezoid(abs(error), time)),
            overshoot_percent=float(max(0., np.max(np.sign(setpoint-p["initial_level"])*(np.asarray(level)-setpoint)))*100/span) if span > 1e-12 else 0.,
            settling_time=settled, saturation_duration=float(saturation), reachable=bool(reachable),
            status="physical_stop" if event else "unreachable" if not reachable else "settled" if settled is not None else "horizon")
    check_cancelled(cancel)
    if progress:
        progress(1.)
    return result


def compare_regimes(values, local_model, gain, integral_time, setpoint, *, cancel=None, progress=None):
    from .calculations import tune_controller_parameters
    from .physical_models import run_model
    p = validate_inputs(values)
    k, t, delay = (float(local_model[n]) for n in ("gain", "time_constant", "delay"))
    if not np.isfinite([k, t, delay]).all() or k <= 0 or t <= 0 or delay < 0:
        raise ValueError("Сначала идентифицируйте положительный канал подачи насоса.")
    tuning = tune_controller_parameters("PI", t, delay)
    settings = (("Исходные", float(gain), float(integral_time)),
                ("Предложенные", tuning["proportional_gain"]/k, tuning["integral_time"]))
    maximum_level = min(p["height"], ((p["pump_limit"]-p["withdrawal"])/p["coefficient"])**2)
    if p["pump_limit"] <= p["withdrawal"]:
        raise ValueError("Нет достижимых рабочих режимов для заданного отбора.")
    levels = (p["initial_level"], .2*maximum_level, .5*maximum_level, .8*maximum_level)
    cases = []
    for number, h in enumerate(levels):
        for name, kp, ti in settings:
            q = equilibrium(p, h)
            inputs = dict(p, initial_level=h, pump=q)
            run = run_model(MODEL_ID, dict(parameters=inputs,
                            controller=dict(gain=kp, integral_time=ti, setpoint=float(setpoint)),
                            local_model=local_model if number == 0 else None), cancel=cancel)
            run["name"] = f"{name}: {h:.4g} м"
            cases.append(run)
        if progress:
            progress((number+1)/len(levels))
    return dict(model_id=MODEL_ID, model_version=MODEL_VERSION, cases=cases,
                nominal_parameters=p, local_model={n:float(local_model[n]) for n in ("gain","time_constant","delay")}, settings=[list(s) for s in settings])
