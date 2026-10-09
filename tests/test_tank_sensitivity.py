import copy
import unittest
from threading import Event

import numpy as np

from app.background_tasks import CalculationCancelled
from app.tank import DEFAULTS, simulate
from app.tank_sensitivity import FIELD_DEFAULTS, assess, calculate, parse_fields, validate_result


class TankSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(DEFAULTS, horizon=80., sample_time=1.)
        self.local = dict(gain=200., time_constant=200., delay=0.)
        self.controller = dict(gain=.015, integral_time=200., setpoint=1.4)
        self.grid, self.requirements = parse_fields(dict(FIELD_DEFAULTS, k_errors="-20;0;20", t_errors="0", l_errors="0;5"))

    def result(self):
        return calculate(self.p, self.local, self.controller, self.grid, self.requirements)

    def test_errors_retune_pi_without_changing_physics_or_nominal_controls(self):
        r = self.result()
        validate_result(r)
        self.assertEqual(len(r["cases"]), 6)
        direct = simulate(self.p, controller=self.controller)
        self.assertEqual(r["controls"][0]["run"], direct)
        zero = next(c for c in r["cases"] if c["offsets"] == [0., 0., 0.])
        self.assertEqual(zero["run"], r["controls"][1]["run"])
        self.assertGreater(r["cases"][0]["run"]["controller"]["gain"], zero["run"]["controller"]["gain"])
        for case in r["controls"]+r["cases"]:
            run = case["run"]
            self.assertEqual(run, simulate(self.p, controller=run["controller"]))
            self.assertEqual(run["parameters"], direct["parameters"])
            self.assertNotIn("prediction", run)
        self.assertEqual(self.local, dict(gain=200., time_constant=200., delay=0.))

    def test_requirements_fail_for_each_metric_and_for_physical_stop(self):
        run = simulate(self.p, controller=self.controller)
        run["metrics"].update(final_error=-.01, iae=2., overshoot_percent=3., settling_time=4., saturation_duration=5., reachable=True)
        limits = dict(final_error=.01, iae=2., overshoot_percent=3., settling_time=4., saturation_duration=5.)
        self.assertTrue(assess(run, limits)["passed"])
        for key in limits:
            bad = dict(limits, **{key: 0.})
            self.assertEqual(assess(run, bad)["reasons"], [key])
        run["metrics"]["settling_time"] = None
        self.assertIn("settling_time", assess(run, limits)["reasons"])
        run["event"] = dict(kind="empty")
        run["metrics"]["reachable"] = False
        self.assertEqual(assess(run, limits)["reasons"][:2], ["physical_stop", "unreachable"])

    def test_negative_delay_and_invalid_or_large_grid_rejected_before_run(self):
        for fields in (dict(FIELD_DEFAULTS, k_errors="-100"), dict(FIELD_DEFAULTS, t_errors="0;0"),
                       dict(FIELD_DEFAULTS, l_errors="nan"), dict(FIELD_DEFAULTS, iae="-1"),
                       dict(FIELD_DEFAULTS, l_errors="0;1;2;3")):
            with self.assertRaises(ValueError):
                parse_fields(fields)
        with self.assertRaises(ValueError):
            calculate(self.p, self.local, self.controller, dict(self.grid, l=[-1.]), self.requirements)
        with self.assertRaises(ValueError):
            calculate(self.p, self.local, dict(self.controller, setpoint=1.), self.grid, self.requirements)
        with self.assertRaises(ValueError):
            calculate(dict(DEFAULTS, sample_time=.015), self.local, self.controller, self.grid, self.requirements)

    def test_snapshot_rejects_inconsistent_cases_and_assessments(self):
        good = self.result()
        for kind in ("offset", "estimate", "controller", "physics", "assessment", "missing", "huge", "boolean"):
            bad = copy.deepcopy(good)
            case = bad["cases"][0]
            if kind == "offset": case["offsets"][0] = 50.
            if kind == "estimate": case["estimate"]["gain"] = 300.
            if kind == "controller": case["run"]["controller"]["gain"] = .02
            if kind == "physics": case["run"]["parameters"]["area"] = 2.
            if kind == "assessment": case["assessment"]["passed"] = not case["assessment"]["passed"]
            if kind == "missing": bad["cases"].pop()
            if kind == "huge": bad["local_model"]["gain"] = 10**400
            if kind == "boolean": bad["local_model"]["delay"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_result(bad)

    def test_cancelled_and_progress(self):
        event = Event(); event.set()
        with self.assertRaises(CalculationCancelled):
            calculate(self.p, self.local, self.controller, self.grid, self.requirements, cancel=event)
        fractions = []
        r = calculate(self.p, self.local, self.controller, self.grid, self.requirements, progress=fractions.append)
        self.assertEqual(fractions[-1], 1.)
        self.assertEqual(len(fractions), len(r["cases"])+2)
        self.assertTrue(np.all(np.diff(fractions)>0))

    def test_downward_step_and_integer_time_cannot_hide_false_settling_metric(self):
        grid = dict(k=[0.], t=[0.], l=[0.])
        r = calculate(dict(self.p, horizon=1000.), self.local, dict(self.controller, setpoint=.8), grid, self.requirements)
        for case in r["controls"]+r["cases"]:
            run = case["run"]
            level = np.asarray(run["level"])
            self.assertAlmostEqual(run["metrics"]["overshoot_percent"], max(0., float(np.max(.8-level)))*100/.2)
            run["time"] = [int(t) for t in run["time"]]
        validate_result(r)
        r["controls"][0]["run"]["metrics"]["settling_time"] = 1.
        r["controls"][0]["assessment"] = assess(r["controls"][0]["run"], self.requirements)
        with self.assertRaises(ValueError): validate_result(r)

    def test_report_csv_keeps_all_cases_metrics_limits_and_stale_flag(self):
        import csv
        import tempfile
        from pathlib import Path
        from app.tank_sensitivity_exporting import write_report
        result = self.result(); result["exported_stale"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"errors.csv"
            write_report(path, result, 0, "csv")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
            self.assertEqual(len(rows), sum(len(c["run"]["time"]) for c in result["controls"]+result["cases"]))
            self.assertEqual(len({r["Опыт"] for r in rows}), 8)
            self.assertTrue(all(r["Снимок прежних настроек"] == "True" for r in rows))
            self.assertEqual(float(rows[0]["Порог: IAE, м·с"]), self.requirements["iae"])

    def test_incomplete_nonterminal_experiment_is_not_a_full_horizon(self):
        from app.tank import pi_metrics
        result = self.result(); run = result["controls"][0]["run"]
        for key in ("time", "level", "pump", "commanded", "withdrawal", "error"): run[key] = run[key][:40]
        run["metrics"] = pi_metrics(run["parameters"],run["time"],run["level"],run["withdrawal"],None,run["controller"]["setpoint"],0.)
        result["controls"][0]["assessment"] = assess(run,result["requirements"])
        with self.assertRaises(ValueError): validate_result(result)


if __name__ == "__main__":
    unittest.main()
