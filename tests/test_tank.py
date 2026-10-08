import math
import unittest
from threading import Event

import numpy as np

from app.background_tasks import CalculationCancelled
from app.tank import DEFAULTS, advance, compare_regimes, equilibrium, simulate, validate_inputs
from app.pi_control import pi_update


class TankTests(unittest.TestCase):
    def test_extreme_flows_and_fractional_prediction_delay(self):
        run = simulate(dict(DEFAULTS,withdrawal=1e8,pump_step=0.,horizon=1.,sample_time=.1,step_time=.5))
        self.assertGreater(run["time"][-1],0.)
        self.assertAlmostEqual(run["time"][-1],1e-8,delta=1e-15)
        h, elapsed, event = advance(1.,.03,0.,1.,dict(DEFAULTS,coefficient=1e-8))
        self.assertAlmostEqual(h,1.02999999,places=9)
        for delay in (.5,.001):
            model=dict(gain=200.,time_constant=200.,delay=delay)
            run=simulate(dict(DEFAULTS,horizon=21.,sample_time=1.),local_model=model)
            expected=1+.2*(1-math.exp(-(1-delay)/200))
            self.assertAlmostEqual(run["prediction"][-1],expected,places=12)

    def test_equilibrium_and_independent_ode_volume_balance(self):
        p = dict(DEFAULTS, horizon=120., sample_time=.5, pump_step=0.)
        r = simulate(p)
        np.testing.assert_allclose(r["level"], 1., atol=1e-12)
        self.assertEqual(equilibrium(p), 1.)
        for change in (.003, -.003):
            p["pump_step"] = change
            r = simulate(p)
            time = np.linspace(20,120,20001)
            h = [1.]
            def derivative(value):
                return (p["pump"]+change-p["coefficient"]*math.sqrt(value))/p["area"]
            for dt in np.diff(time):
                a = derivative(h[-1]); b = derivative(h[-1]+dt*a/2)
                c = derivative(h[-1]+dt*b/2); d = derivative(h[-1]+dt*c)
                h.append(h[-1]+dt*(a+2*b+2*c+d)/6)
            self.assertAlmostEqual(r["level"][-1], h[-1], places=9)
            net = np.trapezoid(p["pump"]+change-p["coefficient"]*np.sqrt(h), time)
            self.assertAlmostEqual(p["area"]*(r["level"][-1]-1), net, places=8)
        self.assertLess(simulate(dict(p, pump_step=0., withdrawal_step=.003))["level"][-1], 1.)

    def test_boundaries_include_exact_tangential_dry_time(self):
        p = dict(DEFAULTS, horizon=500., pump=0., pump_step=0., initial_level=1., step_time=10., sample_time=3.7)
        r = simulate(p)
        self.assertEqual(r["event"]["kind"], "empty")
        self.assertAlmostEqual(r["time"][-1], 200., places=9)
        self.assertAlmostEqual(r["level"][-1], 0., places=12)
        p = dict(DEFAULTS, horizon=500., pump=.04, pump_step=0., initial_level=2.9)
        r = simulate(p)
        self.assertEqual(r["event"]["kind"], "full")
        nodes, weights = np.polynomial.legendre.leggauss(64)
        reference = .05*np.dot(weights,p["area"]/(p["pump_limit"]-p["coefficient"]*np.sqrt(2.95+.05*nodes)))
        self.assertAlmostEqual(r["time"][-1], reference, places=9)
        self.assertEqual(r["level"][-1], 3.)
        self.assertEqual(max(r["pump"]), .03)
        self.assertEqual(max(r["commanded"]), .04)
        r = simulate(dict(DEFAULTS, pump=.001, withdrawal=.02, pump_step=0., horizon=1000.))
        self.assertEqual(r["event"]["kind"], "empty")
        self.assertGreaterEqual(min(r["level"]), 0.)

    def test_sampled_pi_converges_and_checks_unreachable_and_stops(self):
        controller = dict(gain=.015, integral_time=200., setpoint=1.4)
        p = dict(DEFAULTS, horizon=1200., withdrawal_step=.001)
        r = simulate(p, controller=controller)
        self.assertLess(abs(r["level"][-1]-1.4), .01)
        finer = simulate(dict(p,sample_time=.1), controller=controller)
        self.assertLess(abs(r["metrics"]["iae"]-finer["metrics"]["iae"]), .2)
        self.assertTrue(all(0 <= q <= .03 for q in r["pump"]))
        unreachable = simulate(dict(p,pump_limit=.0101, withdrawal_step=0.), controller=controller)
        self.assertEqual(unreachable["metrics"]["status"], "unreachable")
        self.assertIsNone(unreachable["metrics"]["settling_time"])
        stopped = simulate(dict(p,withdrawal_step=.1), controller=controller)
        self.assertEqual(stopped["metrics"]["status"], "physical_stop")
        self.assertIsNone(stopped["metrics"]["settling_time"])

    def test_pi_antiwindup_and_fixed_settings_across_regimes(self):
        actual, integral = pi_update(10., 1., 1., 1., .01,0.,.03,0.)
        self.assertEqual((actual,integral),(.03,0.))
        actual, integral = pi_update(-.001,1.,1.,1.,.01,0.,.03,integral)
        self.assertGreater(actual,0.)
        self.assertLess(actual,.01)
        fit = dict(gain=200., time_constant=200., delay=0.)
        r = compare_regimes(DEFAULTS,fit,.015,200.,1.2)
        for prefix in ("Исходные", "Предложенные"):
            cases = [c for c in r["cases"] if c["name"].startswith(prefix)]
            self.assertEqual(len({(c["controller"]["gain"],c["controller"]["integral_time"]) for c in cases}),1)
            self.assertGreater(len({c["baseline_pump"] for c in cases}),1)

    def test_invalid_inputs_and_cancellation(self):
        for key,value in (("area",0.),("coefficient",float("nan")),("sample_time",1e-10),("withdrawal_step",-.1),("initial_level",3.)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_inputs(dict(DEFAULTS,**{key:value}))
        cancellation=Event(); cancellation.set()
        with self.assertRaises(CalculationCancelled):
            simulate(DEFAULTS,cancel=cancellation)


if __name__ == "__main__":
    unittest.main()
