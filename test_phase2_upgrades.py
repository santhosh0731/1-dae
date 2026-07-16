"""
Phase 2 Advanced Research Upgrades Validation Script
===================================================
Verifies the correct operation of online learning fine-tuning, diagnostics RUL,
genetic/swarm optimization search engines, and LaTeX report generator routines.
"""

import sys
import unittest
import numpy as np
import torch
from pathlib import Path

# Workspace setup
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_learning.replay_buffer import ReplayBuffer
from adaptive_learning.online_learner import OnlineLearner
from adaptive_learning.continual_learning import WeightAnchorRegularizer
from diagnostics.fault_detector import FaultDetector
from diagnostics.anomaly_detector import AnomalyDetector
from diagnostics.health_monitor import HealthMonitor
from diagnostics.rul_estimator import RULMultiplier
from optimization.pso_optimizer import PSOOptimizer
from optimization.nsga2_optimizer import NSGA2Optimizer
from optimization.bayesian_optimizer import BayesianOptimizer
from grid_renewable.grid_events import GridEvents
from grid_renewable.renewable_profiles import BESSModel
from reports.report_generator import ReportGenerator


class TestPhase2Upgrades(unittest.TestCase):

    def test_replay_buffer_and_adaptation(self):
        """Verify replay buffer storing and backpropagation on cloned networks."""
        buffer = ReplayBuffer(max_size=100)
        buffer.push(np.zeros(7), np.zeros(5))
        self.assertEqual(buffer.size(), 1)
        
        # Test online learner on a simple toy model
        toy_model = torch.nn.Linear(7, 5)
        learner = OnlineLearner(toy_model, lr=0.01)
        
        # Train step
        loss = learner.train_step(np.zeros((2, 7)), np.zeros((2, 5)))
        self.assertIsInstance(loss, float)
        
        # Regularization
        reg = WeightAnchorRegularizer(toy_model, lambda_anchor=0.1)
        penalty = reg.penalty(learner.get_model())
        self.assertGreaterEqual(penalty.item(), 0.0)

    def test_diagnostics_and_rul(self):
        """Verify health monitoring, fault classifiers, and remaining useful life (RUL)."""
        t = np.linspace(0, 0.01, 100)
        # Emulate normal waveform
        Vout_normal = np.full(100, 48.0)
        IL_normal = np.full(100, 2.5)
        
        # Nominal case
        faults = FaultDetector.classify_faults(t, Vout_normal, IL_normal, {'max_temp': 35.0}, {'Vin': 48.0})
        self.assertEqual(len(faults['faults_detected']), 0)
        self.assertEqual(faults['status'], 'Healthy')
        
        # Degraded Case (Capacitor ESR degradation)
        Vout_ripple = 48.0 + 5.0 * np.sin(2 * np.pi * 100 * t)
        faults_deg = FaultDetector.classify_faults(t, Vout_ripple, IL_normal, {'max_temp': 35.0}, {'Vin': 48.0, 'ESR': 0.02})
        self.assertIn("Capacitor ESR Degradation", faults_deg['faults_detected'][0])
        
        # RUL estimation under heat
        rul = RULMultiplier.estimate_rul({'mosfet_junction_temp': 90.0, 'inductor_core_temp': 95.0})
        self.assertLess(rul['mosfet_rul_hours'], 50000.0)
        self.assertLess(rul['capacitor_rul_hours'], 5000.0)

    def test_multi_objective_optimizers(self):
        """Verify swarm (PSO) and genetic (NSGA-II) parameter optimization logic."""
        params = {'Vin': 48.0, 'Rload': 5.0, 'T_ambient': 25.0}
        
        def mock_solver(p):
            t = np.linspace(0, 0.001, 20)
            y = np.ones((20, 2)) * 48.0
            return t, y
            
        pso_res = PSOOptimizer.optimize(mock_solver, params, iterations=2)
        self.assertIn('optimal_D', pso_res)
        self.assertGreater(pso_res['predicted_efficiency'], 0.0)
        
        nsga_res = NSGA2Optimizer.optimize(mock_solver, params, generations=2)
        self.assertIn('optimal_D', nsga_res)
        
        bayesian_res = BayesianOptimizer.optimize(mock_solver, params)
        self.assertIn('optimal_D', bayesian_res)

    def test_grid_and_renewable_modulators(self):
        """Verify grid voltage anomalies and BESS power schedules."""
        t = np.linspace(0, 0.005, 100)
        vin_sag = GridEvents.get_voltage_profile(t, Vin_nominal=48.0, event_type='sag')
        self.assertLess(np.min(vin_sag), 48.0)
        
        # BESS SOC
        bess = BESSModel()
        sched = bess.get_power_schedule(pv_power=2000.0, load_demand=1500.0)
        self.assertEqual(sched['bess_status'], 'Charging')
        self.assertGreater(sched['battery_soc'], 0.8)

    def test_report_compilers(self):
        """Verify HTML, Markdown, CSV, and LaTeX report compiles."""
        sim_data = {
            't': [0, 0.001], 'Vout': [48.0, 48.0], 'IL': [2.5, 2.5],
            'power': {'efficiency': 94.5},
            'harmonics': {'vout_thd': 1.2, 'ieee_compliant': 'Compliant', 'switching_ripple': {'1Fs': 0.15}},
            'losses': {
                'total_loss': 4.5,
                'semiconductor': {'semiconductor_total': 2.0, 'mosfet_conduction': 1.0, 'mosfet_switching': 1.0, 'diode_conduction': 0.0, 'diode_switching': 0.0},
                'magnetic': {'inductor_total': 1.5, 'core_loss': 0.5, 'copper_loss': 1.0},
                'capacitor': {'capacitor_loss': 0.5},
                'dead_time': {'dead_time_loss': 0.5}
            },
            'temperatures': {'max_temp': 45.0, 'mosfet_junction_temp': 40.0, 'inductor_core_temp': 42.0},
            'solver': {'method': 'Radau'}
        }
        params = {'Vin': 48.0, 'D': 0.6, 'Fs': 50000.0, 'L': 50e-6, 'C': 47e-6, 'Rload': 5.0}
        
        files = ReportGenerator.generate_all_reports(sim_data, params)
        self.assertTrue(Path("results/reports/" + files['tex']).exists())
        self.assertTrue(Path("results/reports/" + files['pdf']).exists())


if __name__ == '__main__':
    unittest.main()
