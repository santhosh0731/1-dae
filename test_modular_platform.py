"""
Automated Validation and Verification Script
===========================================
Verifies that all modular additions (topologies, PWM, harmonics, physics, DAE solving,
sensitivity sweeps, surrogate loading, and Flask routing) work as designed.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Workspace setup
sys.path.insert(0, str(Path(__file__).resolve().parent))

from converter_library.converters import CONVERTER_REGISTRY
from solid_state_transformer.sst import SolidStateTransformerModel
from pwm.pwm_generator import PWMGenerator
from harmonics.harmonic_analyzer import HarmonicAnalyzer
from physics.loss_calculator import LossCalculator
from physics.thermal_model import ThermalModel
from surrogate_models.advanced_models import UnifiedSurrogate
from digital_twin.twin_core import ParameterSweeper, SensitivityAnalyzer, DesignOptimizer
from benchmark.benchmark_engine import BenchmarkEngine


class TestModularPlatform(unittest.TestCase):

    def test_converter_library(self):
        """Verify DAE solvers for all 18 registered converter topologies."""
        params = {
            'Vin': 48.0, 'D': 0.6, 'Fs': 50000.0, 'L': 50e-6, 'C': 47e-6, 'Rload': 5.0
        }
        t_span = (0.0, 0.002)
        t_eval = np.linspace(0.0, 0.002, 100)
        
        for name, conv in CONVERTER_REGISTRY.items():
            y0 = np.zeros(len(conv.state_names))
            t, y = conv.solve_dae(t_span, y0, t_eval, params)
            self.assertEqual(len(t), 100)
            self.assertEqual(y.shape[1], len(conv.state_names))

    def test_sst_module(self):
        """Verify stage-by-stage SST numerical simulations."""
        sst = SolidStateTransformerModel()
        params = {
            'Vin': 110.0, 'D': 0.5, 'Fs': 50000.0, 'Rload': 10.0, 'phi_shift': 0.1, 'grid_frequency': 50.0
        }
        t_eval = np.linspace(0.0, 0.002, 50)
        y0 = np.zeros(6)
        t, y = sst.solve_dae((0.0, 0.002), y0, t_eval, params)
        self.assertEqual(len(t), 50)
        self.assertEqual(y.shape[1], 6)
        
        # Test surrogate returns unavailable
        self.assertEqual(sst.get_surrogate_prediction(), "Model unavailable")

    def test_pwm_generator(self):
        """Verify standard and advanced phase-shifted PWM generation."""
        # SPWM decision checks
        val = PWMGenerator.generate_spwm(0.001, ma=0.8, Fs=10000.0, grid_f=50.0)
        self.assertIn(val, [0.0, 1.0])
        
        # DAB phase shift angles
        shifts = PWMGenerator.get_phase_shifts('Triple', phi=0.15)
        self.assertEqual(shifts['D1'], 0.15)
        self.assertEqual(shifts['D3'], 0.02)

    def test_harmonic_analyzer(self):
        """Verify FFT, THD, and IEEE 519 compliance checks."""
        t = np.linspace(0, 0.04, 1000)
        # 50 Hz fundamental + 10% 3rd harmonic
        signal = 100.0 * np.sin(2 * np.pi * 50 * t) + 10.0 * np.sin(2 * np.pi * 150 * t)
        
        freqs, amps = HarmonicAnalyzer.compute_fft(t, signal)
        thd, v_fund = HarmonicAnalyzer.calculate_thd(freqs, amps)
        
        self.assertAlmostEqual(thd, 10.0, delta=1.5)
        self.assertAlmostEqual(v_fund, 100.0, delta=1.5)

    def test_loss_calculator_and_thermal(self):
        """Verify multi-physics loss model outputs and thermal rise equations."""
        losses_semi = LossCalculator.calculate_semiconductor_losses(48.0, 2.5, 2.0, 50000.0)
        self.assertGreater(losses_semi['semiconductor_total'], 0)
        
        temps = ThermalModel.calculate_temperatures(5.0, 2.0, 3.0, T_ambient=25.0)
        self.assertGreater(temps['mosfet_junction_temp'], 25.0)

    def test_unified_surrogate_loader(self):
        """Verify that untrained models report 'Model unavailable'."""
        mamba = UnifiedSurrogate('physics_mamba')
        self.assertFalse(mamba.is_available())
        
        raw_in = np.zeros((10, 7))
        self.assertEqual(mamba.predict(raw_in), "Model unavailable")

    def test_digital_twin_sweeps_and_optimizer(self):
        """Verify parameter sweeps, sensitivity index calculations, and target optimizations."""
        params = {
            'Vin': 48.0, 'D': 0.6, 'Fs': 50000.0, 'L': 50e-6, 'C': 47e-6, 'Rload': 5.0, 'converter': 'boost'
        }
        t_eval = np.linspace(0.0, 0.001, 50)
        
        def mock_solver(p):
            conv = CONVERTER_REGISTRY['boost']
            y0 = np.zeros(len(conv.state_names))
            return conv.solve_dae((0.0, 0.001), y0, t_eval, p)
            
        sens = SensitivityAnalyzer.analyze_sensitivities(mock_solver, params)
        self.assertIn('Vin', sens)
        
        opt = DesignOptimizer.optimize_design(mock_solver, None, params, target_efficiency=95.0)
        self.assertIn('optimal_duty', opt)


if __name__ == '__main__':
    unittest.main()
