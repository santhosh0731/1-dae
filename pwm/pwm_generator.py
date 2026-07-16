"""
Advanced PWM Framework Module
=============================
Independent PWM module implementing SPWM, SVPWM, Third Harmonic PWM, DPWM,
Phase Shift PWM variations, and feedback controllers (voltage/current loops).
"""

import numpy as np

class PWMGenerator:
    """Advanced Pulse Width Modulation Generator and Controller Engine."""
    
    @staticmethod
    def generate_spwm(t, ma, Fs, grid_f=50.0):
        """Sinusoidal PWM generator."""
        v_carrier = 2 * (t * Fs - np.floor(t * Fs + 0.5))
        v_mod = ma * np.sin(2 * np.pi * grid_f * t)
        return 1.0 if v_mod >= v_carrier else 0.0

    @staticmethod
    def generate_third_harmonic_pwm(t, ma, Fs, grid_f=50.0):
        """Third Harmonic Injection PWM."""
        v_carrier = 2 * (t * Fs - np.floor(t * Fs + 0.5))
        # Add 1/6th of third harmonic to modulation index
        v_mod = ma * (np.sin(2 * np.pi * grid_f * t) + (1.0 / 6.0) * np.sin(6 * np.pi * grid_f * t))
        return 1.0 if v_mod >= v_carrier else 0.0

    @staticmethod
    def generate_svpwm(t, ma, Fs, grid_f=50.0):
        """Space Vector PWM (Simplified single phase projection)."""
        v_carrier = 2 * (t * Fs - np.floor(t * Fs + 0.5))
        theta = 2 * np.pi * grid_f * t
        # Injected zero-sequence voltage to maximize DC link utilization
        v_offset = 0.5 * (np.max([np.sin(theta), np.sin(theta - 2*np.pi/3)]) + np.min([np.sin(theta), np.sin(theta - 2*np.pi/3)]))
        v_mod = ma * (np.sin(theta) - v_offset)
        return 1.0 if v_mod >= v_carrier else 0.0

    @staticmethod
    def generate_dpwm(t, ma, Fs, grid_f=50.0):
        """Discontinuous PWM (Clamps phases to DC bus to reduce switching losses)."""
        v_carrier = 2 * (t * Fs - np.floor(t * Fs + 0.5))
        theta = 2 * np.pi * grid_f * t
        v_mod = ma * np.sin(theta)
        if np.abs(v_mod) > 0.85:
            # Clamp switch state
            return 1.0 if v_mod > 0 else 0.0
        return 1.0 if v_mod >= v_carrier else 0.0

    @staticmethod
    def get_phase_shifts(mode='Single', phi=0.1, dual_phase=0.05, triple_phase=0.02):
        """Calculates phase shifts for DAB converters (Single, Dual, and Triple Phase Shifts)."""
        if mode == 'Single':
            return {'D1': phi, 'D2': 0.0, 'D3': 0.0}
        elif mode == 'Dual':
            return {'D1': phi, 'D2': dual_phase, 'D3': 0.0}
        elif mode == 'Triple':
            return {'D1': phi, 'D2': dual_phase, 'D3': triple_phase}
        return {'D1': phi, 'D2': 0.0, 'D3': 0.0}

    @staticmethod
    def calculate_current_mode_duty(IL_target, IL_measured, Vout_measured, Vin, Kp=0.5, Ki=20.0, dt=1e-5):
        """Average Current Mode control loop (returns adjusted duty cycle)."""
        # Static PI variables
        if not hasattr(PWMGenerator, '_int_err'):
            PWMGenerator._int_err = 0.0
            
        error = IL_target - IL_measured
        PWMGenerator._int_err += error * dt
        
        duty = Kp * error + Ki * PWMGenerator._int_err
        # Add feedforward voltage compensation
        duty += (Vout_measured - Vin) / (Vout_measured + 1e-9)
        return float(np.clip(duty, 0.05, 0.95))

    @staticmethod
    def get_neural_pwm_decision():
        """Model Unavailable response placeholder for neural PWM."""
        return "Model unavailable"
