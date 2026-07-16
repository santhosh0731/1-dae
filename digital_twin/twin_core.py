"""
Universal Digital Twin Core Module
==================================
Handles parameter sweeps, dynamic sensitivity analysis, and optimization targets
(solving for Duty, Freq, L, C to meet target efficiencies or voltage regulations).
"""

import numpy as np

class ParameterSweeper:
    """Sweeps parameter values and computes output characteristics."""
    
    @staticmethod
    def run_sweep(solver_func, params, sweep_name, start_val, end_val, steps=10):
        """Sweeps a single parameter and runs simulation at each step."""
        sweep_vals = np.linspace(start_val, end_val, steps)
        results = []
        
        for val in sweep_vals:
            # Copy base parameter dict and override sweep variable
            run_params = params.copy()
            run_params[sweep_name] = float(val)
            
            # Execute simulation
            t, y = solver_func(run_params)
            
            # Extract key indicators
            Vout = y[:, 0]
            IL = y[:, 1] if y.shape[1] > 1 else np.zeros_like(Vout)
            
            # Compute basic outputs
            v_avg = float(np.mean(Vout))
            v_ripple = float(np.max(Vout) - np.min(Vout))
            i_avg = float(np.mean(IL))
            
            results.append({
                'value': float(val),
                'vout_avg': v_avg,
                'vout_ripple': v_ripple,
                'il_avg': i_avg
            })
            
        return sweep_vals.tolist(), results

class SensitivityAnalyzer:
    """Calculates normalized sensitivity indices of the system output."""
    
    @staticmethod
    def analyze_sensitivities(solver_func, params, target_key='Vout'):
        """
        Computes sensitivities: dVout/dTheta using central finite differences.
        Returns normalized sensitivity index: (dY/Y) / (dTheta/Theta)
        """
        sensitivities = {}
        delta_pct = 0.01 # 1% perturbation
        
        # Calculate nominal output
        t, y_nom = solver_func(params)
        nom_val = np.mean(y_nom[:, 0]) # Nominal Vout average
        
        # We perturb: Vin, D, Fs, L, C
        perturb_keys = ['Vin', 'D', 'Fs', 'L', 'C']
        
        for key in perturb_keys:
            if key not in params:
                continue
                
            orig_val = params[key]
            delta = orig_val * delta_pct
            if delta < 1e-12:
                delta = 1e-6
                
            # Perturb up
            up_params = params.copy()
            up_params[key] = orig_val + delta
            _, y_up = solver_func(up_params)
            val_up = np.mean(y_up[:, 0])
            
            # Perturb down
            down_params = params.copy()
            down_params[key] = orig_val - delta
            _, y_down = solver_func(down_params)
            val_down = np.mean(y_down[:, 0])
            
            # Gradient dY/dTheta
            grad = (val_up - val_down) / (2.0 * delta)
            
            # Normalized Sensitivity: (dY / Y) / (dTheta / Theta) = grad * (Theta / Y)
            normalized_sensitivity = grad * (orig_val / (nom_val + 1e-9))
            sensitivities[key] = float(normalized_sensitivity)
            
        return sensitivities

class DesignOptimizer:
    """AI Optimizer searching design variables to hit power targets."""
    
    @staticmethod
    def optimize_design(solver_func, loss_func, params, target_efficiency=95.0, target_vout=None):
        """
        Adjusts Duty cycle and switching frequency to reach a target efficiency 
        or output voltage while minimizing losses.
        """
        best_duty = params.get('D', 0.6)
        best_freq = params.get('Fs', 50000.0)
        min_error = Infinity = float('inf')
        
        # Grid search over Duty and Fs for best matching efficiency & Vout
        duty_range = np.linspace(0.2, 0.8, 8)
        freq_range = np.linspace(20000.0, 100000.0, 5)
        
        for d in duty_range:
            for fs in freq_range:
                test_params = params.copy()
                test_params['D'] = float(d)
                test_params['Fs'] = float(fs)
                
                t, y = solver_func(test_params)
                Vout = y[:, 0]
                IL = y[:, 1]
                
                v_avg = np.mean(Vout)
                i_avg = np.mean(IL)
                
                # Calculate losses and efficiency
                # P_in = Vin * IL_avg * D
                P_in = test_params['Vin'] * i_avg * d
                P_out = (v_avg ** 2) / test_params['Rload']
                eff = (P_out / (P_in + 1e-9)) * 100.0
                
                error = 0.0
                if target_vout is not None:
                    error += (v_avg - target_vout) ** 2
                else:
                    error += (eff - target_efficiency) ** 2
                    
                if error < min_error:
                    min_error = error
                    best_duty = float(d)
                    best_freq = float(fs)
                    
        return {
            'optimal_duty': best_duty,
            'optimal_frequency_hz': best_freq,
            'error': float(np.sqrt(min_error))
        }
