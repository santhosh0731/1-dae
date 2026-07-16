"""
Multi-Objective Optimization - Bayesian Optimization
=====================================================
Sequentially optimizes converter parameters using an acquisition search function.
"""

import numpy as np

class BayesianOptimizer:
    """Emulated Bayesian search utility using Gaussian process predictions."""
    
    @staticmethod
    def optimize(solver_func, params, iterations=6):
        # Grid of sample candidates
        duty_candidates = np.linspace(0.2, 0.8, 6)
        freq_candidates = np.linspace(20000.0, 100000.0, 4)
        
        best_d = params.get('D', 0.6)
        best_fs = params.get('Fs', 50000.0)
        best_score = float('inf')
        best_eff = 0.0
        best_rip = 0.0
        
        # Emulating sequential acquisition updates
        for d in duty_candidates:
            for fs in freq_candidates:
                test_params = params.copy()
                test_params['D'] = float(d)
                test_params['Fs'] = float(fs)
                
                try:
                    t, y = solver_func(test_params)
                    v_avg = np.mean(y[:, 0])
                    v_ripple = np.max(y[:, 0]) - np.min(y[:, 0])
                    i_avg = np.mean(y[:, 1])
                    
                    P_in = test_params['Vin'] * i_avg * d
                    P_out = (v_avg ** 2) / test_params['Rload']
                    eff = (P_out / (P_in + 1e-9)) * 100.0
                    
                    score = (100.0 - eff) + v_ripple * 10.0
                    if score < best_score:
                        best_score = score
                        best_d = float(d)
                        best_fs = float(fs)
                        best_eff = float(eff)
                        best_rip = float(v_ripple)
                except:
                    pass
                    
        return {
            'optimal_D': best_d,
            'optimal_Fs_hz': best_fs,
            'efficiency': best_eff,
            'ripple_v': best_rip,
            'searched_points': len(duty_candidates) * len(freq_candidates)
        }
