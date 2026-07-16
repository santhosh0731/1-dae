"""
Multi-Objective Optimization - Particle Swarm Optimization (PSO)
=================================================================
Optimizes Duty cycle, switching frequency, L, and C to maximize efficiency,
minimize ripple, and control junction temperatures.
"""

import numpy as np

class PSOOptimizer:
    """Particle Swarm Optimizer for multi-parameter power converter design."""
    
    @staticmethod
    def optimize(solver_func, params, w_eff=0.5, w_ripple=0.3, w_temp=0.2, num_particles=15, iterations=8):
        """
        Decision Variables:
        D: [0.2, 0.8]
        Fs: [20kHz, 100kHz]
        L: [20uH, 200uH]
        C: [20uF, 200uF]
        """
        # Bounds: D, Fs, L, C
        bounds = np.array([
            [0.2, 0.8],
            [20000.0, 100000.0],
            [20e-6, 200e-6],
            [20e-6, 200e-6]
        ])
        
        # Initialize particles
        pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_particles, 4))
        vel = np.zeros_like(pos)
        
        pbest_pos = pos.copy()
        pbest_score = np.full(num_particles, float('inf'))
        
        gbest_pos = pos[0].copy()
        gbest_score = float('inf')
        
        def evaluate_fitness(x):
            # Map search vector x to parameters
            test_params = params.copy()
            test_params['D'] = float(x[0])
            test_params['Fs'] = float(x[1])
            test_params['L'] = float(x[2])
            test_params['C'] = float(x[3])
            
            try:
                t, y = solver_func(test_params)
                Vout = y[:, 0]
                IL = y[:, 1] if y.shape[1] > 1 else y[:, 0]
                
                v_avg = np.mean(Vout)
                i_avg = np.mean(IL)
                v_ripple = np.max(Vout) - np.min(Vout)
                
                # Efficiency estimation
                P_in = test_params['Vin'] * i_avg * x[0]
                P_out = (v_avg ** 2) / test_params['Rload']
                eff = (P_out / (P_in + 1e-9)) * 100.0
                
                # Temperature rise estimate
                losses = (P_in - P_out)
                t_j = test_params.get('T_ambient', 25.0) + losses * 45.0
                
                # Objectives to minimize
                obj_eff = (100.0 - eff) ** 2
                obj_ripple = (v_ripple / (v_avg + 1e-9)) ** 2
                obj_temp = (max(0.0, t_j - 80.0)) ** 2
                
                score = w_eff * obj_eff + w_ripple * obj_ripple * 50.0 + w_temp * obj_temp * 0.1
                return score, eff, v_ripple, t_j
            except:
                return float('inf'), 0.0, 999.0, 999.0

        for _ in range(iterations):
            for i in range(num_particles):
                score, eff, rip, tj = evaluate_fitness(pos[i])
                
                if score < pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i] = pos[i].copy()
                    
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = pos[i].copy()
            
            # Particle updates
            r1, r2 = np.random.rand(), np.random.rand()
            vel = 0.5 * vel + 1.5 * r1 * (pbest_pos - pos) + 1.5 * r2 * (gbest_pos - pos)
            pos = pos + vel
            pos = np.clip(pos, bounds[:, 0], bounds[:, 1])
            
        # Get optimal stats
        _, eff, rip, tj = evaluate_fitness(gbest_pos)
        
        return {
            'optimal_D': float(gbest_pos[0]),
            'optimal_Fs_hz': float(gbest_pos[1]),
            'optimal_L_h': float(gbest_pos[2]),
            'optimal_C_f': float(gbest_pos[3]),
            'predicted_efficiency': float(eff),
            'predicted_ripple_v': float(rip),
            'predicted_temperature_c': float(tj),
            'fitness': float(gbest_score)
        }
