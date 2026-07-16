"""
Multi-Objective Optimization - NSGA-II Genetic Solver
=====================================================
Selects optimal parameter trade-offs (Pareto Front) between efficiency, ripple, and volume.
"""

import numpy as np

class NSGA2Optimizer:
    """Emulated NSGA-II Genetic Pareto-optimal selector."""
    
    @staticmethod
    def optimize(solver_func, params, pop_size=12, generations=5):
        # Decision bounds: D [0.2 - 0.8], Fs [20k - 100k]
        population = []
        for _ in range(pop_size):
            d = np.random.uniform(0.2, 0.8)
            fs = np.random.uniform(20000.0, 100000.0)
            population.append(np.array([d, fs]))
            
        pareto_front = []
        
        # Evaluate objectives: f1 (efficiency deficit), f2 (voltage ripple)
        for gen in range(generations):
            scores = []
            for ind in population:
                test_params = params.copy()
                test_params['D'] = float(ind[0])
                test_params['Fs'] = float(ind[1])
                
                try:
                    t, y = solver_func(test_params)
                    v_avg = np.mean(y[:, 0])
                    v_ripple = np.max(y[:, 0]) - np.min(y[:, 0])
                    
                    eff = float(v_avg ** 2 / (test_params['Rload'] * test_params['Vin'] * np.mean(y[:, 1]) * ind[0] + 1e-9)) * 100.0
                    eff = np.clip(eff, 0.0, 100.0)
                    
                    f1 = 100.0 - eff # Minimize efficiency deficit
                    f2 = v_ripple     # Minimize ripple
                    scores.append((f1, f2, ind))
                except:
                    scores.append((999.0, 999.0, ind))
                    
            # Identify non-dominated solutions
            dominated = np.zeros(pop_size)
            for i in range(pop_size):
                for j in range(pop_size):
                    # check if j dominates i (better in all or equal and better in one)
                    if (scores[j][0] < scores[i][0] and scores[j][1] <= scores[i][1]) or \
                       (scores[j][0] <= scores[i][0] and scores[j][1] < scores[i][1]):
                        dominated[i] = 1
                        break
                        
            # Pareto front is the set of non-dominated individuals
            pareto_front = [scores[idx] for idx in range(pop_size) if dominated[idx] == 0]
            
            # Simple mutation/crossover to update population
            new_pop = []
            for _ in range(pop_size):
                parent = population[np.random.randint(0, pop_size)]
                # add small mutation
                child = parent + np.random.normal(0, [0.05, 5000.0])
                child[0] = np.clip(child[0], 0.2, 0.8)
                child[1] = np.clip(child[1], 20000.0, 100000.0)
                new_pop.append(child)
            population = new_pop
            
        # Select best candidate from Pareto front
        best_candidate = min(pareto_front, key=lambda x: x[0] + x[1]*10)
        
        return {
            'optimal_D': float(best_candidate[2][0]),
            'optimal_Fs_hz': float(best_candidate[2][1]),
            'efficiency': float(100.0 - best_candidate[0]),
            'ripple_v': float(best_candidate[1]),
            'pareto_candidates_count': len(pareto_front)
        }
