"""
Hardware Validation - Synchronization
=====================================
Aligns experimental and simulation time grids via linear interpolation.
"""

import numpy as np

class TimeSynchronizer:
    """Resamples arrays to match target time domains."""
    
    @staticmethod
    def synchronize(t_target, t_exp, y_exp):
        """Interpolates experimental y_exp values onto the target solver time grid t_target."""
        y_synced = np.interp(t_target, t_exp, y_exp)
        return y_synced
