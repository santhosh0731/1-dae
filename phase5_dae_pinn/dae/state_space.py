"""
State Space Representation
===========================
Maps indices and names of states, parameters, and control vectors.
"""

from typing import List


# Inputs: [time, Vin, D, Fs, L, C, Rload]
INPUT_DIM = 7
INPUT_NAMES = ['time', 'Vin', 'D', 'Fs', 'L', 'C', 'Rload']

# Outputs: [Vout, IL, Vc, dIL_dt, dVc_dt]
OUTPUT_DIM = 5
OUTPUT_NAMES = ['Vout', 'IL', 'Vc', 'dIL_dt', 'dVc_dt']

# State separation indexes
DIFF_STATES = [1, 2]       # [IL, Vc] (predicted outputs indices)
ALGEBRAIC_STATES = [0]    # [Vout] (predicted outputs indices)

CONTROL_INPUTS = [1, 2]    # [Vin, D] (model inputs indices)
PHYSICS_PARAMS = [4, 5, 6] # [L, C, Rload] (model inputs indices)
TIME_INDEX = 0
