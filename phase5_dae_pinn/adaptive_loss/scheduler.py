"""
Phase Curriculum Scheduler
===========================
Defines the 3-phase training schedule for DAE-PINN.
"""

from typing import Dict, Optional


class PhaseCurriculumScheduler:
    """
    3-phase curriculum transitions:
      Phase A: Data-only (warm-start)
      Phase B: Gentle physics + DAE
      Phase C: Full adaptive NTK balancing
    """

    def __init__(self, config: dict):
        self.cfg = config['curriculum']
        self.ph_a = self.cfg['phase_a']
        self.ph_b = self.cfg['phase_b']
        self.ph_c = self.cfg['phase_c']

    def phase_name(self, epoch: int) -> str:
        if epoch <= self.ph_a['end_epoch']:
            return 'Phase A (data-only)'
        elif epoch <= self.ph_b['end_epoch']:
            return 'Phase B (gentle physics)'
        else:
            return 'Phase C (adaptive)'

    def get_fixed_weights(self, epoch: int) -> Dict[str, float]:
        """Return preset weights for current epoch."""
        if epoch <= self.ph_a['end_epoch']:
            return {
                'data': self.ph_a['lambda_data'],
                'kvl':  self.ph_a['lambda_kvl'],
                'kcl':  self.ph_a['lambda_kcl'],
                'dae':  self.ph_a['lambda_dae'],
                'irk':  self.ph_a['lambda_irk'],
                'bc':   self.ph_a['lambda_bc'],
                'ic':   self.ph_a['lambda_ic'],
                'pwr':  self.ph_a['lambda_pwr'],
                'nrg':  0.0,
            }
        elif epoch <= self.ph_b['end_epoch']:
            return {
                'data': self.ph_b['lambda_data'],
                'kvl':  self.ph_b['lambda_kvl'],
                'kcl':  self.ph_b['lambda_kcl'],
                'dae':  self.ph_b['lambda_dae'],
                'irk':  self.ph_b['lambda_irk'],
                'bc':   self.ph_b['lambda_bc'],
                'ic':   self.ph_b['lambda_ic'],
                'pwr':  self.ph_b['lambda_pwr'],
                'nrg':  0.1,
            }
        else:
            # Fallback fixed Phase C weights
            return {
                'data': 1.0,
                'kvl':  1.0,
                'kcl':  1.0,
                'dae':  0.5,
                'irk':  0.5,
                'bc':   0.2,
                'ic':   0.2,
                'pwr':  0.2,
                'nrg':  0.1,
            }
