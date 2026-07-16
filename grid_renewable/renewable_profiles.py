"""
Grid & Renewable Energy - Renewable Profiles
============================================
Models renewable power sources: PV Solar panels, Lithium BESS batteries, and Supercapacitors.
"""

import numpy as np

class PVModel:
    """Models a PV solar module using standard single-diode current relations."""
    def __init__(self, Voc=22.0, Isc=8.5, Vmp=18.0, Imp=8.0):
        self.Voc = Voc
        self.Isc = Isc
        self.Vmp = Vmp
        self.Imp = Imp
        self.Pmp = Vmp * Imp

    def calculate_power(self, irradiance: float, temp_c: float) -> float:
        """Estimates PV output power in Watts given irradiance G (W/m^2) and Temp."""
        # Simple interpolation relative to standard test conditions (STC: 1000 W/m^2, 25 C)
        eff_temp = 1.0 - 0.004 * (temp_c - 25.0)
        power_scale = irradiance / 1000.0
        return float(self.Pmp * power_scale * eff_temp)

class BatteryModel:
    """Models a battery storage unit tracking state-of-charge (SOC)."""
    def __init__(self, capacity_ah=100.0, nominal_voltage=48.0, initial_soc=0.8):
        self.capacity = capacity_ah
        self.nominal_v = nominal_voltage
        self.soc = initial_soc
        self.R_int = 0.01 # Internal resistance

    def charge_discharge(self, current: float, dt: float) -> float:
        """Updates SOC and returns terminal voltage. Positive current = discharging."""
        # Update SOC: dSOC/dt = -I / Capacity
        self.soc -= (current * dt) / (self.capacity * 3600.0)
        self.soc = np.clip(self.soc, 0.0, 1.0)
        
        # Simple voltage vs SOC model
        v_oc = self.nominal_v * (0.85 + 0.20 * self.soc)
        v_terminal = v_oc - current * self.R_int
        return float(v_terminal)

class SupercapacitorModel:
    """Models a supercapacitor RC cell."""
    def __init__(self, capacitance_f=500.0, nominal_voltage=16.0, initial_charge=0.9):
        self.C = capacitance_f
        self.V_nom = nominal_voltage
        self.voltage = nominal_voltage * initial_charge

    def discharge(self, load_resistance: float, dt: float) -> float:
        """Discharges SC over time through load R."""
        self.voltage *= np.exp(-dt / (load_resistance * self.C))
        return float(self.voltage)

class BESSModel:
    """Battery Energy Storage System power management logic."""
    def __init__(self):
        self.battery = BatteryModel()

    def get_power_schedule(self, pv_power: float, load_demand: float) -> dict:
        """Determines BESS charge/discharge balance power based on net grid demand."""
        net_power = pv_power - load_demand
        
        if net_power > 0:
            # Charge battery
            charge_power = min(net_power, 5000.0) # limit charge to 5kW
            self.battery.charge_discharge(-charge_power / self.battery.nominal_v, 1.0)
            status = 'Charging'
        else:
            # Discharge battery to aid load
            discharge_power = min(abs(net_power), 5000.0)
            self.battery.charge_discharge(discharge_power / self.battery.nominal_v, 1.0)
            status = 'Discharging'
            
        return {
            'bess_status': status,
            'net_power_watts': float(net_power),
            'battery_soc': float(self.battery.soc),
            'battery_voltage_v': float(self.battery.nominal_v * (0.85 + 0.20 * self.battery.soc))
        }
