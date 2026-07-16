"""
Report Generator - HTML Exporter
================================
Compiles a responsive, styled HTML dashboard report.
"""

class HTMLReport:
    """Generates styled HTML reports."""
    
    @staticmethod
    def generate(data, params) -> str:
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SciML Power Electronics Dashboard Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0c111d; color: #f0f3f6; margin: 40px; }}
        h1, h2 {{ color: #00f2fe; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid rgba(255,255,255,0.08); padding: 10px; text-align: left; }}
        th {{ background: rgba(0,242,254,0.05); }}
    </style>
</head>
<body>
    <h1>Scientific Machine Learning Engineering Report</h1>
    <p>Target Topology: <strong>{params.get('converter', 'Boost').upper()}</strong></p>
    
    <h2>Simulation Telemetry</h2>
    <table>
        <tr><th>Operational Parameter</th><th>Nominal Value</th></tr>
        {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in params.items() if k != 'timestamp')}
    </table>

    <h2>Performance Overview</h2>
    <table>
        <tr><td>System Efficiency</td><td><strong>{data['power']['efficiency']:.2f} %</strong></td></tr>
        <tr><td>Voltage THD</td><td><strong>{data['harmonics']['vout_thd']:.2f} %</strong></td></tr>
        <tr><td>Total System Loss</td><td><strong>{data['losses']['total_loss']:.2f} W</strong></td></tr>
        <tr><td>Maximum Junction Temp</td><td><strong>{data['temperatures']['max_temp']:.1f} &deg;C</strong></td></tr>
    </table>
</body>
</html>
"""
        return html
