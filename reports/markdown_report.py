"""
Report Generator - Markdown Exporter
===================================
Compiles a clean markdown document summarizing operational results.
"""

class MarkdownReport:
    """Generates Markdown report files."""
    
    @staticmethod
    def generate(data, params) -> str:
        md = f"""# SciML Power Electronics Simulation Report

* **Topology**: {params.get('converter', 'Boost').upper()}
* **Efficiency**: {data['power']['efficiency']:.2f}%
* **Voltage THD**: {data['harmonics']['vout_thd']:.2f}%

## 1. Operating Parameters
"""
        for k, v in params.items():
            if k != 'timestamp':
                md += f"* **{k}**: {v}\n"
                
        md += f"""
## 2. Multi-Physics Losses
* MOSFET Conduction Loss: {data['losses']['semiconductor']['mosfet_conduction']:.2f} W
* Inductor Core Loss: {data['losses']['magnetic']['core_loss']:.2f} W
* Inductor Copper Loss: {data['losses']['magnetic']['copper_loss']:.2f} W
* Total System Loss: {data['losses']['total_loss']:.2f} W
"""
        return md
