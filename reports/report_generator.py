"""
Report Generator Manager
========================
Coordinates compilation and output file writing for HTML, MD, PDF, JSON, CSV, and LaTeX reports.
"""

import json
import csv
from pathlib import Path

from reports.latex_report import LaTeXReport
from reports.html_report import HTMLReport
from reports.markdown_report import MarkdownReport
from reports.pdf_report import PDFReport

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class ReportGenerator:
    """Consolidated multi-format writer."""
    
    @staticmethod
    def generate_all_reports(sim_data, params):
        timestamp = params.get('timestamp', 'latest')
        
        # 1. JSON Export
        json_path = REPORTS_DIR / f"report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'params': params,
                'summary': sim_data.get('summary', {}),
                'metrics': sim_data.get('metrics', {}),
                'harmonics': sim_data.get('harmonics', {}),
                'losses': sim_data.get('losses', {}),
                'temperatures': sim_data.get('temperatures', {}),
                'solver': sim_data.get('solver', {})
            }, f, indent=2)
            
        # 2. CSV Export
        csv_path = REPORTS_DIR / f"report_{timestamp}.csv"
        t = sim_data.get('t', [])
        Vout = sim_data.get('Vout', [])
        IL = sim_data.get('IL', [])
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'Vout (V)', 'IL (A)'])
            for row in zip(t, Vout, IL):
                writer.writerow(row)
                
        # 3. HTML Export
        html_content = HTMLReport.generate(sim_data, params)
        html_path = REPORTS_DIR / f"report_{timestamp}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        # 4. Markdown Export
        md_content = MarkdownReport.generate(sim_data, params)
        md_path = REPORTS_DIR / f"report_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        # 5. LaTeX Export
        latex_content = LaTeXReport.generate(sim_data, params)
        latex_path = REPORTS_DIR / f"report_{timestamp}.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
            
        # 6. PDF Export (represented as plain text PDF equivalent summary)
        pdf_content = PDFReport.generate(sim_data, params)
        pdf_path = REPORTS_DIR / f"report_{timestamp}.pdf"
        with open(pdf_path, 'w', encoding='utf-8') as f:
            f.write(pdf_content)
            
        # Copy to default "latest" suffix files for easier frontend indexing
        for ext in ['json', 'csv', 'html', 'md', 'tex', 'pdf']:
            source = REPORTS_DIR / f"report_{timestamp}.{ext}"
            dest = REPORTS_DIR / f"report_latest.{ext}"
            try:
                dest.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
            except Exception as e:
                print(f"Error copying {ext} report: {e}")
                
        return {
            'json': f"report_{timestamp}.json",
            'csv': f"report_{timestamp}.csv",
            'html': f"report_{timestamp}.html",
            'md': f"report_{timestamp}.md",
            'tex': f"report_{timestamp}.tex",
            'pdf': f"report_{timestamp}.pdf"
        }
