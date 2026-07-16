"""
Report Generator - LaTeX Exporter
==================================
Compiles a publication-ready LaTeX research document summarizing simulation runs.
"""

class LaTeXReport:
    """Generates LaTeX report content."""
    
    @staticmethod
    def generate(data, params) -> str:
        tex = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\title{Scientific Machine Learning Power Electronics Platform: Simulation Report}
\author{Digital Twin Platform Engine}
\date{\today}
\begin{document}
\maketitle

\section{Executive Summary}
This document summarizes the simulation, multi-physics losses, and harmonic power quality analysis. 

\section{System Operating Parameters}
\begin{table}[ht]
\centering
\caption{Operational Parameters}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
"""
        for k, v in params.items():
            if k != 'timestamp':
                tex += f"{k} & {v} \\\\\n"
                
        tex += r"""\bottomrule
\end{tabular}
\end{table}

\section{Loss and Performance Analysis}
\begin{itemize}
"""
        tex += f"\\item Average Efficiency: {data['power']['efficiency']:.2f}\\%\n"
        tex += f"\\item Total Loss: {data['losses']['total_loss']:.2f} W\n"
        tex += f"\\item Junction Temperature: {data['temperatures']['max_temp']:.1f} $^\\circ$C\n"
        
        tex += r"""\end{itemize}

\section{Harmonic Quality Compliance (IEEE 519)}
\begin{itemize}
"""
        tex += f"\\item Total Harmonic Distortion (THD): {data['harmonics']['vout_thd']:.2f}\\%\n"
        tex += f"\\item IEEE 519 Compliance Status: {data['harmonics']['ieee_compliant']}\n"
        
        tex += r"""\end{itemize}

\end{document}
"""
        return tex
