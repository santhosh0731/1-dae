"""
Phase 3 Results Viewer
======================
Displays all benchmark results, saved models, and plots clearly.
Run: python view_results.py
"""
import json
import os

BASE = r'C:\Users\sanmu\OneDrive\Documents\1 dae'

SEP = '=' * 65

def read_json(path):
    full = os.path.join(BASE, path)
    if os.path.exists(full):
        with open(full) as f:
            return json.load(f)
    return None

def get_r2(m):
    for key in ['Vout_avg_R2', 'overall_R2', 'Vout_R2', 'IL_R2']:
        if key in m and m[key] is not None:
            try: return float(m[key])
            except: pass
    return 0.0

def get_rmse(m):
    for key in ['Vout_avg_RMSE', 'overall_RMSE', 'Vout_RMSE']:
        if key in m and m[key] is not None:
            try: return float(m[key])
            except: pass
    return 0.0

print()
print(SEP)
print('   PHASE 3 - COMPLETE RESULTS SUMMARY')
print('   DAE-PINN Framework - Advanced Surrogate Modeling')
print(SEP)

# ── LEVEL 1 ──────────────────────────────────────────────────────
data = read_json('results/benchmarks/level1_benchmark.json')
if data:
    print()
    print('  LEVEL 1 - BASELINE MODELS  [Target: Vout_avg scalar]')
    print('  ' + '-' * 60)
    print(f"  {'Model':<18} {'R2':>8} {'RMSE':>10} {'Train(s)':>10} {'Infer(ms)':>12}")
    print('  ' + '-' * 60)
    for model, m in data.items():
        r2  = get_r2(m)
        rm  = get_rmse(m)
        t   = float(m.get('train_time_s', 0) or 0)
        inf = float(m.get('inference_time_ms', 0) or 0)
        tag = '  <-- BEST' if r2 == max(get_r2(v) for v in data.values()) else ''
        print(f"  {model:<18} {r2:>8.4f} {rm:>10.4f} {t:>10.2f} {inf:>12.2f}{tag}")

# ── LEVEL 2 ──────────────────────────────────────────────────────
data = read_json('results/benchmarks/level2_benchmark.json')
if data:
    print()
    print('  LEVEL 2 - DEEP LEARNING  [Target: Vout(t), IL(t) waveforms]')
    print('  ' + '-' * 60)
    print(f"  {'Model':<18} {'R2(Vout)':>10} {'R2(IL)':>8} {'Train(s)':>10}")
    print('  ' + '-' * 60)
    for model, m in data.items():
        r2v = float(m.get('Vout_R2', 0) or 0)
        r2i = float(m.get('IL_R2', 0) or 0)
        t   = float(m.get('train_time_s', 0) or 0)
        best_r2 = max((float(v.get('overall_R2', 0) or 0) for v in data.values()), default=0)
        tag = '  <-- BEST' if float(m.get('overall_R2', 0) or 0) == best_r2 else ''
        print(f"  {model:<18} {r2v:>10.4f} {r2i:>8.4f} {t:>10.1f}{tag}")

# ── LEVEL 3 ──────────────────────────────────────────────────────
data = read_json('results/benchmarks/level3_benchmark.json')
if data:
    print()
    print('  LEVEL 3 - OPERATOR LEARNING  [Target: Full trajectory]')
    print('  ' + '-' * 60)
    print(f"  {'Model':<18} {'R2(Vout)':>10} {'R2(IL)':>8} {'Train(s)':>10}")
    print('  ' + '-' * 60)
    for model, m in data.items():
        r2v = float(m.get('Vout_R2', 0) or 0)
        r2i = float(m.get('IL_R2', 0) or 0)
        t   = float(m.get('train_time_s', 0) or 0)
        print(f"  {model:<18} {r2v:>10.4f} {r2i:>8.4f} {t:>10.1f}")

# ── LEVEL 4 ──────────────────────────────────────────────────────
data = read_json('results/benchmarks/level4_benchmark.json')
if data:
    print()
    print('  LEVEL 4 - CONTINUOUS-TIME MODELS  [State dynamics IL(t), Vout(t)]')
    print('  ' + '-' * 60)
    print(f"  {'Model':<20} {'R2(IL)':>8} {'R2(Vout)':>10} {'Train(s)':>10}")
    print('  ' + '-' * 60)
    for model, m in data.items():
        if 'error' in m:
            print(f"  {model:<20}  ERROR: {m['error'][:30]}")
            continue
        r2i = float(m.get('IL_R2', 0) or 0)
        r2v = float(m.get('Vout_R2', 0) or 0)
        t   = float(m.get('train_time_s', 0) or 0)
        print(f"  {model:<20} {r2i:>8.4f} {r2v:>10.4f} {t:>10.1f}")

# ── BEST MODELS ───────────────────────────────────────────────────
print()
print(SEP)
print('  BEST MODEL SELECTION - Phase 4 Candidates')
print('  ' + '-' * 60)
best = read_json('results/benchmarks/best_models.json')
if best:
    for cat, info in best.items():
        r2 = info.get('R2', 0)
        try: r2str = f"{float(r2):.4f}"
        except: r2str = str(r2)
        print(f"  [{cat.upper():<10}]  {info['model']:<20}  R2 = {r2str}")
else:
    print("  (Best models file not found - run benchmarking first)")

# ── SAVED FILES ───────────────────────────────────────────────────
print()
print(SEP)
print('  SAVED MODELS')
print('  ' + '-' * 60)
model_dir = os.path.join(BASE, 'results', 'models')
for level_dir in sorted(os.listdir(model_dir)):
    level_path = os.path.join(model_dir, level_dir)
    if os.path.isdir(level_path):
        files = os.listdir(level_path)
        print(f"  {level_dir.upper()}:")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(level_path, f))
            print(f"    - {f:<30} ({size/1024:.0f} KB)")

print()
print(SEP)
print('  PLOTS SAVED TO: results/plots/')
print('  LOGS  SAVED TO: logs/')
print('  REPORT:         results/benchmarks/final_report.txt')
print(SEP)

# ── PLOTS SUMMARY ─────────────────────────────────────────────────
print()
print('  ALL GENERATED PLOTS')
print('  ' + '-' * 60)
plots_dir = os.path.join(BASE, 'results', 'plots')
for level_dir in sorted(os.listdir(plots_dir)):
    level_path = os.path.join(plots_dir, level_dir)
    if os.path.isdir(level_path):
        files = [f for f in os.listdir(level_path) if f.endswith('.png')]
        print(f"  {level_dir.upper()} ({len(files)} plots):")
        for f in sorted(files):
            print(f"    - {f}")

print()
print('  To open results folder:')
print('  explorer "C:\\Users\\sanmu\\OneDrive\\Documents\\1 dae\\results"')
print()
