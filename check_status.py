"""Check implementation status of all Phase 3 files."""
import sys, os
sys.path.insert(0, '.')

files = {
    'Stage 1 - Validation':     'src/pipeline/p01_data_validation.py',
    'Stage 2 - Cleaning':       'src/pipeline/p02_data_cleaning.py',
    'Stage 3 - Features':       'src/pipeline/p03_feature_engineering.py',
    'Stage 4 - Datasets':       'src/pipeline/p04_dataset_construction.py',
    'Level 1 - Baselines':      'src/models/level1_baselines/train_baselines.py',
    'Level 2 - Deep Learning':  'src/models/level2_deep_learning/train_deep_surrogates.py',
    'Level 3 - Operator':       'src/models/level3_operator/train_operator_models.py',
    'Level 4 - Neural ODE':     'src/models/level4_continuous/train_continuous_models.py',
    'Benchmarking':             'src/evaluation/benchmarking.py',
    'Master Runner':            'run_phase3.py',
}

print('=== IMPLEMENTATION STATUS ===')
all_ok = True
for name, path in files.items():
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    status = '[OK]' if exists else '[MISSING]'
    print(f'  {status}  {name:30s}  ({size:,} bytes)')
    if not exists:
        all_ok = False

print()
print('=== DATASET STATUS ===')
datasets = [
    'data/scalar_dataset/scalar_train.npz',
    'data/scalar_dataset/scalar_val.npz',
    'data/scalar_dataset/scalar_test.npz',
    'data/waveform_dataset/waveform_train.npz',
    'data/waveform_dataset/waveform_val.npz',
    'data/waveform_dataset/waveform_test.npz',
    'data/dynamic_dataset/dynamic_train.npz',
    'data/processed/params_clean.csv',
    'data/scalar_dataset/scalar_features.csv',
]
for d in datasets:
    exists = os.path.exists(d)
    size   = os.path.getsize(d) if exists else 0
    status = '[OK]' if exists else '[MISSING]'
    print(f'  {status}  {d:50s} ({size/1024:.1f} KB)')

print()
print('=== COMPLETED MODEL RESULTS ===')
benchmarks = [
    ('results/benchmarks/level1_benchmark.json', 'Level 1'),
    ('results/benchmarks/level2_benchmark.json', 'Level 2'),
    ('results/benchmarks/level3_benchmark.json', 'Level 3'),
    ('results/benchmarks/level4_benchmark.json', 'Level 4'),
]
for path, label in benchmarks:
    exists = os.path.exists(path)
    status = '[DONE]' if exists else '[PENDING]'
    print(f'  {status}  {label}  {path}')

print()
print('ALL SYSTEMS READY:', 'YES' if all_ok else 'SOME FILES MISSING')
