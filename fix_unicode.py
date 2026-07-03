"""Fix Unicode characters in Python source files for Windows cp1252 compatibility."""
import os
import re

SRC_DIR = r'C:\Users\sanmu\OneDrive\Documents\1 dae\src'
RUN_FILE = r'C:\Users\sanmu\OneDrive\Documents\1 dae\run_phase3.py'

replacements = {
    '\u2713': '[OK]',
    '\u2714': '[OK]',
    '\u2715': '[FAIL]',
    '\u2717': '[FAIL]',
    '\u26a0': '[WARN]',
    '\u2705': '[DONE]',
    '\u274c': '[ERROR]',
    '\U0001f3c6': '[BEST]',
    '\u2b50': '*',
    '\u2192': '->',
    '\u2714': '[OK]',
}

def fix_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    new = content
    for orig, repl in replacements.items():
        new = new.replace(orig, repl)
    if new != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new)
        print(f'  Fixed: {os.path.basename(path)}')
    else:
        print(f'  OK (no changes): {os.path.basename(path)}')

# Fix all .py files in src/
for root, dirs, files in os.walk(SRC_DIR):
    for fname in files:
        if fname.endswith('.py'):
            fix_file(os.path.join(root, fname))

fix_file(RUN_FILE)
print('\nUnicode fix complete.')
