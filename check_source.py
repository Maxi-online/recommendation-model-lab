"""Compile tracked Python source without importing optional dependencies."""
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
paths=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
count=0
for name in paths:
    if name.endswith('.py'):
        compile((root/name).read_text(encoding='utf-8-sig'),name,'exec')
        count+=1
print(json.dumps({'compiled_python_sources':count}))
