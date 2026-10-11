"""Check publishable files and local documentation links without printing secrets."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                                text=True).splitlines()
paths = sorted(set(paths))
secret_patterns = re.compile(r'sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
problems = []
for relative in paths:
    path = ROOT / relative
    if any(part in {'.venv', '.cache', '.navdiag', '__pycache__', 'raw', 'processed'} for part in path.relative_to(ROOT).parts):
        problems.append(f'Local-only content selected: {relative}')
    if path.suffix.lower() in {'.jpg', '.png'} or not path.is_file():
        continue
    text = path.read_text(encoding='utf-8-sig')
    if secret_patterns.search(text):
        problems.append(f'Possible credential in: {relative}')
    if path.suffix == '.md':
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link or link.startswith('#'):
                continue
            target = link.split('#')[0]
            if target and not (path.parent / target).exists():
                problems.append(f'Broken local link in {relative}: {target}')
if problems:
    print('\n'.join(problems))
    raise SystemExit(1)
print(f'Release checks passed: {len(paths)} publishable files; local links exist; no matching credential patterns.')
print('This pattern check is not a comprehensive secret scanner.')
