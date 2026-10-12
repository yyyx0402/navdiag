# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch only the pinned GPL RINEX helper, without downloading the full KLT dataset."""
import hashlib
from urllib.request import urlopen

from prepare import EXTRACTOR_SHA, ROOT

COMMIT = 'b13aa856f723a472bd35d45171017580642054cf'
URL = f'https://raw.githubusercontent.com/fossettlab/gnss-nlos-notebook/{COMMIT}/extract_features.py'
target = ROOT / '.cache/klt_reference/extract_features.py'
if target.exists():
    content = target.read_bytes()
else:
    with urlopen(URL, timeout=45) as response:
        content = response.read(1_000_001)
    if len(content) > 1_000_000:
        raise ValueError('Unexpectedly large external helper')
if hashlib.sha256(content).hexdigest() != EXTRACTOR_SHA:
    raise ValueError('External helper differs from the reviewed pinned source; existing files are preserved')
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    target.write_bytes(content)
print('Pinned GPL helper verified in the Git-ignored local cache.')
