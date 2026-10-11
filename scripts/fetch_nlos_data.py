"""Fetch the published KLT feature table and its upstream license, without an account."""
import json

from fetch_public_data import ROOT, download


if __name__ == '__main__':
    sources = json.loads((ROOT / 'config/nlos_sources.json').read_text(encoding='utf-8'))
    manifest = [download(item) for item in sources]
    (ROOT / 'data/raw/klt_features/manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
