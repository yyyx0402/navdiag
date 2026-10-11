"""Download pinned public UrbanNav/SatRef/NOAA inputs; keep datasets local."""
import gzip
import hashlib
import json

from fetch_klt_extended import ROOT, download


if __name__ == '__main__':
    sources = json.loads((ROOT / 'config/urbannav_sources.json').read_text(encoding='utf-8'))
    for item in sources['files']:
        download(item, allowed_folders=['data/raw/urbannav', '.cache/klt_reference'])
        if 'decompressed_sha256' in item:
            path = ROOT / item['path']
            raw = gzip.decompress(path.read_bytes())
            if hashlib.sha256(raw).hexdigest() != item['decompressed_sha256']:
                raise ValueError('Decompressed RINEX hash differs from the pinned input')
            path.with_suffix('').write_bytes(raw)
