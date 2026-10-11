"""Download pinned public GNSS/label inputs; images remain local, no ROS bags."""
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(item, allowed_folders=None):
    destination = (ROOT / item['path']).resolve()
    allowed_folders = allowed_folders or ['data/raw/klt_extended', '.cache/klt_reference']
    if not any(destination.is_relative_to(ROOT / folder) for folder in allowed_folders):
        raise ValueError('Extended source destination must stay inside its local data/cache directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != item['bytes'] or sha256(destination) != item['sha256']:
            raise ValueError(f'Existing source changed: {item["path"]}; move it aside before refetching')
        print('verified', item['path'], flush=True)
        return item
    partial = destination.with_suffix(destination.suffix + '.part')
    if item['bytes'] >= 128 * 1024 * 1024:
        offset = partial.stat().st_size if partial.exists() else 0
        block_size = 4 * 1024 * 1024
        blocks = [(a, min(item['bytes'] - 1, a + block_size - 1)) for a in range(offset, item['bytes'], block_size)]
        chunks = destination.parent / (destination.name + '.chunks')
        chunks.mkdir(exist_ok=True)

        def get_block(bounds):
            a, b = bounds
            block = chunks / f'{a}-{b}.bin'
            if block.exists() and block.stat().st_size == b - a + 1:
                return block
            for attempt in range(8):
                try:
                    with urlopen(Request(item['url'], headers={'Range': f'bytes={a}-{b}', 'User-Agent': 'NavDiag'}), timeout=45) as response:
                        if response.status != 206 or response.getheader('Content-Range') != f'bytes {a}-{b}/{item["bytes"]}':
                            raise ValueError('Public server did not honor the requested byte range')
                        data = response.read()
                    if len(data) != b - a + 1:
                        raise ValueError('Incomplete public data block')
                    block.write_bytes(data)
                    return block
                except (OSError, ValueError, IncompleteRead):
                    if attempt == 7:
                        # Some public servers repeatedly truncate a large range.
                        # Recover in small verified ranges; the full SHA is still mandatory.
                        pieces = []
                        for start in range(a, b + 1, 65536):
                            end = min(b, start + 65535)
                            for retry in range(8):
                                try:
                                    with urlopen(Request(item['url'], headers={'Range': f'bytes={start}-{end}'}), timeout=45) as response:
                                        if response.status != 206 or response.getheader('Content-Range') != f'bytes {start}-{end}/{item["bytes"]}':
                                            raise ValueError('Invalid recovery range')
                                        piece = response.read()
                                    if len(piece) != end - start + 1:
                                        raise ValueError('Incomplete recovery range')
                                    pieces.append(piece)
                                    break
                                except (OSError, ValueError, IncompleteRead):
                                    if retry == 7:
                                        raise
                                    time.sleep(1)
                        block.write_bytes(b''.join(pieces))
                        return block
                    time.sleep(1)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, block in enumerate(pool.map(get_block, blocks)):
                with partial.open('ab') as target, block.open('rb') as source:
                    shutil.copyfileobj(source, target)
                if index % 32 == 0:
                    print(item['path'], partial.stat().st_size, '/', item['bytes'], flush=True)
    else:
        for attempt in range(8):
            try:
                with urlopen(Request(item['url'], headers={'User-Agent': 'NavDiag'}), timeout=45) as response, partial.open('wb') as target:
                    shutil.copyfileobj(response, target)
                break
            except (OSError, IncompleteRead):
                if attempt == 7:
                    raise
                time.sleep(1)
    if partial.stat().st_size != item['bytes'] or sha256(partial) != item['sha256']:
        raise ValueError(f'Downloaded source hash differs: {item["path"]}')
    partial.replace(destination)
    print('downloaded', item['path'], flush=True)
    return item


if __name__ == '__main__':
    sources = json.loads((ROOT / 'config/klt_extended_sources.json').read_text(encoding='utf-8'))
    for item in sources['files']:
        download(item)
    raw = ROOT / 'data/raw/klt_extended'
    for name in ['manifest', 'nav_manifest']:
        (raw / f'{name}.json').write_text(json.dumps(sources[name], indent=2), encoding='utf-8')
    (raw / 'config_commit.txt').write_text(sources['config_commit'], encoding='utf-8')
