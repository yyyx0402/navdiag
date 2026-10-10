"""Download commit-pinned GSDC examples, checking recorded size and SHA-256."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
FILES = json.loads((ROOT / 'config' / 'public_sources.json').read_text(encoding='utf-8'))


def download(item):
    path = ROOT / item['path']
    url, expected = item['url'], item['bytes']
    local = path.relative_to(ROOT / 'data' / 'raw')
    parts = url.split('/')
    repo, commit, remote = '/'.join(parts[3:5]), parts[5], '/'.join(parts[6:])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError(f'Existing file differs from the pinned source: {local}. Move it aside before downloading.')
    if not path.exists() or path.stat().st_size != expected:
        partial = path.with_suffix(path.suffix + '.part')
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'NavDiag'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            target = url if attempt < 3 else f'https://api.github.com/repos/{repo}/contents/{remote}?ref={commit}'
            if attempt >= 3:
                headers['Accept'] = 'application/vnd.github.raw+json'
            print('Downloading', local, 'attempt', attempt + 1, 'offset', offset, flush=True)
            try:
                with urlopen(Request(target, headers=headers), timeout=45) as response:
                    resumed = response.status == 206 and response.getheader('Content-Range', '').startswith(f'bytes {offset}-')
                    mode = 'ab' if resumed else 'wb'
                    total = offset if resumed else 0
                    with partial.open(mode) as destination:
                        while chunk := response.read(32768):
                            destination.write(chunk)
                            total += len(chunk)
                            if total % (1024 * 1024) == 0:
                                print(local, total, 'bytes received', flush=True)
                if partial.stat().st_size == expected:
                    if hashlib.sha256(partial.read_bytes()).hexdigest() != item['sha256']:
                        raise ValueError('SHA-256 differs from the pinned published file')
                    partial.replace(path)
                    break
                raise ValueError('Size differs from published file metadata')
            except (OSError, ValueError) as error:
                print(local, type(error).__name__, 'will retry', flush=True)
                time.sleep(1)
        else:
            raise RuntimeError(f'Download did not complete: {local}. Re-run to resume.')
    print(local, path.stat().st_size, flush=True)
    return {**item, 'path': path.relative_to(ROOT).as_posix()}


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(download, FILES))
    (ROOT / "data" / "raw" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
