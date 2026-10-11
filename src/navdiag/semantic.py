"""Optional, strictly local Chinese passage embeddings with no remote inference."""
import hashlib
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_ID = 'BAAI/bge-small-zh-v1.5'
MODEL_REVISION = '7999e1d3359715c523056ef9478215996d62a620'
QUERY_INSTRUCTION = '为这个句子生成表示以用于检索相关文章：'
_lock = threading.RLock()


def model_directory():
    return Path(os.environ.get('NAVDIAG_EMBEDDING_MODEL_DIR', '.cache/models/bge-small-zh-v1.5')).resolve()


def available():
    directory = model_directory()
    return all((directory / name).is_file() for name in ('embedding_manifest.json', 'model.safetensors', 'config.json'))


def _identity(directory):
    manifest = json.loads((directory / 'embedding_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('model') != MODEL_ID or manifest.get('dimensions') != 512:
        raise ValueError('Unsupported local embedding manifest')
    checksum = manifest.get('weights_sha256', '')
    if len(checksum) != 64 or any(char not in '0123456789abcdef' for char in checksum):
        raise ValueError('Invalid embedding checksum')
    stat = (directory / 'model.safetensors').stat()
    return f"{manifest['revision']}:{checksum}:{stat.st_size}:{stat.st_mtime_ns}"


class LocalEncoder:
    def __init__(self, directory):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(directory), local_files_only=True,
                                               trust_remote_code=False, use_safetensors=True).to('cpu').eval()

    def encode(self, texts, query=False):
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        encoded = []
        for offset in range(0, len(texts), 8):
            part = texts[offset:offset + 8]
            if query:
                part = [QUERY_INSTRUCTION + text for text in part]
            tokens = self.tokenizer(part, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with self.torch.inference_mode():
                vectors = self.model(**tokens).last_hidden_state[:, 0]
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
            encoded.append(vectors.cpu().numpy().astype(np.float32))
        return np.concatenate(encoded)


@lru_cache(maxsize=1)
def _encoder(directory, identity):
    return LocalEncoder(Path(directory))


@lru_cache(maxsize=1)
def _corpus_vectors(directory, identity, signature, texts):
    encoder = _encoder(directory, identity)
    cache = Path(directory).parent.parent / 'rag-index' / (signature + '.npz')
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as stored:
                vectors = stored['embeddings']
            if (vectors.shape == (len(texts), 512) and np.isfinite(vectors).all()
                    and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
                return vectors
        except (ValueError, OSError, KeyError):
            pass
    vectors = encoder.encode(list(texts))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, embeddings=vectors)
    os.replace(temporary, cache)
    return vectors


@lru_cache(maxsize=64)
def _query_vector(directory, identity, query):
    return _encoder(directory, identity).encode([query], query=True)


def similarity(query, records):
    if not available():
        return None
    directory = model_directory()
    identity = _identity(directory)
    texts = tuple(record.get('heading', record['title']) + '\n' + record['content'] for record in records)
    signature = hashlib.sha256((identity + json.dumps(texts, ensure_ascii=False)).encode()).hexdigest()
    with _lock:
        matrix = _corpus_vectors(str(directory), identity, signature, texts)
        return (matrix @ _query_vector(str(directory), identity, query).T).ravel()
