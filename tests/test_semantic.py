import json

import numpy as np

from navdiag import semantic


def test_corrupted_cached_vectors_are_rebuilt_without_network(monkeypatch, tmp_path):
    directory = tmp_path / 'models' / 'encoder'
    directory.mkdir(parents=True)
    cache = tmp_path / 'rag-index' / 'fixture.npz'
    cache.parent.mkdir()
    np.savez_compressed(cache, embeddings=np.zeros((2, 512), dtype=np.float32))
    calls = []
    class Encoder:
        def encode(self, texts):
            calls.append(texts)
            result = np.zeros((len(texts), 512), dtype=np.float32)
            result[:, 0] = 1
            return result
    monkeypatch.setattr(semantic, '_encoder', lambda *args: Encoder())
    semantic._corpus_vectors.cache_clear()
    result = semantic._corpus_vectors(str(directory), 'identity', 'fixture', ('first', 'second'))
    assert calls == [['first', 'second']]
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1)
    semantic._corpus_vectors.cache_clear()
    semantic._corpus_vectors(str(directory), 'identity', 'fixture', ('first', 'second'))
    assert len(calls) == 1
    semantic._corpus_vectors.cache_clear()


def test_model_identity_invalidates_cache_when_weights_change(tmp_path):
    (tmp_path / 'embedding_manifest.json').write_text(json.dumps({
        'model': semantic.MODEL_ID, 'dimensions': 512, 'revision': semantic.MODEL_REVISION,
        'weights_sha256': 'a' * 64}), encoding='utf-8')
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'fixture')
    before = semantic._identity(tmp_path)
    weights.write_bytes(b'different-size-fixture')
    assert semantic._identity(tmp_path) != before
