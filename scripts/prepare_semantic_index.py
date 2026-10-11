"""Download a pinned public encoder; encode project passages locally."""
import argparse
import hashlib
import json
from pathlib import Path

from navdiag.semantic import MODEL_ID, MODEL_REVISION, model_directory, similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Explicitly download the public model weights')
    args = parser.parse_args()
    directory = model_directory()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False, local_dir=directory,
                          allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                                          'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'README.md'])
        checksum = hashlib.sha256((directory / 'model.safetensors').read_bytes()).hexdigest()
        (directory / 'embedding_manifest.json').write_text(json.dumps(
            {'model': MODEL_ID, 'revision': MODEL_REVISION, 'weights_sha256': checksum,
             'dimensions': 512, 'device': 'cpu', 'remote_inference': False}, indent=2), encoding='utf-8')
    from navdiag.knowledge import _index
    records = _index()[0]
    scores = similarity('香港的结果能代表其他城市吗？', records)
    if scores is None:
        raise SystemExit('No local encoder. Run with --download or configure NAVDIAG_EMBEDDING_MODEL_DIR.')
    print(json.dumps({'model': MODEL_ID, 'revision': MODEL_REVISION, 'passages': len(records),
                      'remote_inference_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
