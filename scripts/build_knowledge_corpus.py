"""Build a publishable corpus from an explicit list of original project documents."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    'docs/OBSTRUCTION_LABELS.md', 'docs/MODEL_CARD.md',
    'docs/lessons/05_nlos_zhang2023.md', 'docs/lessons/06_generalization.md',
    'docs/research/zhang2023_reproduction.md', 'docs/OPEN_SOURCE_REFERENCES.md',
    'docs/experiments/nlos_domain_audit.md', 'docs/experiments/nlos_protocols.md',
    'docs/experiments/urbannav_generalization.md',
    'docs/research/positioning_methods.md',
)
MAX_CHARS = 480


def chunks(text, document_id, source):
    heading, title, buffered, fenced = '', '', [], False
    def emit():
        content = '\n'.join(line for _, line in buffered).strip()
        if not content:
            return None
        digest = hashlib.sha256((heading + '\n' + content).encode()).hexdigest()[:12]
        return {'id': f'{document_id}-{digest}', 'document_id': document_id, 'title': title,
                'heading': heading, 'keywords': [heading], 'content': content,
                'source_url': source, 'source_type': 'project_document',
                'start_line': buffered[0][0], 'end_line': buffered[-1][0]}
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith('```'):
            fenced = not fenced
            continue
        if fenced or re.match(r'^\s*!\[', line):
            continue
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            part = emit()
            if part:
                yield part
            buffered = []
            if len(match[1]) == 1:
                title = match[2]
            heading = match[2]
            continue
        if '练习' in heading or '阅读代码的顺序' in heading:
            continue
        if not line.strip() and not buffered:
            continue
        # Keep exact source line numbers even when a long table row must be split.
        fragments = [line[index:index + MAX_CHARS] for index in range(0, len(line), MAX_CHARS)] or ['']
        for fragment in fragments:
            if sum(len(value) + 1 for _, value in buffered) + len(fragment) > MAX_CHARS and buffered:
                part = emit()
                if part:
                    yield part
                overlap = buffered[-1:] if buffered[-1][1].strip() and len(buffered[-1][1]) <= 120 else []
                if sum(len(value) + 1 for _, value in overlap) + len(fragment) > MAX_CHARS:
                    overlap = []
                buffered = overlap
            buffered.append((line_number, fragment))
    while buffered and not buffered[-1][1].strip():
        buffered.pop()
    part = emit()
    if part:
        yield part


def build():
    records, sources = [], []
    for number, source in enumerate(DOCUMENTS, 1):
        text = (ROOT / source).read_text(encoding='utf-8')
        records.extend(chunks(text, f'D{number:02}', source))
        sources.append({'path': source, 'sha256': hashlib.sha256(text.encode()).hexdigest()})
    corpus = {'version': 2, 'max_chunk_characters': MAX_CHARS,
              'policy': 'Original project explanations; no raw PDF, secrets, data or external fetches.',
              'sources': sources, 'chunks': records}
    target = ROOT / 'src/navdiag/resources/rag_documents.json'
    target.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Built {len(records)} chunks from {len(sources)} allowlisted project documents.')
    return corpus


if __name__ == '__main__':
    build()
