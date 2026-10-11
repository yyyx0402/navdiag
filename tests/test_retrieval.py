import json
from importlib.resources import files

import numpy as np
import pytest

from navdiag.agent_context import evidence_context
from navdiag import knowledge
from navdiag.knowledge import normalize_query, retrieve_knowledge, search_knowledge


def test_model_typo_is_corrected_only_for_retrieval():
    assert normalize_query('Transformeru 为什么更好') == 'transformer 为什么更好'
    assert normalize_query('HDOP 不是 HDOQ') == 'HDOP 不是 HDOQ'
    result = search_knowledge('论文中各颗卫星怎样交换信息', limit=5)
    assert any(item['heading'] == '论文网络为什么同时看时间和其他卫星' for item in result)
    assert all(item['source_url'] and item['document_id'] for item in result)


def test_paraphrase_future_observations_retrieves_causal_context():
    result = search_knowledge('模型会不会读到后面的观测', limit=5)
    assert any(item['document_id'] == 'K8' or item['heading'] == '论文网络为什么同时看时间和其他卫星' for item in result)


def test_unrelated_question_does_not_fabricate_passages():
    assert search_knowledge('苹果派怎么烤') == []


def test_chunks_are_bounded_traceable_and_from_allowlisted_docs():
    corpus = json.loads(files('navdiag').joinpath('resources/rag_documents.json').read_text(encoding='utf-8'))
    assert len({item['id'] for item in corpus['chunks']}) == len(corpus['chunks'])
    sources = {item['path'] for item in corpus['sources']}
    assert all(item['source_url'] in sources and 0 < item['start_line'] <= item['end_line'] and len(item['content']) <= 480
               for item in corpus['chunks'])
    assert not any('raw' in source or '.navdiag' in source for source in sources)
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for item in corpus['chunks']:
        lines = (root / item['source_url']).read_text(encoding='utf-8').splitlines()
        assert item['content'].splitlines()[0] in lines[item['start_line'] - 1]


def test_sparse_mode_never_initializes_optional_encoder(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Sparse retrieval must not load or download a model')
    monkeypatch.setattr(knowledge.semantic, 'similarity', forbidden)
    result = retrieve_knowledge('HDOP 是什么', mode='sparse')
    assert result['embedding_model'] is None and result['notes']


@pytest.mark.parametrize('failure', [None, ImportError, ValueError])
def test_missing_or_broken_local_encoder_has_explicit_sparse_fallback(monkeypatch, failure):
    def unavailable(*args):
        if failure:
            raise failure('fixture encoder error')
        return None
    monkeypatch.setattr(knowledge.semantic, 'similarity', unavailable)
    result = retrieve_knowledge('HDOP 是什么', mode='auto')
    assert result['notes'] == retrieve_knowledge('HDOP 是什么', mode='sparse')['notes']
    assert result['fallback_reason'] and result['embedding_model'] is None


def test_semantic_gate_rejects_even_maximum_normalized_keyword_score(monkeypatch):
    # A maximum BM25 ranking score is not a probability of relevance.
    monkeypatch.setattr(knowledge.semantic, 'similarity', lambda query, records: np.full(len(records), 0.40))
    result = retrieve_knowledge('量子色动力学怎么解释', mode='hybrid')
    assert result['notes'] == [] and result['embedding_model']


def test_exact_technical_term_keeps_lexical_channel_when_encoder_under_ranks_it(monkeypatch):
    monkeypatch.setattr(knowledge.semantic, 'similarity', lambda query, records: np.full(len(records), 0.30))
    result = retrieve_knowledge('HDOP 是什么', mode='hybrid')
    assert any(note['id'] == 'K2' for note in result['notes'])


def test_published_research_is_retrievable_with_scope_and_primary_source():
    result = search_knowledge('多频特征缺失 分支逻辑回归 开放天空预归一化', limit=5, mode='sparse')
    paper = next(note for note in result if note['id'] == 'L04')
    assert paper['source_type'] == 'research_summary'
    assert paper['doi'] == '10.3389/frobt.2023.1171255'
    assert paper['verification']['full_text_read_this_update'] is False
    assert paper['source_url'].startswith('https://www.frontiersin.org/')
    assert '不能替代' in paper['content']


def test_literature_registry_separates_publication_year_and_reading_scope():
    registry = json.loads(files('navdiag').joinpath('resources/literature.json').read_text(encoding='utf-8'))
    assert len({item['id'] for item in registry['records']}) == len(registry['records']) == 7
    for record in registry['records']:
        assert record['authors'] and record['year'] <= 2026 and record['doi']
        assert record['verification']['read_scope'] and record['verification']['checked_at']
        assert record['source_type'] == 'research_summary' and len(record['content']) <= 480


def test_semantic_paraphrase_can_retrieve_without_lexical_overlap(monkeypatch):
    def scores(query, records):
        return np.array([0.9 if item['id'] == 'K2' else 0.3 for item in records])
    monkeypatch.setattr(knowledge.semantic, 'similarity', scores)
    result = retrieve_knowledge('雅可比矩阵会放大测量误差吗', mode='hybrid')
    assert result['notes'][0]['id'] == 'K2'
    assert result['notes'][0]['retrieval_scores']['semantic_cosine'] == 0.9


def test_context_projection_preserves_scores_premise_and_limits_without_mutating_evidence():
    metrics = {'precision': 0.7, 'recall': 0.6, 'f1': 0.646, 'confusion_matrix': [[5, 2], [3, 4]]}
    models = {'transformer': {'test': metrics, 'training': {'history': [{'epoch': 1}]},
                              'validation_threshold_tradeoff': [{'threshold': i / 100} for i in range(100)]},
              'random_forest': {'test': metrics}, 'gru': {'test': metrics}}
    payload = {'available': True, 'models': models, 'selected_model': 'gru', 'splits': {},
               'label_sources': ['camera'], 'warning': 'Not cross-city evidence',
               'protocol_comparison': {'cross_region': {'models': models}}}
    evidence = [{'evidence_id': 'E1', 'tool': 'compare_nlos', 'data_kind': 'real', 'payload': payload}]
    original = json.dumps(evidence)
    question = '为什么 Transformeru 跨区域更好'
    context, stats = evidence_context(question, evidence, 'satellite_nlos')
    reduced = context['evidence'][0]['payload']
    assert context['question'] == question and 'transformer' in context['retrieval_query']
    assert reduced['models']['transformer']['test'] == metrics
    assert reduced['warning'] == payload['warning']
    assert set(reduced['models']) == {'transformer', 'random_forest'}
    assert 'validation_threshold_tradeoff' not in reduced['models']['transformer']
    assert reduced['premise_checks'][0]['recorded_comparison']['f1']['transformer_minus_rf'] == 0
    assert reduced['premise_checks'][0]['causal_explanation_available'] is False
    assert stats['sent_evidence_characters'] < stats['original_evidence_characters']
    assert json.dumps(evidence) == original
