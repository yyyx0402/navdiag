"""Compare local retrieval; never equate source hits with answer correctness."""
import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path

from evaluate_retrieval import baseline_retrievers
from navdiag.knowledge import retrieve_knowledge
from navdiag.semantic import MODEL_ID, MODEL_REVISION


def evaluate(output, split=None):
    resources = files('navdiag').joinpath('resources')
    # All old questions are now regression cases: their earlier holdout has been seen.
    regression = [{**item, 'split': 'regression'} for item in json.loads(
        resources.joinpath('retrieval_cases.json').read_text(encoding='utf-8'))]
    new = json.loads(resources.joinpath('retrieval_cases_v2.json').read_text(encoding='utf-8'))
    cases = [item for item in regression + new if split is None or item['split'] == split]
    retrievers = baseline_retrievers()
    actual_methods = {}
    for mode in ('sparse', 'hybrid'):
        def search(query, mode=mode):
            result = retrieve_knowledge(query, limit=5, mode=mode)
            actual_methods[mode] = {'method': result['retrieval_method'], 'fallback_reason': result['fallback_reason']}
            return result['notes']
        retrievers[mode] = search
    results, summaries = {}, {}
    for method, search in retrievers.items():
        rows = []
        for case in cases:
            hits = search(case['question'])
            relevant = [i + 1 for i, hit in enumerate(hits) if any(
                hit['document_id'] == target['document_id'] and
                (not target.get('heading') or target['heading'] == hit['heading'])
                for target in case.get('expected', []))]
            negative = case.get('expect_no_results', False)
            rows.append({**case, 'passed': not hits if negative else bool(relevant),
                         'reciprocal_rank': 1 / relevant[0] if relevant else 0,
                         'hits': [{key: hit[key] for key in ('id', 'document_id', 'heading', 'source_url')}
                                  for hit in hits]})
        results[method] = rows
        summaries[method] = {}
        for label in sorted({case['split'] for case in cases}):
            positive = [item for item in rows if item['split'] == label and not item.get('expect_no_results')]
            negative = [item for item in rows if item['split'] == label and item.get('expect_no_results')]
            summaries[method][label] = {
                'positive_cases': len(positive),
                'hit_at_5': sum(item['passed'] for item in positive) / len(positive) if positive else None,
                'mrr_at_5': sum(item['reciprocal_rank'] for item in positive) / len(positive) if positive else None,
                'negative_cases': len(negative), 'negative_rejections': sum(item['passed'] for item in negative)}
    report = {'version': 2, 'top_k': 5, 'methods': summaries, 'actual_methods': actual_methods,
              'embedding_model': MODEL_ID, 'embedding_revision': MODEL_REVISION,
              'fixed_settings': {'semantic_weight': 0.55, 'lexical_weight': 0.45, 'min_semantic_cosine': 0.46,
                                 'sparse_min_char': 0.065, 'sparse_min_bm25_raw': 2.0,
                                 'technical_anchor_min_char': 0.035, 'technical_anchor_min_bm25_raw': 2.0},
              'corpus_sha256': hashlib.sha256(resources.joinpath('rag_documents.json').read_bytes()).hexdigest(),
              'queries_v2_sha256': hashlib.sha256(resources.joinpath('retrieval_cases_v2.json').read_bytes()).hexdigest(),
              'scope': 'Local source/heading retrieval only; no model API or remote embedding calls.',
              'limitations': ['Small project-authored queries, not representative user traffic.',
                              'Old 23 queries are regression cases, including their formerly reserved cases.',
                              'Reserved queries have now been seen; final scores are regression checks after technical-term repairs.',
                              'A source hit does not establish passage completeness or answer correctness.',
                              'Char baselines have no relevance gate; rejection and ranking are distinct.'],
              'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'methods': summaries, 'actual_methods': actual_methods}, ensure_ascii=False))
    print('Hybrid misses: ' + ', '.join(item['id'] for item in results['hybrid'] if not item['passed']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/rag_retrieval_v2_eval.json'))
    parser.add_argument('--split', choices=('development', 'regression', 'reserved'))
    args = parser.parse_args()
    evaluate(args.output, args.split)
