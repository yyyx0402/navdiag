"""Evaluate local retrieval separately from LLM fluency and answer factuality."""
import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from navdiag.knowledge import retrieve_knowledge


def baseline_retrievers():
    resources = files('navdiag').joinpath('resources')
    notes = json.loads(resources.joinpath('knowledge.json').read_text(encoding='utf-8'))
    notes = [{**item, 'document_id': item['id'], 'heading': item['title']} for item in notes]
    documents = json.loads(resources.joinpath('rag_documents.json').read_text(encoding='utf-8'))['chunks']
    methods = {}
    for name, records in [('notes_only_char_tfidf', notes), ('same_corpus_char_tfidf', notes + documents)]:
        texts = [record['title'] + ' ' + record['content'] + ' ' + ' '.join(record['keywords']) for record in records]
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        def search(query, records=records, vectorizer=vectorizer, matrix=matrix):
            scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
            return [records[i] for i in np.argsort(-scores, kind='stable')[:5] if scores[i] > 0]
        methods[name] = search
    return methods


def evaluate(output):
    cases = json.loads(files('navdiag').joinpath('resources/retrieval_cases.json').read_text(encoding='utf-8'))
    results = []
    method = None
    for case in cases:
        retrieval = retrieve_knowledge(case['question'], limit=5)
        method, hits = retrieval['retrieval_method'], retrieval['notes']
        relevant = [index + 1 for index, hit in enumerate(hits) if any(
            hit['document_id'] == expected['document_id'] and
            (not expected.get('heading') or expected['heading'] == hit.get('heading'))
            for expected in case.get('expected', []))]
        negative = case.get('expect_no_results', False)
        results.append({**case, 'passed': not hits if negative else bool(relevant),
                        'reciprocal_rank': 1 / relevant[0] if relevant else 0,
                        'hits': [{key: hit[key] for key in ('id', 'document_id', 'heading', 'source_url', 'relevance_score')}
                                 for hit in hits]})
    summaries = {}
    for split in ('development', 'holdout'):
        positive = [item for item in results if item['split'] == split and not item.get('expect_no_results')]
        negative = [item for item in results if item['split'] == split and item.get('expect_no_results')]
        summaries[split] = {'positive_cases': len(positive), 'hit_at_5': sum(item['passed'] for item in positive) / len(positive),
                            'mrr_at_5': sum(item['reciprocal_rank'] for item in positive) / len(positive),
                            'negative_cases': len(negative), 'negative_rejections': sum(item['passed'] for item in negative)}
    baselines = {}
    for method, search in baseline_retrievers().items():
        by_split = {}
        for split in ('development', 'holdout'):
            positive_ranks, negatives = [], []
            for case in [item for item in cases if item['split'] == split]:
                hits = search(case['question'])
                if case.get('expect_no_results'):
                    negatives.append(not hits)
                else:
                    ranks = [i + 1 for i, hit in enumerate(hits) if any(
                        hit['document_id'] == target['document_id'] and
                        (not target.get('heading') or target['heading'] == hit['heading']) for target in case['expected'])]
                    positive_ranks.append(1 / ranks[0] if ranks else 0)
            by_split[split] = {'hit_at_5': sum(rank > 0 for rank in positive_ranks) / len(positive_ranks),
                               'mrr_at_5': sum(positive_ranks) / len(positive_ranks),
                               'negative_rejections': sum(negatives), 'negative_cases': len(negatives)}
        baselines[method] = by_split
    resources = files('navdiag').joinpath('resources')
    report = {'retrieval_method': method, 'suite_status': 'legacy regression; all queries have been seen',
              'top_k': 5, 'splits': summaries, 'baselines': baselines,
              'corpus_sha256': hashlib.sha256(resources.joinpath('rag_documents.json').read_bytes()).hexdigest(),
              'queries_sha256': hashlib.sha256(resources.joinpath('retrieval_cases.json').read_bytes()).hexdigest(),
              'scope': 'Source/heading retrieval only. No external API calls.',
              'limitations': ['Small manually authored query set, not a representative user benchmark.',
                              'A source hit does not establish that every returned passage is relevant or the answer is correct.',
                              'Historical split names are preserved; their scores are now regression checks.'], 'results': results}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False))
    print('Misses: ' + ', '.join(item['id'] for item in results if not item['passed']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/rag_retrieval_eval.json'))
    evaluate(parser.parse_args().output)
