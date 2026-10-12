"""Project recorded evidence into a smaller API context without changing its metrics."""
import json
import re

from navdiag.knowledge import normalize_query


def _pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def _model_focus(question):
    query = normalize_query(question).lower()
    aliases = {'transformer': ('transformer',), 'random_forest': ('randomforest', 'random forest', '随机森林', 'rf'),
               'logistic_regression': ('logistic regression', '逻辑回归', 'lr'),
               'mlp': ('mlp',), 'cnn': ('cnn',), 'lstm': ('lstm',), 'gru': ('gru',),
               'cn0_rule': ('c/n0', 'cn0', '载噪比规则', '固定规则')}
    focus = {name for name, terms in aliases.items() if any(
        term in query if not term.isascii() or ' ' in term or '/' in term
        else re.search(r'(?<![a-z])' + re.escape(term) + r'(?![a-z])', query) for term in terms)}
    if len(focus) == 1 and next(iter(focus)) != 'cn0_rule':
        focus.add('transformer' if 'random_forest' in focus else 'random_forest')
    return focus or None


def _models(models, focus):
    result = {}
    for name, model in models.items():
        if focus and name not in focus:
            continue
        item = {key: value for key, value in model.items() if key not in
                {'training', 'training_history', 'validation_threshold_tradeoff', 'per_test_run'}}
        if 'training' in model:
            item['training'] = {key: value for key, value in model['training'].items()
                                if key not in {'history', 'torch_version', 'device'}}
            history = model['training'].get('history', [])
            if history:
                item['training']['history_summary'] = {'epochs_recorded': len(history),
                                                       'first': history[0], 'last': history[-1]}
        result[name] = item
    return result


def _comparison(value, focus):
    result = _pick(value, ('selected_model', 'selection_policy', 'source_protocol', 'source_splits',
                           'summary_interpretation', 'observed_test_order', 'rows', 'total_rows'))
    result['models'] = _models(value.get('models', {}), focus)
    return result


def _premise_checks(payload):
    """Surface protocol and area comparisons, without inventing their causes."""
    scopes = [('selected_experiment', payload)]
    scopes.extend((name, value) for name, value in payload.get('protocol_comparison', {}).items())
    scopes.extend((name, value) for name, value in payload.get('new_district_generalization', {}).items()
                  if isinstance(value, dict) and 'models' in value)
    checks = []
    for scope, value in scopes:
        models = value.get('models', {})
        if not all(name in models for name in ('transformer', 'random_forest')):
            continue
        a, b = models['transformer'], models['random_forest']
        def compare(left, right):
            return {metric: {'transformer': left[metric], 'random_forest': right[metric],
                             'transformer_minus_rf': left[metric] - right[metric]}
                    for metric in ('precision', 'recall', 'f1') if isinstance(left.get(metric), (int, float))
                    and isinstance(right.get(metric), (int, float))}
        metrics = compare(a.get('test_macro', a.get('test', {})), b.get('test_macro', b.get('test', {})))
        areas = {area: compare(scores, b.get('per_area', {}).get(area, {}))
                 for area, scores in a.get('per_area', {}).items() if isinstance(scores, dict)}
        checks.append({'scope': scope, 'aggregation': 'area_macro' if 'test_macro' in a else 'selected_fold_test',
                       'recorded_comparison': metrics, 'per_area': areas,
                       'causal_explanation_available': False})
    return checks


def _nlos(payload, question, focus):
    if not payload.get('available'):
        return payload
    result = _pick(payload, ('available', 'task_type', 'task', 'data_kind', 'unit', 'features',
                             'label_sources', 'splits', 'protocol', 'split_policy', 'selected_model',
                             'selection_policy', 'comparison_policy', 'warning', 'evaluation_limitations',
                             'experiment', 'reference_labels_used', 'comparison_limitations',
                             'comparison_unit', 'comparison_test_sets_shared', 'fixed_rule_warning'))
    result['models'] = _models(payload['models'], focus)
    query = normalize_query(question).lower()
    if any(word in query for word in ('同域', '跨区域', '泛化', 'in-domain', 'cross-region', 'generaliz')):
        result['protocol_comparison'] = {name: _comparison(value, focus) for name, value in
                                         payload.get('protocol_comparison', {}).items()}
    if any(word in query for word in ('新城区', '跨城区', '跨区域', '泛化', 'urbannav', 'klt', '旺角',
                                     '黄埔', '尖沙咀', 'mk', 'tst', 'wp', 'generaliz', 'district')):
        new = payload.get('new_district_generalization')
        if new:
            result['new_district_generalization'] = {
                **_pick(new, ('label_source', 'limitations')),
                **{name: _comparison(new[name], focus) for name in ('external_transfer', 'district_holdout')}}
    if any(word in query for word in ('误报', '漏报', '错误', '残差', '仰角', '信噪比', 'c/n0', 'cn0', 'error')):
        result['error_analysis'] = payload.get('error_analysis', {})
    if focus and {'transformer', 'random_forest'} <= focus:
        result['premise_checks'] = _premise_checks(result)
    return result


def evidence_context(question, evidence, task_context=None, trip_id=None):
    """Keep exact scores, reference sources and caveats; omit duplicated curves/history."""
    focus = _model_focus(question)
    projected = []
    for item in evidence:
        payload = item['payload']
        if item['tool'] == 'compare_nlos':
            payload = _nlos(payload, question, focus)
        elif item['tool'] == 'compare_models':
            payload = {**payload, 'models': _models(payload['models'], None)}
        projected.append({**item, 'payload': payload})
    original_size = len(json.dumps(evidence, ensure_ascii=False, separators=(',', ':')))
    projected_size = len(json.dumps(projected, ensure_ascii=False, separators=(',', ':')))
    context = {'question': question, 'retrieval_query': normalize_query(question),
               'selected_task': task_context, 'selected_trip': trip_id, 'evidence': projected,
               'evidence_catalog': [{'evidence_id': item['evidence_id'], 'tool': item['tool'],
                                     'document_ids': sorted({note.get('document_id', note['id'])
                                         for note in item['payload'].get('notes', [])})}
                                    for item in projected],
               'task_definitions': ({'positive_class': 'NLOS', 'negative_class': 'LOS',
                                     'unknown': 'Excluded from supervision and scoring; not a negative label.'}
                                    if task_context == 'satellite_nlos' else {}),
               'context_scope': {'models': sorted(focus) if focus else 'all recorded models',
                                 'omitted': ['full threshold scans', 'full training histories',
                                             'duplicate per-run metrics', 'unrelated experiment sections'],
                                 'metrics_are_recorded_values': True}}
    return context, {'original_evidence_characters': original_size, 'sent_evidence_characters': projected_size,
                     'question_normalized_for_retrieval': normalize_query(question) != question}
