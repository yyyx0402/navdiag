"""Select question-specific evidence without altering recorded measurements."""
import json
import re

from navdiag.knowledge import normalize_query

MODEL_ALIASES = {
    'transformer': ('transformer',), 'random_forest': ('randomforest', 'random forest', '随机森林', 'rf'),
    'logistic_regression': ('logistic regression', '逻辑回归', 'lr'),
    'mlp': ('mlp',), 'cnn': ('cnn',), 'lstm': ('lstm',), 'gru': ('gru',),
    'cn0_rule': ('c/n0规则', 'cn0 rule', '载噪比规则', '固定规则'),
}
CLASS_METRICS = ('threshold', 'rows', 'positives', 'precision', 'recall', 'f1', 'accuracy',
                 'false_positives', 'false_negatives', 'confusion_matrix', 'total_rows', 'unknown_rows', 'label_coverage')


def _pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def _contains(query, term):
    return term in query if not term.isascii() or ' ' in term or '/' in term else bool(
        re.search(r'(?<![a-z])' + re.escape(term) + r'(?![a-z])', query))


def question_plan(question, task_context=None):
    query = normalize_query(question).lower()
    focus = {name for name, terms in MODEL_ALIASES.items() if any(_contains(query, term) for term in terms)}
    training = any(word in query for word in ('超参', '学习率', '训练设置', '训练使用', '训练用了',
                    '训练参数', '优化器', '几轮', '早停', 'hyperparameter', 'learning rate', 'optimizer'))
    performance = any(word in query for word in ('效果', '表现', '成绩', '性能', '优势', '占优', '更好', '更优', '不如', '排名',
                     '对比模型', '比较模型', '哪个模型', '模型对比', '模型比较',
                     'precision', 'recall', 'f1', '准确率', '正确率', '误报', '漏报', 'performance'))
    dataset = any(word in query for word in ('数据规模', '数据量', '多少观测', '多少历元', '数据集',
                  '数据划分', '训练集', '验证集', '测试集', '标签覆盖', 'unknown', 'dataset', 'split'))
    # Naming a model does not turn a paper-architecture/concept question into a score query.
    concept = any(word in query for word in ('完整复现', '核心差别', '跨卫星', '能否证明', '是否足以证明',
                   '能保证', '原理', '必要条件', '论文', '含义', 'what is'))
    positioning = any(word in query for word in ('定位精度', '位置误差', 'rmse', 'p95', '滤波',
                       'kalman', '断解', '可用率'))
    nlos = task_context == 'satellite_nlos' or any(word in query for word in ('nlos', '遮挡识别',
                           '同域', '跨区域', '城区互斥', '新城区'))
    intent = ('training' if training else 'positioning' if positioning and (not concept or performance)
              else 'performance' if performance else 'dataset' if dataset else 'concept')
    if concept and not performance and not training and not dataset:
        intent = 'concept'
    if intent == 'performance' and len(focus) == 1 and any(word in query for word in ('更好', '不如', '优势', 'better')):
        only = next(iter(focus))
        if only != 'cn0_rule':
            focus.add('transformer' if only == 'random_forest' else 'random_forest')
    scopes = []
    urban = any(word in query for word in ('新城区', '城区互斥', 'urbannav', '旺角', '黄埔', '尖沙咀',
                    '外部迁移', '外部测试', '源模型', '冻结', 'district'))
    klt = any(word in query for word in ('同域', '双协议', '熟悉区域', '未见区域', 'in-domain')) or ('klt' in query and not urban)
    ambiguous_region = any(word in query for word in ('跨区域', '泛化', 'generaliz')) and not klt and not urban
    if klt or ambiguous_region:
        scopes.extend(('in_domain', 'cross_region'))
    if urban:
        explicit = []
        if any(word in query for word in ('外部迁移', '外部测试', '冻结', '源模型', 'external_transfer')):
            explicit.append('external_transfer')
        if any(word in query for word in ('城区互斥', 'district_holdout')):
            explicit.append('district_holdout')
        scopes.extend(explicit or ('external_transfer', 'district_holdout'))
    elif ambiguous_region:
        scopes.extend(('external_transfer', 'district_holdout'))
    if not scopes:
        scopes = ['selected_experiment']
    return {'intent': intent, 'task': 'satellite_nlos' if nlos else task_context or 'position_error',
            'models': sorted(focus), 'scopes': scopes,
            'include_directions': any(word in query for word in ('每折', '各方向', '阈值', 'threshold')),
            'include_errors': any(word in query for word in ('误报', '漏报', '错误', '残差', '仰角', '信噪比', 'c/n0')),
            'scope_is_ambiguous': ambiguous_region}


def _model_focus(question):
    return set(question_plan(question)['models']) or None


def _scores(value):
    return _pick(value, CLASS_METRICS)


def _models(models, focus, plan=None):
    plan = plan or {'intent': 'performance', 'include_directions': False}
    result = {}
    for name, model in models.items():
        if focus and name not in focus:
            continue
        if plan['intent'] == 'training':
            training = {key: value for key, value in model.get('training', {}).items()
                        if key not in {'history', 'torch_version', 'device'}}
            if 'configuration' in training:
                config = dict(training['configuration'])
                if name != 'transformer':
                    config.pop('heads', None)
                if name != 'cnn':
                    config.pop('kernel_size', None)
                training['configuration'] = config
            result[name] = {'features': model.get('features', []), 'training': training,
                            **_pick(model, ('configuration', 'parameters'))}
            continue
        item = _pick(model, ('threshold', 'threshold_origin', 'validation_f1_mean'))
        for split in ('validation', 'test', 'test_macro'):
            if split in model:
                item[split] = _scores(model[split])
        if 'per_area' in model:
            item['per_area'] = {}
            for area, values in model['per_area'].items():
                if 'test_mean' in values:
                    area_scores = {'test_mean': _scores(values['test_mean'])}
                    if plan['include_directions']:
                        area_scores['directions'] = {key: _scores(scores) for key, scores in values['directions'].items()}
                else:
                    area_scores = _scores(values)
                item['per_area'][area] = area_scores
        result[name] = item
    return result


def _comparison(value, focus, plan):
    result = _pick(value, ('selected_model', 'selection_policy', 'source_protocol',
                          'summary_interpretation', 'rows', 'total_rows'))
    result['models'] = _models(value.get('models', {}), focus, plan)
    return result


def _premise_checks(payload):
    scopes = [('selected_experiment', payload)]
    scopes.extend(payload.get('protocol_comparison', {}).items())
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
                             'transformer_minus_rf': round(left[metric] - right[metric], 8)}
                    for metric in ('precision', 'recall', 'f1') if isinstance(left.get(metric), (int, float))
                    and isinstance(right.get(metric), (int, float))}
        areas = {area: compare(scores.get('test_mean', scores),
                              b.get('per_area', {}).get(area, {}).get('test_mean', b.get('per_area', {}).get(area, {})))
                 for area, scores in a.get('per_area', {}).items()}
        checks.append({'scope': scope, 'recorded_comparison': compare(a.get('test_macro', a.get('test', {})),
                       b.get('test_macro', b.get('test', {}))), 'per_area': areas,
                       'causal_explanation_available': False})
    return checks


def _nlos(payload, question, focus, plan=None):
    if not payload.get('available'):
        return payload
    plan = plan or question_plan(question, 'satellite_nlos')
    result = _pick(payload, ('available', 'task_type', 'task', 'data_kind', 'unit', 'features', 'reference_labels_used', 'warning', 'project_geography'))
    if 'selected_experiment' in plan['scopes'] or plan['intent'] == 'training':
        result.update(_pick(payload, ('label_sources', 'protocol', 'split_policy', 'selected_model',
                                     'selection_policy', 'warning', 'experiment')))
    if any(scope in plan['scopes'] for scope in ('in_domain', 'cross_region')):
        result.update(_pick(payload, ('comparison_limitations', 'comparison_unit',
                                     'comparison_test_sets_shared', 'fixed_rule_warning')))
    if plan['intent'] == 'training' or 'selected_experiment' in plan['scopes']:
        result['splits'] = payload.get('splits', {})
    if plan['intent'] != 'dataset':
        result['models'] = _models(payload['models'], focus, plan) if ('selected_experiment' in plan['scopes'] or plan['intent'] == 'training') else {}
    else:
        result['models'] = {}
    if plan['intent'] not in {'training', 'dataset'}:
        result['protocol_comparison'] = {name: _comparison(value, focus, plan) for name, value in
                                        payload.get('protocol_comparison', {}).items() if name in plan['scopes']}
    new = payload.get('new_district_generalization')
    if new and any(scope in plan['scopes'] for scope in ('external_transfer', 'district_holdout')):
        result['new_district_generalization'] = _pick(new, ('label_source', 'limitations'))
        for name in ('external_transfer', 'district_holdout'):
            if name not in plan['scopes']:
                continue
            if plan['intent'] == 'dataset':
                ext = new['external_transfer']
                example = next(iter(ext['models'].values()))
                result['new_district_generalization']['dataset_coverage'] = {
                    **_pick(ext, ('rows', 'total_rows')),
                    'per_area': {area: _pick(scores, ('rows', 'total_rows', 'unknown_rows', 'label_coverage', 'epochs'))
                                 for area, scores in example['per_area'].items()}}
                break
            if plan['intent'] != 'training':
                result['new_district_generalization'][name] = _comparison(new[name], focus, plan)
    if plan['include_errors']:
        result['error_analysis'] = payload.get('error_analysis', {})
    if focus and {'transformer', 'random_forest'} <= focus:
        result['premise_checks'] = _premise_checks(result)
    return result


def evidence_context(question, evidence, task_context=None, trip_id=None):
    plan = question_plan(question, task_context)
    focus = set(plan['models']) or None
    projected = []
    for item in evidence:
        payload = item['payload']
        if item['tool'] == 'compare_nlos':
            payload = _nlos(payload, question, focus, plan)
        elif item['tool'] == 'compare_models':
            payload = {**payload, 'models': _models(payload['models'], focus, plan)}
        elif item['tool'] == 'compare_positioning':
            methods = payload.get('methods', {})
            if 'wls' in methods and 'doppler_filter' in methods:
                comparisons = {}
                for metric in ('median_m', 'p95_m', 'rmse_m'):
                    baseline, doppler = methods['wls'].get(metric), methods['doppler_filter'].get(metric)
                    if isinstance(baseline, (int, float)) and isinstance(doppler, (int, float)):
                        comparisons[metric] = {'baseline': baseline, 'doppler': doppler,
                                               'doppler_minus_baseline': doppler - baseline,
                                               'doppler_improved': doppler < baseline}
                payload = {**payload, 'baseline_comparison': {'population': 'active test drive, same scored fixes',
                           'direction': 'lower error is better; availability is separate', 'metrics': comparisons}}
        elif item['tool'] == 'search_knowledge':
            notes = payload['notes']
            if plan['intent'] == 'positioning' and not any(word in question.lower() for word in
                    ('论文', '文献', '参考方法', '研究', 'paper', 'graphgnss', 'suzuki')):
                notes = [note for note in notes if note.get('source_type') != 'research_summary'][:2]
            elif plan['intent'] in {'training', 'performance'}:
                notes = notes[:3]
            payload = {**payload, 'notes': [_pick(note, ('id', 'document_id', 'title', 'heading', 'content',
                       'source_url', 'source_type', 'start_line', 'end_line', 'authors', 'year', 'verification',
                       'relevance_score', 'retrieval_scores')) for note in notes]}
        projected.append({**item, 'payload': payload})
    context = {'question': question, 'retrieval_query': normalize_query(question),
               'selected_task': task_context, 'selected_trip': trip_id, 'evidence': projected,
               'evidence_catalog': [{'evidence_id': item['evidence_id'], 'tool': item['tool'],
                    'document_ids': sorted({note.get('document_id', note['id'])
                    for note in item['payload'].get('notes', [])})} for item in projected],
               'task_definitions': ({'positive_class': 'NLOS', 'negative_class': 'LOS',
                    'unknown': 'Excluded from supervision and scoring; not a negative label.',
                    'selection_rules': {'neural_checkpoint': 'lowest validation BCE',
                                        'classification_threshold': 'validation F1', 'model_family': 'validation F1'}}
                    if task_context == 'satellite_nlos' else {}),
               'context_scope': {**plan, 'models': plan['models'] or 'all recorded models',
                    'omitted': ['full threshold scans', 'training histories', 'unrequested architectures',
                                'duplicate fold metrics', 'unrelated protocols'], 'metrics_are_recorded_values': True}}
    compact_size = len(json.dumps(projected, ensure_ascii=False, separators=(',', ':')))
    original_size = len(json.dumps(evidence, ensure_ascii=False, separators=(',', ':')))
    return context, {'original_evidence_characters': original_size, 'sent_evidence_characters': compact_size,
                     'question_normalized_for_retrieval': normalize_query(question) != question,
                     'intent': plan['intent'], 'scopes': plan['scopes']}
