"""Bounded checks of explicit metric claims; never a semantic truth certificate."""
import math
import re

from navdiag.agent_context import MODEL_ALIASES, _contains

METRICS = {'precision': 'precision', '精确率': 'precision', '查准率': 'precision',
           'recall': 'recall', '召回率': 'recall', 'f1': 'f1', 'accuracy': 'accuracy', '准确率': 'accuracy',
           'rmse': 'rmse_m', 'p95': 'p95_m', 'median': 'median_m', '中位误差': 'median_m'}
PATTERN = re.compile(r'(?<![a-z0-9])(' + '|'.join(METRICS) + r')\s*(?:值\s*)?(?:为|是|=|:|：)\s*([0-9]+(?:\.[0-9]+)?)(\s*%)?', re.I)


def _recorded(evidence):
    records = []
    def visit(value, path, evidence_id):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in set(METRICS.values()) and isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
                    records.append({'evidence_id': evidence_id, 'path': path + '/' + key, 'metric': key, 'value': item})
                elif key not in {'premise_checks', 'notes', 'training'}:
                    visit(item, path + '/' + key, evidence_id)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + '/' + str(index), evidence_id)
    for item in evidence:
        if item['tool'] in {'compare_nlos', 'compare_models', 'compare_positioning'}:
            visit(item['payload'], '', item['evidence_id'])
    return records


def check_answer_metrics(answer, evidence):
    records = _recorded(evidence)
    checks, scope, headers = [], None, None
    all_citations = set(re.findall(r'\bE\d+\b', answer))
    scopes = {'external_transfer': ('外部迁移', '外部测试', '冻结 klt', 'external_transfer'),
              'district_holdout': ('城区互斥', 'district_holdout'),
              'in_domain': ('同域', 'in_domain'), 'cross_region': ('跨区域', 'cross_region')}
    for line in answer.splitlines():
        line = line.replace('**', '').replace('`', '')
        lower = line.lower()
        named_scopes = [key for key, terms in scopes.items() if any(term in lower for term in terms)]
        if named_scopes:
            target_scopes = [key for key in named_scopes if key in {'external_transfer', 'district_holdout'}]
            candidates = target_scopes or named_scopes
            scope = candidates[0] if len(candidates) == 1 else None
        if not line.strip():
            headers = None
            continue
        # Author results, examples and target values are not claims of recorded project performance.
        if any(word in lower for word in ('作者', '文献', '论文', '例如', '假设', '假如', '目标值', '举例')):
            continue
        claims = [(METRICS[m[1].lower()], m[2], bool(m[3])) for m in PATTERN.finditer(line)]
        if line.strip().startswith('|'):
            cells = [cell.strip().strip('*') for cell in line.strip().strip('|').split('|')]
            columns = {i: METRICS[cell.lower()] for i, cell in enumerate(cells) if cell.lower() in METRICS}
            if columns:
                headers = columns
                continue
            if headers and not re.fullmatch(r'[\s|:\-]+', line):
                for index, metric in headers.items():
                    if index < len(cells):
                        match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(\s*%)?', cells[index])
                        if match:
                            claims.append((metric, match[1], bool(match[2])))
        models = {name for name, terms in MODEL_ALIASES.items() if any(_contains(lower, term) for term in terms)}
        citations = set(re.findall(r'\bE\d+\b', line)) or all_citations
        for metric, raw, percent in claims:
            broad = [record for record in records if record['metric'] == metric and
                     (not citations or record['evidence_id'] in citations)]
            if not broad:
                checks.append({'metric': metric, 'value': float(raw), 'status': 'not_checked',
                               'reason': 'No corresponding recorded metric in cited tools.'})
                continue
            candidates = broad
            if len(models) == 1:
                name = next(iter(models))
                candidates = [record for record in candidates if '/models/' + name + '/' in record['path']]
            if scope:
                candidates = [record for record in candidates if '/' + scope + '/' in record['path']]
            split = 'validation' if re.search(r'(?:验证(?:集)?(?:的|上)?|validation)\s*(?:precision|recall|f1|准确率|精确率|召回率)', lower) else None
            if split:
                candidates = [record for record in candidates if '/validation/' in record['path']]
            else:
                candidates = [record for record in candidates if '/validation/' not in record['path']]
            value = float(raw) / (100 if percent else 1)
            decimals = len(raw.split('.')[1]) if '.' in raw else 0
            tolerance = 0.5 * 10 ** -decimals / (100 if percent else 1)
            if metric in {'precision', 'recall', 'f1', 'accuracy'}:
                tolerance = min(tolerance, .005)
            matches = [record for record in candidates if abs(record['value'] - value) <= tolerance + 1e-10]
            checks.append({'metric': metric, 'value': value, 'status': 'matched' if matches else 'needs_review',
                           'matched_paths': [record['evidence_id'] + ':' + record['path'] for record in matches]})
    return {'scope': 'Explicit labelled values, simple metric tables and bounded comparison rules; not full semantic verification.',
            'claims': checks, 'matched': sum(row['status'] == 'matched' for row in checks),
            'needs_review': sum(row['status'] == 'needs_review' for row in checks),
            'not_checked': sum(row['status'] == 'not_checked' for row in checks),
            'comparison_warnings': check_comparisons(answer, evidence)}


def check_comparisons(answer, evidence):
    """Check explicit active-drive Doppler claims and registered NLOS checkpoint rules."""
    warnings = []
    for item in evidence:
        comparisons = item['payload'].get('baseline_comparison', {}).get('metrics', {})
        if not comparisons:
            continue
        for sentence in re.split(r'[。！？\n]', answer.lower().replace('**', '')):
            if 'doppler' not in sentence or not any(word in sentence for word in ('旺角', 'mk', '主测试', '当前测试', '主要测试')):
                continue
            for label, key in (('p95', 'p95_m'), ('rmse', 'rmse_m'), ('中位误差', 'median_m')):
                if label not in sentence or key not in comparisons:
                    continue
                tail = sentence.split(label, 1)[1][:35]
                tail = re.split(r'p95|rmse|中位误差|median', tail)[0]
                if any(word in tail for word in ('没有', '并非', '不一定', '未改善', '不是')):
                    continue
                better = any(word in tail for word in ('改善', '略优', '更好', '降低', '下降'))
                worse = any(word in tail for word in ('变差', '更差', '增大', '上升', '恶化'))
                if (better and not comparisons[key]['doppler_improved']) or (worse and comparisons[key]['doppler_improved']):
                    warnings.append(f'当前测试路线 Doppler 的 {label.upper()} 改善方向与记录不符，请核对基线对照。')
    if any(item['tool'] == 'compare_nlos' for item in evidence):
        for sentence in re.split(r'[。！？\n]', answer.lower()):
            if any(word in sentence for word in ('论文', '例如', '可以', '建议', '不是', '不按')):
                continue
            if re.search(r'f1[^，,；;\n]{0,8}早停|早停[^，,；;\n]{0,8}f1', sentence):
                warnings.append('当前神经网络检查点/早停依据验证 BCE；验证 F1 用于分类阈值与选模，请核对回答。')
    return list(dict.fromkeys(warnings))
