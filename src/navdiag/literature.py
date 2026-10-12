"""Curated reading recommendations; local retrieval only, no generation or live search."""
from functools import lru_cache
from importlib.resources import files
import json
import re
from urllib.parse import urlparse

from navdiag.knowledge import retrieve_knowledge
from navdiag.knowledge import normalize_query

TOPICS = {'nlos': 'LOS/NLOS 识别', 'preprocessing': '特征与预处理',
          'generalization': '模型比较与泛化', 'positioning': '定位与观测加权',
          'fusion': '因子图与传感器融合', 'maps': '地图与环境辅助',
          'smartphone': '手机 GNSS', 'datasets': '数据与评估'}
LEVELS = {'introductory': '入门', 'intermediate': '进阶', 'advanced': '深入'}


@lru_cache(maxsize=1)
def registry():
    records = json.loads(files('navdiag').joinpath('resources/literature.json').read_text(encoding='utf-8'))['records']
    if len({record['id'] for record in records}) != len(records):
        raise ValueError('Duplicate literature IDs')
    for record in records:
        for url in [record['source_url'], record.get('open_access_url', record['source_url'])]:
            parsed = urlparse(url)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('Literature links must be HTTPS public source URLs')
        learning = record['learning']
        if not set(learning['topics']) <= TOPICS.keys() or learning['difficulty'] not in LEVELS:
            raise ValueError('Invalid learning metadata')
    return records


def reading_list(query: str = '', topic: str = '', difficulty: str = '', limit: int = 5,
                 mode: str | None = None) -> dict:
    if not isinstance(query, str) or len(query) > 2000:
        raise ValueError('请输入不超过 2000 字符的问题。')
    if topic not in ('', *TOPICS) or difficulty not in ('', *LEVELS):
        raise ValueError('未知的文献主题或阅读难度。')
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5:
        raise ValueError('推荐数量应为 1–5。')
    normalized = normalize_query(query).lower()
    routes = {'generalization': ('transformer', '随机森林', '树模型', '泛化', '跨区域', '跨城市'),
              'smartphone': ('手机', 'smartphone', 'android', 'gsdc', 'tdcp'),
              'fusion': ('因子图', 'fgo', 'imu', 'ins', '融合', '卡尔曼', 'ekf', 'pdr'),
              'maps': ('地图', '点云', 'lidar', 'shadow', '反射点'),
              'positioning': ('定位', 'positioning', 'wls', '观测加权', '删星', 'rtk'),
              'datasets': ('数据集', 'dataset', '标签覆盖', '参考真值')}
    inferred = [key for key, terms in routes.items() if any(
        term in normalized if not term.isascii() or len(term) > 4 else
        re.search(r'(?<![a-z])' + re.escape(term) + r'(?![a-z])', normalized) for term in terms)] if not topic else []
    eligible = [record for record in registry() if (not topic or topic in record['learning']['topics'])
                and (not inferred or set(inferred) & set(record['learning']['topics']))
                and (not difficulty or difficulty == record['learning']['difficulty'])]
    query = query.strip()
    metadata = {'query': query, 'library_count': len(registry()), 'filtered_count': len(eligible),
                'topics': TOPICS, 'levels': LEVELS, 'model_api_called': False,
                'inferred_topics': inferred,
                'notice': '按本地资料相关性推荐，排序不是论文质量或可信度评分；摘要核对不等于全文阅读。'}
    if not query:
        papers = sorted(eligible, key=lambda record: (list(LEVELS).index(record['learning']['difficulty']),
                                                     -record['year'], record['id']))
        return {**metadata, 'mode': 'browse', 'papers': papers, 'retrieval_method': None,
                'fallback_reason': None}
    result = retrieve_knowledge(query, limit=limit, mode=mode, source_type='research_summary',
                                document_ids={record['id'] for record in eligible})
    matches = {record['id']: record for record in eligible}
    papers = [{**matches[note['id']], 'retrieval_method': note['retrieval_method'],
               'matched_keywords': [word for word in matches[note['id']]['keywords']
                                    if word.lower() in result['retrieval_query'].lower()]}
              for note in result['notes']]
    return {**metadata, 'mode': 'recommend', 'papers': papers,
            'retrieval_method': result['retrieval_method'], 'fallback_reason': result['fallback_reason']}
